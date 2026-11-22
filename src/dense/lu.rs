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
use super::{block, fail, fence, gather_rows, launch, nth, per_matrix, stack_batch, subgroups, Dims};
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
        for (; i + {last}u < m; i = i + {step}u) {{
            let r0 = base + i * w + c;
{copy_loads}{copy_stores}            if (c == 0u) {{
{copy_cands}            }}
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
        copy_loads = (0..UNROLL).map(|k| format!("            let v{k} = a[r0 + {}u * w];\n", k * g)).collect::<String>(),
        copy_stores = (0..UNROLL).map(|k| format!("            f[r0 + {}u * w] = v{k};\n", k * g)).collect::<String>(),
        copy_cands = (0..UNROLL).map(|k| format!("                if (abs(v{k}) > best) {{ best = abs(v{k}); bi = i + {}u; }}\n", k * g)).collect::<String>(),
        upd_l = (0..UNROLL).map(|k| format!("            let l{k} = f[r0 + {}u * w + j] * rpiv;\n", k * g)).collect::<String>(),
        upd_x = (0..UNROLL).map(|k| format!("                let x{k} = fma(-l{k}, prow, f[r0 + {}u * w + c]);\n", k * g)).collect::<String>(),
        upd_store = (0..UNROLL).map(|k| format!("                f[r0 + {}u * w + c] = x{k};\n", k * g)).collect::<String>(),
        upd_cands = (0..UNROLL).map(|k| format!("                    if (abs(x{k}) > best) {{ best = abs(x{k}); bi = i + {}u; }}\n", k * g)).collect::<String>(),
    )
}

/// Columns of a panel the register kernel factors.
const REG_COLS: usize = 8;
/// Most rows per invocation of the register kernel (`8 * REG_ROWS` registers of data).
const REG_ROWS: usize = 8;

/// How a leaf factors a `[w, m]` matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    /// Workgroup memory, up to 64 x 64.
    Small,
    /// Registers: panels of up to [`REG_COLS`] columns, `rows` rows per invocation of `threads`.
    Registers { threads: usize, rows: usize },
    /// Global memory, panels of up to [`PANEL`] columns.
    Panel,
}

fn leaf_kind(ctx: &crate::Context, m: usize) -> Leaf {
    if m <= SMALL {
        return Leaf::Small;
    }
    let limits = &ctx.info().limits;
    let threads = (limits.max_workgroup_invocations.min(limits.max_workgroup_size[0]) as usize).min(1024) / 32 * 32;
    if threads >= 128 && m <= threads * REG_ROWS && std::env::var_os("LUMA_LINALG_NOREG").is_none() {
        return Leaf::Registers { threads, rows: m.div_ceil(threads).next_power_of_two() };
    }
    Leaf::Panel
}

/// The widest matrix (in columns) a leaf of `m` rows factors in one launch.
fn leaf_width(ctx: &crate::Context, m: usize) -> usize {
    match leaf_kind(ctx, m) {
        Leaf::Small => SMALL,
        Leaf::Registers { .. } => REG_COLS,
        Leaf::Panel => PANEL,
    }
}

/// Where the recursion splits `w` columns: about half, at a multiple of 8 for narrow panels.
fn split_cols(w: usize) -> usize {
    if w <= 64 { (w / 2).next_multiple_of(REG_COLS).min(w - 1).max(1) } else { split(w) }
}

/// Register kernel: a panel of up to 8 columns, row `t + k * T` of the invocation `t` held in
/// registers for its whole factorisation (no traffic to memory between columns). Per column: the
/// pivot by a tree reduction in workgroup memory, the pivot row and the displaced row exchanged
/// through workgroup memory, the update in registers. The loops over columns and rows are
/// generated unrolled so that every register is indexed statically.
fn reg_source(fence: &str, threads: usize, rows: usize, subgroups: bool) -> String {
    let t = threads;
    let mut body = String::new();
    // Load.
    for k in 0..rows {
        body += &format!("    let row{k} = t + {}u;\n", k * t);
        body += &format!("    pr[{k}] = row{k};\n    if (row{k} < m) {{\n");
        for c in 0..REG_COLS {
            body += &format!("        if ({c}u < w) {{ x[{}] = a[base + row{k} * w + {c}u]; }}\n", k * REG_COLS + c);
        }
        body += "    }\n";
    }
    for j in 0..REG_COLS {
        body += &format!("    if ({j}u < min(m, w)) {{\n        var best = -1.0;\n        var bi = {j}u;\n");
        for k in 0..rows {
            body += &format!("        if (row{k} >= {j}u && row{k} < m) {{ let av = abs(x[{}]); if (av > best) {{ best = av; bi = row{k}; }} }}\n", k * REG_COLS + j);
        }
        if subgroups {
            body += &format!(
                "        let bm = subgroupMax(best);
        let bidx = subgroupMin(select(0xffffffffu, bi, best == bm));
        if (lane == 0u) {{ rv[sgid] = bm; ri[sgid] = bidx; }}
        workgroupBarrier();
        var pvv = -2.0;
        var pv = 0xffffffffu;
        for (var q = 0u; q < nsg; q = q + 1u) {{
            let ov = rv[q];
            let oi = ri[q];
            if (ov > pvv || (ov == pvv && oi < pv)) {{ pvv = ov; pv = oi; }}
        }}
"
            );
        } else {
            body += &format!(
                "        rv[t] = best;
        ri[t] = bi;
        workgroupBarrier();
        for (var h = {half}u; h > 0u; h = h >> 1u) {{
            if (t < h) {{
                let ov = rv[t + h];
                let oi = ri[t + h];
                if (ov > rv[t] || (ov == rv[t] && oi < ri[t])) {{ rv[t] = ov; ri[t] = oi; }}
            }}
            workgroupBarrier();
        }}
        let pv = ri[0];
",
                half = t / 2
            );
        }
        for k in 0..rows {
            body += &format!("        if (row{k} == pv) {{\n");
            for c in 0..REG_COLS {
                body += &format!("            sp[{c}] = x[{}];\n", k * REG_COLS + c);
            }
            body += &format!("            spi = pr[{k}];\n        }}\n        if (row{k} == {j}u) {{\n");
            for c in 0..REG_COLS {
                body += &format!("            sj[{c}] = x[{}];\n", k * REG_COLS + c);
            }
            body += &format!("            sji = pr[{k}];\n        }}\n");
        }
        body += "        workgroupBarrier();\n";
        for k in 0..rows {
            body += &format!("        if (row{k} == {j}u) {{\n");
            for c in 0..REG_COLS {
                body += &format!("            x[{}] = sp[{c}];\n", k * REG_COLS + c);
            }
            body += &format!("            pr[{k}] = spi;\n        }} else if (row{k} == pv) {{\n");
            for c in 0..REG_COLS {
                body += &format!("            x[{}] = sj[{c}];\n", k * REG_COLS + c);
            }
            body += &format!("            pr[{k}] = sji;\n        }}\n");
        }
        body += &format!("        let piv = sp[{j}];\n        let rpiv = select(0.0, 1.0 / piv, piv != 0.0);\n");
        for k in 0..rows {
            body += &format!("        if (row{k} > {j}u && row{k} < m) {{\n            let l = x[{}] * rpiv;\n            x[{}] = l;\n", k * REG_COLS + j, k * REG_COLS + j);
            for c in j + 1..REG_COLS {
                body += &format!("            if ({c}u < w) {{ x[{}] = fma(-l, sp[{c}], x[{}]); }}\n", k * REG_COLS + c, k * REG_COLS + c);
            }
            body += "        }\n";
        }
        body += "        workgroupBarrier();\n    }\n";
    }
    // Store.
    for k in 0..rows {
        body += &format!("    if (row{k} < m) {{\n        perm[pb + row{k}] = pr[{k}];\n");
        for c in 0..REG_COLS {
            body += &format!("        if ({c}u < w) {{ f[base + row{k} * w + {c}u] = x[{}]; }}\n", k * REG_COLS + c);
        }
        body += "    }\n";
    }
    let _ = fence;
    format!(
        "struct Params {{ m: u32, w: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> f: array<f32>;
@group(1) @binding(2) var<storage, read_write> perm: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> rv: array<f32, {t}>;
var<workgroup> ri: array<u32, {t}>;
var<workgroup> sp: array<f32, {cols}>;
var<workgroup> sj: array<f32, {cols}>;
var<workgroup> spi: u32;
var<workgroup> sji: u32;

@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32{sg_args}) {{
    let mi = wid.x + wid.y * nwg.x;
    let m = select(0u, p.m, mi < p.batch);
    let w = p.w;
    let base = mi * p.m * w;
    let pb = mi * p.m;
{sg_setup}    var x: array<f32, {n}>;
    var pr: array<u32, {rows}>;
{body}}}
",
        cols = REG_COLS,
        n = rows * REG_COLS,
        sg_args = if subgroups { ",\n    @builtin(subgroup_invocation_id) lane: u32,\n    @builtin(subgroup_size) sg: u32" } else { "" },
        sg_setup = if subgroups { format!("    let sgid = t / sg;\n    let nsg = {t}u / sg;\n") } else { String::new() },
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
    let result = match leaf_kind(ctx, d.rows) {
        Leaf::Small => launch(ctx, "lu_small", "lu_small", small_source, per_matrix(ctx, d.batch), &args),
        Leaf::Registers { threads, rows } if d.cols <= REG_COLS => launch(ctx, &format!("lu_regs/{threads}/{rows}/{}", subgroups(ctx)), "lu_regs", || reg_source(fence(ctx), threads, rows, subgroups(ctx)), per_matrix(ctx, d.batch), &args),
        _ => {
            let threads = panel_threads(ctx, d.rows);
            launch(ctx, &format!("lu_panel/{threads}/{}", fence(ctx).len()), "lu_panel", || panel_source(fence(ctx), threads), per_matrix(ctx, d.batch), &args)
        }
    };
    match result {
        Ok(()) => Lu { lu: f, perm },
        Err(e) => poisoned(e),
    }
}

/// The recursion over one `[w, m]` matrix, `m >= w`.
fn recurse(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let [w, m, ..] = a.shape().dims();
    if w <= leaf_width(a.context(), m) {
        return leaf(a, Dims { rows: m, cols: w, batch: 1 }, op, loc);
    }
    let n1 = split_cols(w);
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
        Ok(d) => {
            // Wide: factor the square left block, then U12 = L11⁻¹ (P A12).
            let left = factor(&a.slice(0, 0..d.rows), op, loc);
            let right = gather_rows(&a.slice(0, d.rows..d.cols), &left.perm, loc);
            let u12 = lower_left(&left.lu, &right, true, op, loc);
            return Lu { lu: concat(0, [&left.lu, &u12]), perm: left.perm };
        }
        Err(e) => return Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, Shape::from(1), Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) },
    };
    if d.cols <= leaf_width(ctx, d.rows) {
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
