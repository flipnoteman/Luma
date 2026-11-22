//! Triangular solves for a few right-hand sides in one launch.
//!
//! `L X = B` with `L` lower triangular (a view: any of the flipped and transposed forms of
//! [`super::tri`]) and at most [`KMAX`] columns in `B`. The blocked solves of `tri.rs` cost two
//! launches and a product a block of rows, a chain of about 30 us a block on an RTX 3090, 4.5 ms
//! for a 4096 x 4096 system; this kernel streams the triangle once at the speed of the memory
//! and leaves only the unavoidable chain, one block of rows after the other.
//!
//! One workgroup a block of [`NB`] rows, taken in order from an atomic ticket (so the blocks a
//! workgroup waits for have all started, whatever the order the device launches workgroups in).
//! Block `b` is left-looking: for every block `j` before it, it waits for the words of `j`, reads
//! the solved `x_j` (published through atomics, every word complemented so that 0 means "not there yet":
//! the reader spins on the words it needs, no status flag, no fence; the ticket is the one of the
//! decoupled look-back of [`crate::scan`]) and adds `L[b, j] x_j` to its partial sums; the tiles of
//! `L` do not depend on anything, so the next one is already in registers while the workgroup waits.
//! Then the block's own triangle is solved by substitution in workgroup memory, `x_b` is published.
//! Only where the device orders memory across workgroups ([`Tuning::lookback_scan`]; CUDA).

use std::panic::Location;

use super::{fail, launch};
use crate::array::Array;
use crate::kernel::{uniform, view, KernelArg};

/// Rows a workgroup solves.
const NB: usize = 64;
/// Invocations of a workgroup: `NB` rows by [`SEG`] segments of the columns of a tile.
const T: usize = 256;
/// Segments of the columns of a tile (the partial sums of a row are added up at the end).
const SEG: usize = T / NB;
/// Columns of a tile an invocation multiplies.
const SEGW: usize = NB / SEG;
/// Elements of a tile an invocation loads.
const PER: usize = NB * NB / T;
/// Most right-hand sides this kernel takes.
pub(crate) const KMAX: usize = 32;
/// Columns of the right-hand sides a workgroup takes (`LUMA_TRSV_KC`): more are solved by workgroups of their own,
/// which read the triangle again but do not lengthen the chain from block to block.
fn group_columns(kp: usize) -> usize {
    let kc = std::env::var("LUMA_TRSV_KC").ok().and_then(|v| v.parse().ok()).filter(|kc| [1, 2, 4, 8, 16, 32].contains(kc)).unwrap_or(8);
    kp.min(kc)
}
/// Pitch of a tile in workgroup memory.
const LP: usize = NB + 1;

#[derive(encase::ShaderType)]
struct Params {
    n: u32,
    k: u32,
    unit: u32,
    nblocks: u32,
    ncg: u32,
}

/// Whether `L X = B` of `n` rows and `k` columns goes to this kernel.
pub(crate) fn applies(ctx: &crate::Context, n: usize, k: usize, batch: usize) -> bool {
    // The tile, the packed triangle and the inverse of inverse mode, the right-hand sides and the partial sums.
    let need = 4 * (2 * NB * LP + NB * (NB + 1) / 2 + 2 * NB * 9) + 256;
    batch == 1 && ctx.info().limits.max_workgroup_storage_bytes as usize >= need && k <= KMAX && n > super::tri::BASE && ctx.tuning().lookback_scan && std::env::var("LUMA_TRSV").ok().is_none_or(|v| v != "0")
}

/// The pieces that read a tile of `L` through the view into the registers `v0..` of an invocation (columns from `c0`,
/// rows from `r0`, the elements of a tile `T` apart so that the warps read along whichever axis is contiguous) and
/// put them into `lt` (pitch [`LP`]) or, the triangle only, into the packed `dt`.
fn tile_text() -> (String, String, String, String) {
    let (mut decls, mut loads, mut stores, mut dstores) = (String::new(), String::new(), String::new(), String::new());
    for u in 0..PER {
        let (a, q) = (format!("select(e{u} % {NB}u, e{u} / {NB}u, jfast)"), format!("select(e{u} / {NB}u, e{u} % {NB}u, jfast)"));
        decls += &format!("    var v{u} = 0.0;\n    let e{u} = t + {}u;\n", u * T);
        loads += &format!("        if (r0 + {a} < n && c0 + {q} < n) {{ v{u} = ldl(r0 + {a}, c0 + {q}); }} else {{ v{u} = 0.0; }}\n");
        stores += &format!("        lt[{a} * {LP}u + {q}] = v{u};\n");
        dstores += &format!("    if ({q} <= {a}) {{ dt[{a} * ({a} + 1u) / 2u + {q}] = v{u}; }}\n");
    }
    (decls, loads, stores, dstores)
}

/// Inverts the lower triangle `dt` (packed, `rows` of it, the reciprocals of its diagonal in `rd`, 1 for `p.unit`)
/// into `iv` (pitch [`LP`], below the diagonal and on it): an invocation a column of the inverse, `y_i = -(sum_q
/// l_iq y_q) / l_ii`, two chains of multiply-adds. Ends with a barrier.
fn invert_text() -> String {
    format!(
        "    if (t < rows) {{ rd[t] = select(1.0 / dt[t * (t + 1u) / 2u + t], 1.0, p.unit != 0u); }}
    workgroupBarrier();
    if (t < rows) {{
        iv[t * {LP}u + t] = rd[t];
        for (var i = t + 1u; i < rows; i = i + 1u) {{
            let o = i * (i + 1u) / 2u;
            var s0 = 0.0;
            var s1 = 0.0;
            var q = t;
            for (; q + 1u < i; q = q + 2u) {{
                s0 = fma(dt[o + q], iv[q * {LP}u + t], s0);
                s1 = fma(dt[o + q + 1u], iv[(q + 1u) * {LP}u + t], s1);
            }}
            if (q < i) {{ s0 = fma(dt[o + q], iv[q * {LP}u + t], s0); }}
            iv[i * {LP}u + t] = -(s0 + s1) * rd[i];
        }}
    }}
"
    )
}

fn source(kc: usize, inv: bool) -> String {
    let ap = if kc == 1 { 1 } else { kc + 1 };
    let cpt = (kc / SEG).max(1);
    let (decls, loads, stores, dstores) = tile_text();
    let mut zero = String::new();
    let mut fma = String::new();
    for c in 0..kc {
        zero += &format!("    var p{c} = 0.0;\n");
        fma += &format!("            p{c} = fma(lq, xs[q * {ap}u + {c}u], p{c});\n");
    }
    // The partial sums of the segments go to `red` (the tile buffer in inverse mode: it is free by then), and
    // every entry of `acc` starts from the right-hand side and takes all of them away.
    let red = if inv { "lt" } else { "rs" };
    let mut fold = String::new();
    for c in 0..kc {
        fold += &format!("    {red}[sg * {} + rr * {ap}u + {c}u] = p{c};\n", NB * ap);
    }
    fold += &format!(
        "    workgroupBarrier();
    for (var e = t; e < {NB}u * {kc}u; e = e + {T}u) {{
        let r = e / {kc}u;
        let c = e % {kc}u;
        if (r < rows) {{
            var s = ldb(r0 + r, c);
            for (var g = 0u; g < {SEG}u; g = g + 1u) {{ s = s - {red}[g * {} + r * {ap}u + c]; }}
            acc[r * {ap}u + c] = s;
        }}
    }}
    workgroupBarrier();
",
        NB * ap
    );
    let mut solve = String::new();
    for u in 0..cpt {
        solve += &format!(
            "        {{
            let c = sg * {cpt}u + {u}u;
            if (c < {kc}u) {{
                let xi = acc[i * {ap}u + c] * ri;
                if (rr > i) {{ acc[rr * {ap}u + c] = fma(-li, xi, acc[rr * {ap}u + c]); }}
                if (rr == i) {{ xs[i * {ap}u + c] = xi; }}
            }}
        }}\n"
        );
    }
    // The solved block `x_j` of the column group, `NB * kc` words, read `T` at a time with all the loads first.
    let words = NB * kc;
    let steps = words.div_ceil(T);
    let guard = || if words >= T { String::new() } else { format!("if (t < {words}u) ") };
    let mut xload = String::new();
    for u in 0..steps {
        xload += &format!(
            "        {}{{
            let e = t + {}u;
            var w = 0u;
            loop {{
                w = atomicLoad(&st[xb + j * {words}u + e]);
                if (w != 0u) {{ break; }}
            }}
            xs[(e / {kc}u) * {ap}u + e % {kc}u] = bitcast<f32>(~w);
        }}\n",
            guard(),
            u * T
        );
    }
    let mut publish = String::new();
    for u in 0..steps {
        // A solved value is stored as the complement of its bits (never 0, which means not yet there: only
        // the NaN of all ones would give 0, and it is stored as another NaN): no status word, no fence.
        publish += &format!("    {{
        let e = t + {}u;
        if (e < {words}u && e / {kc}u < rows) {{
            let w = ~bitcast<u32>(xs[(e / {kc}u) * {ap}u + e % {kc}u]);
            atomicStore(&st[xb + blk * {words}u + e], select(w, 1u, w == 0u));
        }}
    }}\n", u * T);
    }
    // Inverse mode: the diagonal block is read first and inverted while the workgroup waits for the blocks before it
    // (a column of the inverse an invocation, `dt` the packed triangle, `iv` the inverse), then `x = iv acc` is
    // a product, not 64 dependent steps with a barrier each (the same trick as the inverted diagonal blocks of
    // MAGMA's `trsm`, Haidar et al.; here the inversion is off the critical path).
    let mut mv = String::new();
    for u in 0..cpt {
        mv += &format!(
            "    {{
        let c = sg * {cpt}u + {u}u;
        if (c < {kc}u && rr < rows) {{
            var s0 = 0.0;
            for (var q = 0u; q <= rr; q = q + 1u) {{ s0 = fma(iv[rr * {LP}u + q], acc[q * {ap}u + c], s0); }}
            xs[rr * {ap}u + c] = s0;
        }}
    }}\n"
        );
    }
    let invert = invert_text();
    let pre = if inv {
        format!(
            "    c0 = r0;
{loads}{dstores}    workgroupBarrier();
{invert}    workgroupBarrier();
    c0 = 0u;
{loads}"
        )
    } else {
        format!("    c0 = select(r0, 0u, blk > 0u);\n{loads}")
    };
    let next_tile = if inv {
        format!("        if (j + 1u < blk) {{\n            c0 = (j + 1u) * {NB}u;\n{loads}        }}\n")
    } else {
        format!("        c0 = select(r0, (j + 1u) * {NB}u, j + 1u < blk);\n{loads}")
    };
    let tail = if inv {
        format!("    workgroupBarrier();\n{fold}{mv}    workgroupBarrier();\n")
    } else {
        format!(
            "    workgroupBarrier();
{stores}    workgroupBarrier();
    if (t < rows) {{ rd[t] = select(1.0 / lt[t * {LP}u + t], 1.0, p.unit != 0u); }}
{fold}    for (var i = 0u; i < rows; i = i + 1u) {{
        let ri = rd[i];
        let li = lt[rr * {LP}u + i];
{solve}        workgroupBarrier();
    }}
"
        )
    };
    let extra_decls = if inv { format!("var<workgroup> dt: array<f32, {}>;\nvar<workgroup> iv: array<f32, {}>;\n", NB * (NB + 1) / 2, NB * LP) } else { format!("var<workgroup> rs: array<f32, {}>;\n", SEG * NB * ap) };
    format!(
        "struct Params {{ n: u32, k: u32, unit: u32, nblocks: u32, ncg: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> bv: LumaView;
@group(1) @binding(4) var<storage, read_write> o: array<f32>;
@group(1) @binding(5) var<storage, read_write> st: array<atomic<u32>>;
@group(1) @binding(6) var<uniform> p: Params;

var<workgroup> lt: array<f32, {lsize}>;
var<workgroup> xs: array<f32, {xsize}>;
var<workgroup> acc: array<f32, {xsize}>;
var<workgroup> rd: array<f32, {NB}>;
var<workgroup> ticket: u32;
{extra_decls}
fn ldl(i: u32, j: u32) -> f32 {{ return l[luma_view_idx3(lv, j, i, 0u)]; }}
// Column `c` of the group's columns of row `i` of the right-hand side (0 past the last one).
fn ldb(i: u32, c: u32) -> f32 {{ if (col0 + c < p.k) {{ return b[luma_view_idx3(bv, col0 + c, i, 0u)]; }} return 0.0; }}

var<private> col0: u32;

@compute @workgroup_size({T})
fn main(@builtin(local_invocation_index) t: u32) {{
    if (t == 0u) {{ ticket = atomicAdd(&st[0], 1u); }}
    for (var e = t; e < {xsize}u; e = e + {T}u) {{ xs[e] = 0.0; }}
    workgroupBarrier();
    let blk = ticket / p.ncg;
    let cg = ticket % p.ncg;
    col0 = cg * {kc}u;
    let n = p.n;
    let k = p.k;
    let xb = 1u + cg * n * {kc}u;
    let r0 = blk * {NB}u;
    let rows = min({NB}u, n - r0);
    let rr = t % {NB}u;
    let sg = t / {NB}u;
    let jfast = abs(lv.stride.x) == 1;
{decls}    var c0 = 0u;
{pre}{zero}    for (var j = 0u; j < blk; j = j + 1u) {{
        workgroupBarrier();
{stores}{next_tile}{xload}        workgroupBarrier();
        for (var q = sg * {SEGW}u; q < sg * {SEGW}u + {SEGW}u; q = q + 1u) {{
            let lq = lt[rr * {LP}u + q];
{fma}        }}
    }}
{tail}{publish}    for (var e = t; e < rows * {kc}u; e = e + {T}u) {{
        let c = e % {kc}u;
        if (col0 + c < k) {{ o[(r0 + e / {kc}u) * k + col0 + c] = xs[(e / {kc}u) * {ap}u + c]; }}
    }}
}}
",
        lsize = NB * LP,
        xsize = NB * ap,
    )
}

/// `L X = B` for one `[n, n]` lower triangle and `[k, n]` right-hand sides, `k <= KMAX`; see the module doc.
pub(crate) fn lower_left(l: &Array<f32>, b: &Array<f32>, unit: bool, inverse: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let n = l.shape().dims()[1];
    let k = b.shape().dims()[0];
    let nblocks = n.div_ceil(NB);
    let kc = group_columns(k.next_power_of_two());
    let ncg = k.div_ceil(kc);
    let setup = (Array::<f32>::alloc(ctx, b.shape()), Array::<u32>::zeros_on(ctx, 1 + ncg * n * kc));
    let (mut out, mut st) = match setup {
        (Ok(o), Ok(s)) => (o, s),
        (Err(e), _) | (_, Err(e)) => return fail(ctx, b.shape(), e, op, loc),
    };
    let params = Params { n: n as u32, k: k as u32, unit: u32::from(unit), nblocks: nblocks as u32, ncg: ncg as u32 };
    let args: [KernelArg<'_>; 5] = [view(l), view(b), (&mut out).into(), (&mut st).into(), uniform(&params)];
    match launch(ctx, &format!("trsv/{kc}/{}", u8::from(inverse)), "trsv", || source(kc, inverse), [(nblocks * ncg) as u32, 1, 1], &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, b.shape(), e, op, loc),
    }
}

fn invert_source() -> String {
    let (decls, loads, _, dstores) = tile_text();
    let invert = invert_text();
    let mut out = String::new();
    for u in 0..PER {
        out += &format!("    {{
        let e = t + {}u;
        let a = e / {NB}u;
        let q = e % {NB}u;
        iw[(r0 + a) * {NB}u + q] = select(0.0, iv[a * {LP}u + q], q <= a && a < rows);
    }}\n", u * T);
    }
    format!(
        "struct Params {{ n: u32, unit: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read_write> iw: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;

var<workgroup> dt: array<f32, {dsize}>;
var<workgroup> iv: array<f32, {isize}>;
var<workgroup> rd: array<f32, {NB}>;

fn ldl(i: u32, j: u32) -> f32 {{ return l[luma_view_idx3(lv, j, i, 0u)]; }}

@compute @workgroup_size({T})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = p.n;
    let r0 = wid.x * {NB}u;
    let c0 = r0;
    let rows = min({NB}u, n - r0);
    let jfast = abs(lv.stride.x) == 1;
{decls}{loads}{dstores}    workgroupBarrier();
{invert}    workgroupBarrier();
{out}}}
",
        dsize = NB * (NB + 1) / 2,
        isize = NB * LP,
    )
}

#[derive(encase::ShaderType)]
struct InvParams {
    n: u32,
    unit: u32,
}

/// The inverses of the diagonal blocks of [`NB`] rows of the `[n, n]` lower triangle `l`, stacked: `[NB, n.next_multiple_of(NB)]`,
/// block `i` is rows `i * NB ..` (the last one's inverse is in its top left corner, zero elsewhere). One launch.
pub(crate) fn invert_blocks(l: &Array<f32>, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let n = l.shape().dims()[1];
    let nblocks = n.div_ceil(NB);
    let shape = crate::shape::Shape::from([NB, nblocks * NB]);
    let mut out = match Array::<f32>::alloc(ctx, shape) {
        Ok(o) => o,
        Err(e) => return fail(ctx, shape, e, op, loc),
    };
    let params = InvParams { n: n as u32, unit: u32::from(unit) };
    let args: [KernelArg<'_>; 3] = [view(l), (&mut out).into(), uniform(&params)];
    match launch(ctx, "trsv_invert", "trsv_invert", invert_source, [nblocks as u32, 1, 1], &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, shape, e, op, loc),
    }
}

/// Whether the device has the workgroup memory of [`invert_blocks`] and [`apply_inverse`] (two tiles).
pub(crate) fn blocks_fit(ctx: &crate::Context) -> bool {
    ctx.info().limits.max_workgroup_storage_bytes as usize >= 2 * 4 * NB * LP + 256
}

fn apply_source() -> String {
    let mut acc = String::new();
    let mut fma = String::new();
    let mut store = String::new();
    for i in 0..4 {
        for j in 0..4 {
            acc += &format!("    var s{i}{j} = 0.0;\n");
            fma += &format!("        s{i}{j} = fma(a{i}, b{j}, s{i}{j});\n");
            store += &format!("    if (ty * 4u + {i}u < rows && c0 + tx * 4u + {j}u < k) {{ o[(row0 + ty * 4u + {i}u) * k + c0 + tx * 4u + {j}u] = s{i}{j}; }}\n");
        }
    }
    let loads: String = (0..4).map(|i| format!("        let a{i} = iv[(ty * 4u + {i}u) * {LP}u + q];\n")).collect();
    let bloads: String = (0..4).map(|j| format!("        let b{j} = bs[q * {LP}u + tx * 4u + {j}u];\n")).collect();
    format!(
        "struct Params {{ rows: u32, k: u32, row0: u32, pad: u32 }}
@group(1) @binding(0) var<storage, read> inv: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<uniform> bv: LumaView;
@group(1) @binding(3) var<storage, read_write> o: array<f32>;
@group(1) @binding(4) var<uniform> p: Params;

var<workgroup> iv: array<f32, {size}>;
var<workgroup> bs: array<f32, {size}>;

@compute @workgroup_size({T})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let rows = p.rows;
    let k = p.k;
    let row0 = p.row0;
    let c0 = wid.x * {NB}u;
    for (var e = t; e < {NB}u * {NB}u; e = e + {T}u) {{
        let r = e / {NB}u;
        let q = e % {NB}u;
        iv[r * {LP}u + q] = select(0.0, inv[(row0 + r) * {NB}u + q], r < rows && q < rows);
        // The right-hand side tile, along whichever axis of the view the warp should run.
        let cc = e % {NB}u;
        let qq = e / {NB}u;
        var v = 0.0;
        if (qq < rows && c0 + cc < k) {{ v = b[luma_view_idx3(bv, c0 + cc, qq, 0u)]; }}
        bs[qq * {LP}u + cc] = v;
    }}
    workgroupBarrier();
    let ty = t / 16u;
    let tx = t % 16u;
{acc}    for (var q = 0u; q < {NB}u; q = q + 1u) {{
{loads}{bloads}{fma}    }}
{store}}}
",
        size = NB * LP,
    )
}

#[derive(encase::ShaderType)]
struct ApplyParams {
    rows: u32,
    k: u32,
    row0: u32,
    pad: u32,
}

/// Rows `row0 ..` of `out` (`[k, n]`, row `i` at `i * k`) `= inv_block(row0) * b`, `b` is `[k, rows]` (a view of any strides) and
/// `invs` the output of [`invert_blocks`]: a block of at most [`NB`] rows of a triangular solve in one launch.
pub(crate) fn apply_inverse(invs: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, row0: usize) -> Result<(), crate::error::LumaError> {
    let ctx = invs.context();
    let [k, rows, ..] = b.shape().dims();
    let params = ApplyParams { rows: rows as u32, k: k as u32, row0: row0 as u32, pad: 0 };
    let args: [KernelArg<'_>; 4] = [invs.into(), view(b), out.into(), uniform(&params)];
    launch(ctx, "trsv_apply", "trsv_apply", apply_source, [k.div_ceil(NB) as u32, 1, 1], &args)
}
