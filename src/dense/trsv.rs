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
//! Block `b` is left-looking: for every block `j` before it, it waits for the flag of `j`, reads
//! the solved `x_j` (published through atomics, a status word after a fence: the protocol of the
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
    batch == 1 && k <= KMAX && n > super::tri::BASE && ctx.tuning().lookback_scan && std::env::var("LUMA_TRSV").ok().is_none_or(|v| v != "0")
}

fn source(kc: usize, inv: bool) -> String {
    let ap = if kc == 1 { 1 } else { kc + 1 };
    let cpt = (kc / SEG).max(1);
    // The loads of a tile of `L` (columns from `c0`) into registers; the stores of those to workgroup memory.
    let mut decls = String::new();
    let mut loads = String::new();
    let mut stores = String::new();
    for u in 0..PER {
        let (a, q) = (format!("select(e{u} % {NB}u, e{u} / {NB}u, jfast)"), format!("select(e{u} / {NB}u, e{u} % {NB}u, jfast)"));
        decls += &format!("    var v{u} = 0.0;\n    let e{u} = t + {}u;\n", u * T);
        loads += &format!("        if (r0 + {a} < n && c0 + {q} < n) {{ v{u} = ldl(r0 + {a}, c0 + {q}); }} else {{ v{u} = 0.0; }}\n");
        stores += &format!("        lt[{a} * {LP}u + {q}] = v{u};\n");
    }
    let mut zero = String::new();
    let mut fma = String::new();
    for c in 0..kc {
        zero += &format!("    var p{c} = 0.0;\n");
        fma += &format!("            p{c} = fma(lq, xs[q * {ap}u + {c}u], p{c});\n");
    }
    // Partial sums folded into `acc`: segment 0 starts it from the right-hand side.
    let mut fold = String::new();
    for s in 0..SEG {
        fold += &format!("    if (sg == {s}u && rr < rows) {{\n");
        for c in 0..kc {
            if s == 0 {
                fold += &format!("        acc[rr * {ap}u + {c}u] = ldb(r0 + rr, {c}u) - p{c};\n");
            } else {
                fold += &format!("        acc[rr * {ap}u + {c}u] = acc[rr * {ap}u + {c}u] - p{c};\n");
            }
        }
        fold += "    }\n    workgroupBarrier();\n";
    }
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
        xload += &format!("        let w{u} = atomicLoad(&st[xb + j * {words}u + t + {}u]);\n", u * T);
    }
    for u in 0..steps {
        xload += &format!("        {}{{ let e = t + {}u; xs[(e / {kc}u) * {ap}u + e % {kc}u] = bitcast<f32>(w{u}); }}\n", guard(), u * T);
    }
    let mut publish = String::new();
    for u in 0..steps {
        publish += &format!("    {{
        let e = t + {}u;
        if (e < {words}u && e / {kc}u < rows) {{ atomicStore(&st[xb + blk * {words}u + e], bitcast<u32>(xs[(e / {kc}u) * {ap}u + e % {kc}u])); }}
    }}\n", u * T);
    }
    // Inverse mode: the diagonal block is read first and inverted while the workgroup waits for the blocks before it
    // (a column of the inverse an invocation, `dt` the packed triangle, `iv` the inverse), then `x = iv acc` is
    // a product, not 64 dependent steps with a barrier each (the same trick as the inverted diagonal blocks of
    // MAGMA's `trsm`, Haidar et al.; here the inversion is off the critical path).
    let mut dstores = String::new();
    for u in 0..PER {
        let (a, q) = (format!("select(e{u} % {NB}u, e{u} / {NB}u, jfast)"), format!("select(e{u} / {NB}u, e{u} % {NB}u, jfast)"));
        dstores += &format!("    if ({q} <= {a}) {{ dt[{a} * ({a} + 1u) / 2u + {q}] = v{u}; }}\n");
    }
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
    let pre = if inv {
        format!(
            "    c0 = r0;
{loads}{dstores}    workgroupBarrier();
    if (t < rows) {{ rd[t] = select(1.0 / dt[t * (t + 1u) / 2u + t], 1.0, p.unit != 0u); }}
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
    workgroupBarrier();
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
    let extra_decls = if inv { format!("var<workgroup> dt: array<f32, {}>;\nvar<workgroup> iv: array<f32, {}>;\n", NB * (NB + 1) / 2, NB * LP) } else { String::new() };
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
    let fb = 1u + cg * p.nblocks;
    let xb = 1u + p.ncg * p.nblocks + cg * n * {kc}u;
    let r0 = blk * {NB}u;
    let rows = min({NB}u, n - r0);
    let rr = t % {NB}u;
    let sg = t / {NB}u;
    let jfast = abs(lv.stride.x) == 1;
{decls}    var c0 = 0u;
{pre}{zero}    for (var j = 0u; j < blk; j = j + 1u) {{
        workgroupBarrier();
{stores}{next_tile}        if (t == 0u) {{
            loop {{
                if (atomicLoad(&st[fb + j]) != 0u) {{ break; }}
            }}
        }}
        storageBarrier();
{xload}        workgroupBarrier();
        for (var q = sg * {SEGW}u; q < sg * {SEGW}u + {SEGW}u; q = q + 1u) {{
            let lq = lt[rr * {LP}u + q];
{fma}        }}
    }}
{tail}{publish}    storageBarrier();
    if (t == 0u) {{ atomicStore(&st[fb + blk], 1u); }}
    for (var e = t; e < rows * {kc}u; e = e + {T}u) {{
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
    let setup = (Array::<f32>::alloc(ctx, b.shape()), Array::<u32>::zeros_on(ctx, 1 + ncg * nblocks + ncg * n * kc));
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
