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
//!
//! Two ways to solve a block's own triangle: substitution (64 dependent steps, a barrier each: about
//! 3 us of the 6.3 us a block adds to the chain on an RTX 3090), or, for the solves inside `solve` and
//! `solve_spd` ([`super::tri::lower_left_factor`]), by the explicit inverse of the block, computed
//! while the workgroup waits for the blocks before it (a column of the inverse an invocation) and
//! applied by one product: the chain is then 2.8 us a block (4096 rows, 1 column: 0.18 ms, ArrayFire's
//! `trsm` 0.2 ms). The idea is that of the inverted diagonal blocks of MAGMA's `trsm` (BSD-3, Haidar,
//! Dong, Tomov et al. on batched `trsm` for GPUs); only the idea is taken, the kernel is written here.
//! The inverse of a block is not backward stable for an ill-conditioned triangle, so `solve_triangular` of
//! arbitrary data does not use it (`LUMA_TRSV_INV=1` forces it, `LUMA_TRSV=0` turns the kernel off,
//! `LUMA_TRSV_KC` sets the columns a workgroup takes, 8 by default: more columns are more workgroups
//! per block, which read the triangle again but do not lengthen the chain).

use std::panic::Location;

use super::{fail, launch_unchecked};
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
            for (; q + 8u <= i; q = q + 8u) {{
                let d0 = dt[o + q];
                let d1 = dt[o + q + 1u];
                let d2 = dt[o + q + 2u];
                let d3 = dt[o + q + 3u];
                let d4 = dt[o + q + 4u];
                let d5 = dt[o + q + 5u];
                let d6 = dt[o + q + 6u];
                let d7 = dt[o + q + 7u];
                let y0 = iv[q * {LP}u + t];
                let y1 = iv[(q + 1u) * {LP}u + t];
                let y2 = iv[(q + 2u) * {LP}u + t];
                let y3 = iv[(q + 3u) * {LP}u + t];
                let y4 = iv[(q + 4u) * {LP}u + t];
                let y5 = iv[(q + 5u) * {LP}u + t];
                let y6 = iv[(q + 6u) * {LP}u + t];
                let y7 = iv[(q + 7u) * {LP}u + t];
                s0 = fma(d0, y0, s0);
                s1 = fma(d1, y1, s1);
                s0 = fma(d2, y2, s0);
                s1 = fma(d3, y3, s1);
                s0 = fma(d4, y4, s0);
                s1 = fma(d5, y5, s1);
                s0 = fma(d6, y6, s0);
                s1 = fma(d7, y7, s1);
            }}
            for (; q < i; q = q + 1u) {{ s0 = fma(dt[o + q], iv[q * {LP}u + t], s0); }}
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
        for w in 0..4 {
            fma += &format!("            p{c} = fma(lq{w}, xs[(q + {w}u) * {ap}u + {c}u], p{c});\n");
        }
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
            var s1 = 0.0;
            var q = 0u;
            for (; q + 8u <= rr + 1u; q = q + 8u) {{
                let l0 = iv[rr * {LP}u + q];
                let l1 = iv[rr * {LP}u + q + 1u];
                let l2 = iv[rr * {LP}u + q + 2u];
                let l3 = iv[rr * {LP}u + q + 3u];
                let l4 = iv[rr * {LP}u + q + 4u];
                let l5 = iv[rr * {LP}u + q + 5u];
                let l6 = iv[rr * {LP}u + q + 6u];
                let l7 = iv[rr * {LP}u + q + 7u];
                let x0 = acc[q * {ap}u + c];
                let x1 = acc[(q + 1u) * {ap}u + c];
                let x2 = acc[(q + 2u) * {ap}u + c];
                let x3 = acc[(q + 3u) * {ap}u + c];
                let x4 = acc[(q + 4u) * {ap}u + c];
                let x5 = acc[(q + 5u) * {ap}u + c];
                let x6 = acc[(q + 6u) * {ap}u + c];
                let x7 = acc[(q + 7u) * {ap}u + c];
                s0 = fma(l0, x0, s0);
                s1 = fma(l1, x1, s1);
                s0 = fma(l2, x2, s0);
                s1 = fma(l3, x3, s1);
                s0 = fma(l4, x4, s0);
                s1 = fma(l5, x5, s1);
                s0 = fma(l6, x6, s0);
                s1 = fma(l7, x7, s1);
            }}
            for (; q <= rr; q = q + 1u) {{ s0 = fma(iv[rr * {LP}u + q], acc[q * {ap}u + c], s0); }}
            xs[rr * {ap}u + c] = s0 + s1;
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
        for (var q = sg * {SEGW}u; q < sg * {SEGW}u + {SEGW}u; q = q + 4u) {{
            let lq0 = lt[rr * {LP}u + q];
            let lq1 = lt[rr * {LP}u + q + 1u];
            let lq2 = lt[rr * {LP}u + q + 2u];
            let lq3 = lt[rr * {LP}u + q + 3u];
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

/// The columns of the right-hand sides a workgroup takes and whether the diagonal blocks are inverted, for `n` rows and `k` columns:
/// `inverse` asks for the inverted blocks, which the kernel takes from `LUMA_TRSV_INV_MIN` rows on (768; the inverse of the first block is
/// on the chain whatever the size, 10 us or so a launch: substitution is faster below a few blocks).
pub(crate) fn plan(n: usize, k: usize, inverse: bool) -> (usize, bool) {
    let inverse = inverse && n >= std::env::var("LUMA_TRSV_INV_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(768);
    (group_columns(k.next_power_of_two()), inverse)
}

/// `L X = B` for one `[n, n]` lower triangle and `[k, n]` right-hand sides, `k <= KMAX`; see the module doc.
pub(crate) fn lower_left(l: &Array<f32>, b: &Array<f32>, unit: bool, kc: usize, inverse: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let n = l.shape().dims()[1];
    let k = b.shape().dims()[0];
    let nblocks = n.div_ceil(NB);
    let ncg = k.div_ceil(kc);
    let setup = (Array::<f32>::alloc(ctx, b.shape()), Array::<u32>::zeros_on(ctx, 1 + ncg * n * kc));
    let (mut out, mut st) = match setup {
        (Ok(o), Ok(s)) => (o, s),
        (Err(e), _) | (_, Err(e)) => return fail(ctx, b.shape(), e, op, loc),
    };
    let params = Params { n: n as u32, k: k as u32, unit: u32::from(unit), nblocks: nblocks as u32, ncg: ncg as u32 };
    let args: [KernelArg<'_>; 5] = [view(l), view(b), (&mut out).into(), (&mut st).into(), uniform(&params)];
    match launch_unchecked(ctx, &format!("trsv/{kc}/{}", u8::from(inverse)), "trsv", || source(kc, inverse), [(nblocks * ncg) as u32, 1, 1], &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, b.shape(), e, op, loc),
    }
}
