//! The cooperative-matrix `f16` product (W3-R7, `GemmUnit::CoopF16`): `Backend::compile_gemm` on wgpu for the stored `f16` operands of
//! `Array<f16>::matmul` on Vulkan devices whose subgroups are 32 wide (the `coop_mat16x16<f16>` fragments of NVIDIA's tensor cores).
//!
//! One workgroup of 4 subgroups computes a `bm x bn` tile (`64 x 64` or `128 x 128`), each subgroup a quarter of it as `(bm/32) x
//! (bn/32)` fragments of `16 x 16`. Slices of 32 of K are staged through double-buffered workgroup memory (padded rows against bank
//! conflicts) while the previous slice is consumed, the next slice sitting in registers meanwhile: the operands are `f16` elements,
//! two per `u32` word as the arrays store them, read as 16-byte chunks of eight (`array<vec4<u32>>`) and cut at the edges of the
//! matrix (the chunks past an extent are zeros, both operands, so a zero product is exact). Fragments leave through workgroup memory
//! because naga's SPIR-V output cannot store a fragment with a runtime stride (as in `benches/common/sgemm.rs`, the kernel this
//! productionises). A half output is rounded to nearest even in the store, by integer arithmetic: `pack2x16float` leaves the rounding
//! of the conversion (and what an overflow gives) to the driver.
//!
//! The words of the uniform are those of every [`GemmSpec`] kernel: `{m, n, k, kc, lda, ldb, a_off, b_off}`; the bindings `c`, `a`, `b`,
//! `p`. Leading dimensions and offsets are multiples of 8 (`Vec4` and `Vec4Tails` lanes); the grid is `[column tiles, row tiles,
//! slices of K]`, slice `z` of an `f32` output written at `z * m * n`. A half output needs an even `n` (a word holds two columns of a
//! row): `Vec4` lanes only.

use luma_core::{GemmLanes, GemmOut, GemmSpec, GemmUnit};

/// Threads of a workgroup: 4 subgroups of 32.
pub(crate) const THREADS: u32 = 128;

/// The tiles, largest first.
pub(crate) const TILES: [(u32, u32); 2] = [(128, 128), (64, 64)];

/// Bytes of workgroup memory of a kernel of the tile `bm x bn`: the two double-buffered operand tiles and the fragment staging.
pub(crate) fn workgroup_bytes(spec: &GemmSpec) -> u64 {
    let (a, b) = (operand_halves(spec.rows, spec.lhs_transposed), operand_halves(spec.cols, !spec.rhs_transposed));
    2 * (2 * u64::from(a) + 2 * u64::from(b)) + 4 * 1024
}

/// Elements of one buffer of an operand tile of `edge` rows (of M or N) by 32 of K: `[row][k]` at a stride of 40 when the buffer holds
/// the rows contiguously in K, `[k][row]` at a stride of `edge + 8` when it holds them along the rows.
fn operand_halves(edge: u32, along_rows: bool) -> u32 {
    if along_rows { 32 * (edge + 8) } else { edge * 40 }
}

/// The source of the kernel of `spec`, `None` for a spec this generator does not take.
pub(crate) fn source(spec: &GemmSpec) -> Option<String> {
    let (bm, bn) = (spec.rows, spec.cols);
    let tile_ok = TILES.contains(&(bm, bn));
    let lanes_ok = matches!(spec.lanes, GemmLanes::Vec4 | GemmLanes::Vec4Tails);
    // A half store packs two columns of a row into a word: only where `n` is even, which `Vec4` guarantees.
    let out_ok = match spec.out {
        GemmOut::F32 => true,
        GemmOut::F16 => spec.lanes == GemmLanes::Vec4,
        _ => false,
    };
    if spec.unit != GemmUnit::CoopF16 || !tile_ok || !lanes_ok || !out_ok || spec.half || spec.hook.is_some() {
        return None;
    }
    let (fi, fj) = (bm / 32, bn / 32);
    let (sa_half, sb_half) = (operand_halves(bm, spec.lhs_transposed), operand_halves(bn, !spec.rhs_transposed));
    let (a_stride, b_stride) = (bm + 8, bn + 8);
    let half_out = spec.out == GemmOut::F16;

    // The chunks of eight a thread stages of each operand per slice: `rows x 32` elements over the threads.
    let (a_chunks, b_chunks) = (bm * 4 / THREADS, bn * 4 / THREADS);

    // `chunk_a` / `chunk_b`: where chunk `i` of a thread lies, as the code that fetches it into `ra{i}` / `rb{i}` and the code that
    // stores it into workgroup memory.
    let mut fetch_a = String::new();
    let mut commit_a = String::new();
    for i in 0..a_chunks {
        let q = format!("(t + {}u)", THREADS * i);
        if spec.lhs_transposed {
            // The buffer holds `[k][m]` (m fastest): a chunk is eight consecutive rows of M of one k.
            let per_row = bm / 8;
            fetch_a += &format!(
                "    {{ let kk = @K@ + {q} / {per_row}u; let mg = row0 + ({q} % {per_row}u) * 8u;
      let valid = select(0u, min(8u, p.m - min(mg, p.m)), kk < ke && mg < p.m);
      ra{i} = chunk_a((p.a_off + kk * p.lda + mg) / 8u, valid); }}\n"
            );
            commit_a += &format!("    store_a(@BUF@ * {sa_half}u + ({q} / {per_row}u) * {a_stride}u + ({q} % {per_row}u) * 8u, ra{i});\n");
        } else {
            // The buffer holds `[m][k]` (k fastest): a chunk is eight consecutive k of one row.
            fetch_a += &format!(
                "    {{ let r = row0 + {q} / 4u; let kg = @K@ + ({q} % 4u) * 8u;
      let valid = select(0u, min(8u, ke - min(kg, ke)), r < p.m && kg < ke);
      ra{i} = chunk_a((p.a_off + r * p.lda + kg) / 8u, valid); }}\n"
            );
            commit_a += &format!("    store_a(@BUF@ * {sa_half}u + ({q} / 4u) * 40u + ({q} % 4u) * 8u, ra{i});\n");
        }
    }
    let mut fetch_b = String::new();
    let mut commit_b = String::new();
    for i in 0..b_chunks {
        let q = format!("(t + {}u)", THREADS * i);
        if spec.rhs_transposed {
            // The buffer holds `[n][k]` (k fastest).
            fetch_b += &format!(
                "    {{ let r = col0 + {q} / 4u; let kg = @K@ + ({q} % 4u) * 8u;
      let valid = select(0u, min(8u, ke - min(kg, ke)), r < p.n && kg < ke);
      rb{i} = chunk_b((p.b_off + r * p.ldb + kg) / 8u, valid); }}\n"
            );
            commit_b += &format!("    store_b(@BUF@ * {sb_half}u + ({q} / 4u) * 40u + ({q} % 4u) * 8u, rb{i});\n");
        } else {
            // The buffer holds `[k][n]` (n fastest).
            let per_row = bn / 8;
            fetch_b += &format!(
                "    {{ let kk = @K@ + {q} / {per_row}u; let ng = col0 + ({q} % {per_row}u) * 8u;
      let valid = select(0u, min(8u, p.n - min(ng, p.n)), kk < ke && ng < p.n);
      rb{i} = chunk_b((p.b_off + kk * p.ldb + ng) / 8u, valid); }}\n"
            );
            commit_b += &format!("    store_b(@BUF@ * {sb_half}u + ({q} / {per_row}u) * {b_stride}u + ({q} % {per_row}u) * 8u, rb{i});\n");
        }
    }
    let regs: String = (0..a_chunks).map(|i| format!("    var ra{i} = vec4<u32>(0u);\n")).chain((0..b_chunks).map(|i| format!("    var rb{i} = vec4<u32>(0u);\n"))).collect();
    let fetch = |k: &str| format!("{}{}", fetch_a.replace("@K@", k), fetch_b.replace("@K@", k));
    let commit = |buf: &str| format!("{}{}", commit_a.replace("@BUF@", buf), commit_b.replace("@BUF@", buf));

    // The fragments of one slice: the B fragments of a step first, then each A fragment right before its products (fewer live registers).
    let mut decl = String::new();
    for i in 0..fi {
        for j in 0..fj {
            decl += &format!("    var c{i}_{j} = coopLoad<coop_mat16x16<f32, C>>(&stage[0], 16u);\n");
        }
    }
    let mut frags = String::new();
    for kk in [0u32, 16] {
        frags += "        {\n";
        for j in 0..fj {
            frags += &if spec.rhs_transposed {
                format!("            let b{j} = coopLoad<coop_mat16x16<f16, B>>(&sb[bbase + (wn + {}u) * 40u + {kk}u], 40u);\n", 16 * j)
            } else {
                format!("            let b{j} = coopLoadT<coop_mat16x16<f16, B>>(&sb[bbase + {}u + wn + {}u], {b_stride}u);\n", kk * b_stride, 16 * j)
            };
        }
        for i in 0..fi {
            frags += &if spec.lhs_transposed {
                format!("            let a{i} = coopLoad<coop_mat16x16<f16, A>>(&sa[abase + {}u + wm + {}u], {a_stride}u);\n", kk * a_stride, 16 * i)
            } else {
                format!("            let a{i} = coopLoadT<coop_mat16x16<f16, A>>(&sa[abase + (wm + {}u) * 40u + {kk}u], 40u);\n", 16 * i)
            };
            for j in 0..fj {
                frags += &format!("            c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});\n");
            }
        }
        frags += "        }\n";
    }

    // Each fragment leaves through its subgroup's 256 floats of `stage`.
    let mut store = String::new();
    for i in 0..fi {
        for j in 0..fj {
            store += &format!("    coopStoreT(c{i}_{j}, &stage[sg * 256u], 16u);\n    workgroupBarrier();\n");
            if half_out {
                store += &format!(
                    "    for (var e = lane; e < 128u; e = e + 32u) {{
        let er = e / 8u;
        let ec = (e % 8u) * 2u;
        let r = row0 + wm + {}u + er;
        let col = col0 + wn + {}u + ec;
        if (r < p.m && col < p.n) {{
            c[(r * p.n + col) / 2u] = f16_bits(stage[sg * 256u + er * 16u + ec]) | (f16_bits(stage[sg * 256u + er * 16u + ec + 1u]) << 16u);
        }}
    }}
    workgroupBarrier();
",
                    16 * i,
                    16 * j
                );
            } else {
                store += &format!(
                    "    for (var e = lane; e < 256u; e = e + 32u) {{
        let r = row0 + wm + {}u + e / 16u;
        let col = col0 + wn + {}u + e % 16u;
        if (r < p.m && col < p.n) {{ c[zo + r * p.n + col] = stage[sg * 256u + e]; }}
    }}
    workgroupBarrier();
",
                    16 * i,
                    16 * j
                );
            }
        }
    }

    let out_ty = if half_out { "u32" } else { "f32" };
    let (first_fetch, first_commit) = (fetch("kb"), commit("0u"));
    let (next_fetch, next_commit) = (fetch("(k0 + 32u)"), commit("(1u - cur)"));
    Some(format!(
        "enable f16;
enable wgpu_cooperative_matrix;
struct Params {{ m: u32, n: u32, k: u32, kc: u32, lda: u32, ldb: u32, a_off: u32, b_off: u32 }}
@group(1) @binding(0) var<storage, read_write> c: array<{out_ty}>;
@group(1) @binding(1) var<storage, read> a: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read> b: array<vec4<u32>>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> sa: array<f16, {sa_size}>;
var<workgroup> sb: array<f16, {sb_size}>;
var<workgroup> stage: array<f32, 1024>;

// The word `w` of two halves keeping `left` of them (the low one first).
fn keep(w: u32, left: u32) -> u32 {{
    if (left >= 2u) {{ return w; }}
    if (left == 1u) {{ return w & 0xFFFFu; }}
    return 0u;
}}
fn clip(v: vec4<u32>, valid: u32) -> vec4<u32> {{
    return vec4<u32>(keep(v.x, valid), keep(v.y, select(0u, valid - 2u, valid > 2u)), keep(v.z, select(0u, valid - 4u, valid > 4u)), keep(v.w, select(0u, valid - 6u, valid > 6u)));
}}
// Chunk `i` (eight halves) of an operand with its first `valid` halves kept, the others zero; nothing is read where none is kept.
fn chunk_a(i: u32, valid: u32) -> vec4<u32> {{
    if (valid == 0u) {{ return vec4<u32>(0u); }}
    return select(clip(a[i], valid), a[i], valid >= 8u);
}}
fn chunk_b(i: u32, valid: u32) -> vec4<u32> {{
    if (valid == 0u) {{ return vec4<u32>(0u); }}
    return select(clip(b[i], valid), b[i], valid >= 8u);
}}
fn store_a(at: u32, v: vec4<u32>) {{
    let h0 = unpack2x16float(v.x); let h1 = unpack2x16float(v.y); let h2 = unpack2x16float(v.z); let h3 = unpack2x16float(v.w);
    sa[at] = f16(h0.x); sa[at + 1u] = f16(h0.y); sa[at + 2u] = f16(h1.x); sa[at + 3u] = f16(h1.y); sa[at + 4u] = f16(h2.x); sa[at + 5u] = f16(h2.y); sa[at + 6u] = f16(h3.x); sa[at + 7u] = f16(h3.y);
}}
fn store_b(at: u32, v: vec4<u32>) {{
    let h0 = unpack2x16float(v.x); let h1 = unpack2x16float(v.y); let h2 = unpack2x16float(v.z); let h3 = unpack2x16float(v.w);
    sb[at] = f16(h0.x); sb[at + 1u] = f16(h0.y); sb[at + 2u] = f16(h1.x); sb[at + 3u] = f16(h1.y); sb[at + 4u] = f16(h2.x); sb[at + 5u] = f16(h2.y); sb[at + 6u] = f16(h3.x); sb[at + 7u] = f16(h3.y);
}}
// `x` as f16 bits, to nearest even (an overflow gives an infinity, a NaN the canonical quiet NaN).
fn f16_bits(x: f32) -> u32 {{
    let bits = bitcast<u32>(x);
    let sgn = (bits >> 16u) & 0x8000u;
    let ex = i32((bits >> 23u) & 0xFFu);
    let man = bits & 0x7FFFFFu;
    if (ex == 255) {{ return sgn | 0x7C00u | select(0u, 0x200u, man != 0u); }}
    let e = ex - 112;
    if (e >= 31) {{ return sgn | 0x7C00u; }}
    if (e <= 0) {{
        if (e < -10) {{ return sgn; }}
        let m = man | 0x800000u;
        let shift = u32(14 - e);
        var h = m >> shift;
        let rm = m & ((1u << shift) - 1u);
        let mid = 1u << (shift - 1u);
        if (rm > mid || (rm == mid && (h & 1u) == 1u)) {{ h = h + 1u; }}
        return sgn | h;
    }}
    var h = (u32(e) << 10u) | (man >> 13u);
    let rm = man & 0x1FFFu;
    if (rm > 0x1000u || (rm == 0x1000u && (h & 1u) == 1u)) {{ h = h + 1u; }}
    return sgn | h;
}}

@compute @workgroup_size({THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let row0 = wid.y * {bm}u;
    let col0 = wid.x * {bn}u;
    let sg = t / 32u;
    let lane = t % 32u;
    let wm = (sg / 2u) * {}u;
    let wn = (sg % 2u) * {}u;
    let kb = wid.z * p.kc;
    let ke = min(p.k, kb + p.kc);
    let zo = wid.z * p.m * p.n;
    for (var e = t; e < 1024u; e = e + {THREADS}u) {{ stage[e] = 0.0; }}
    workgroupBarrier();
{decl}{regs}{first_fetch}{first_commit}    workgroupBarrier();
    for (var k0 = kb; k0 < ke; k0 = k0 + 32u) {{
        let cur = ((k0 - kb) / 32u) % 2u;
        let more = k0 + 32u < ke;
        let abase = cur * {sa_half}u;
        let bbase = cur * {sb_half}u;
        if (more) {{
{next_fetch}        }}
{frags}        if (more) {{
{next_commit}        }}
        workgroupBarrier();
    }}
{store}}}
",
        bm / 2,
        bn / 2,
        sa_size = 2 * sa_half,
        sb_size = 2 * sb_half,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant the unit has generates a source that parses (the device validates it where the feature exists).
    #[test]
    fn every_variant_generates_a_source() {
        for &(rows, cols) in &TILES {
            for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails] {
                for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                    for out in [GemmOut::F32, GemmOut::F16] {
                        let spec = GemmSpec::new(rows, cols, lhs_t, rhs_t, GemmUnit::CoopF16).with_lanes(lanes).with_out(out);
                        let text = source(&spec);
                        assert_eq!(text.is_some(), !(out == GemmOut::F16 && lanes == GemmLanes::Vec4Tails), "{spec:?}");
                        assert!(workgroup_bytes(&spec) <= 48 * 1024, "{spec:?}: {}", workgroup_bytes(&spec));
                    }
                }
            }
        }
        assert!(source(&GemmSpec::new(128, 128, false, false, GemmUnit::F16)).is_none());
        assert!(source(&GemmSpec::new(64, 128, false, false, GemmUnit::CoopF16)).is_none());
        assert!(source(&GemmSpec::new(64, 64, false, false, GemmUnit::CoopF16).with_lanes(GemmLanes::Scalar)).is_none());
    }
}
