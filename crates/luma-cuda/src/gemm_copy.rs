//! What the PTX matrix products ([`crate::tensor`], [`crate::sgemm`]) share: the copy of a chunk of four floats from global to shared
//! memory in each of the [`GemmLanes`], and the words of the eight-word uniform.
//!
//! A chunk is four consecutive elements of one row of an operand's tile. Its registers: `%k{x}{j}` the `k` of the chunk (its first one
//! if the chunk runs along `k`, its row's otherwise), `%v{x}{j}` whether its row (along `k`) or its first column is inside the matrix,
//! `%z{x}{j}` the elements of its row left from its first column (a chunk that runs along `m` or `n`), `%g{x}{j}` its global address.

use luma_core::GemmLanes;

use crate::ptx::Ptx;

/// The words of the uniform: `{m, n, k, kc, lda, ldb, a_off, b_off, tri, 0, 0, 0}` (`tri` is `TriangularK`'s word, see [`k_range`]; the last three
/// are reserved). A launch that passes fewer words has the rest zero.
pub(crate) const UNIFORM_BYTES: usize = 48;

/// How the copies of a kernel guard the edges of the matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Mode {
    /// 16-byte copies of whole chunks: the extents are multiples of 4.
    Aligned,
    /// 16-byte copies, the chunks across an edge cut.
    Tails,
    /// 4-byte copies.
    Scalar,
}

impl Mode {
    pub(crate) fn of(lanes: GemmLanes) -> Mode {
        match lanes {
            GemmLanes::Vec4 => Mode::Aligned,
            GemmLanes::Vec4Tails => Mode::Tails,
            // A lane the kernels do not know is the one that takes any shape.
            _ => Mode::Scalar,
        }
    }

    /// Whether stores and the edges of the copies are guarded element by element.
    pub(crate) fn guarded(self) -> bool {
        self != Mode::Aligned
    }
}

/// The `cp.async` copies of the chunk `j` of operand `x` (`a` or `b`) to the shared address in `dst`, then the advance of its global
/// pointer by `step`. `kleft` is the register with the elements of K left in the slice; `k_contig` says the chunk runs along `k`.
/// Clobbers `%r32`, `%r36`, `%p12`, `%p13`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_chunk(w: &mut Ptx, mode: Mode, k_contig: bool, x: &str, j: u32, dst: &str, kleft: &str, step: &str) {
    match mode {
        Mode::Aligned => {
            p!(w, "setp.lt.s32 %p12, %k{x}{j}, {kleft};");
            p!(w, "and.pred %p12, %p12, %v{x}{j};");
            p!(w, "selp.b32 %r32, 16, 0, %p12;");
            p!(w, "cp.async.cg.shared.global [{dst}], [%g{x}{j}], 16, %r32;");
        }
        Mode::Tails => {
            if k_contig {
                p!(w, "sub.s32 %r36, {kleft}, %k{x}{j};");
                p!(w, "max.s32 %r36, %r36, 0;");
                p!(w, "min.s32 %r36, %r36, 4;");
                p!(w, "shl.b32 %r36, %r36, 2;");
                p!(w, "selp.b32 %r32, %r36, 0, %v{x}{j};");
            } else {
                p!(w, "setp.lt.s32 %p12, %k{x}{j}, {kleft};");
                p!(w, "max.s32 %r36, %z{x}{j}, 0;");
                p!(w, "min.s32 %r36, %r36, 4;");
                p!(w, "shl.b32 %r36, %r36, 2;");
                p!(w, "selp.b32 %r32, %r36, 0, %p12;");
            }
            p!(w, "cp.async.cg.shared.global [{dst}], [%g{x}{j}], 16, %r32;");
        }
        Mode::Scalar => {
            if k_contig {
                p!(w, "sub.s32 %r36, {kleft}, %k{x}{j};");
            } else {
                p!(w, "setp.lt.s32 %p13, %k{x}{j}, {kleft};");
            }
            for e in 0..4u32 {
                if k_contig {
                    p!(w, "setp.gt.s32 %p12, %r36, {e};");
                    p!(w, "and.pred %p12, %p12, %v{x}{j};");
                } else {
                    p!(w, "setp.gt.s32 %p12, %z{x}{j}, {e};");
                    p!(w, "and.pred %p12, %p12, %p13;");
                }
                p!(w, "selp.b32 %r32, 4, 0, %p12;");
                p!(w, "cp.async.ca.shared.global [{dst}+{}], [%g{x}{j}+{}], 4, %r32;", 4 * e, 4 * e);
            }
        }
    }
    p!(w, "add.u64 %g{x}{j}, %g{x}{j}, {step};");
}

/// Declares the registers of the chunks of an operand: `n` chunks of operand `x`.
pub(crate) fn declare_chunks(text: &mut String, x: &str, n: u32) {
    text.push_str(&format!("    .reg .b64 %g{x}<{n}>;\n    .reg .b32 %d{x}<{n}>;\n    .reg .b32 %k{x}<{n}>;\n    .reg .b32 %z{x}<{n}>;\n    .reg .pred %v{x}<{n}>;\n"));
}

/// Loads the uniform: `%r0..%r3` `m, n, k, kc`, `%r50..%r53` `lda, ldb, a_off, b_off`, `%r54` `tri`; the offsets are added to the pointers `%rd1` and
/// `%rd2` (clobbers `%rd3`).
pub(crate) fn load_uniform(w: &mut Ptx) {
    p!(w, "ld.param.u32 %r0, [pu];");
    p!(w, "ld.param.u32 %r1, [pu+4];");
    p!(w, "ld.param.u32 %r2, [pu+8];");
    p!(w, "ld.param.u32 %r3, [pu+12];");
    p!(w, "ld.param.u32 %r50, [pu+16];");
    p!(w, "ld.param.u32 %r51, [pu+20];");
    p!(w, "ld.param.u32 %r52, [pu+24];");
    p!(w, "ld.param.u32 %r53, [pu+28];");
    p!(w, "ld.param.u32 %r54, [pu+32];");
    p!(w, "ld.param.u64 %rd0, [pc];");
    p!(w, "ld.param.u64 %rd1, [pa];");
    p!(w, "ld.param.u64 %rd2, [pb];");
    p!(w, "mul.wide.u32 %rd3, %r52, 4;");
    p!(w, "add.u64 %rd1, %rd1, %rd3;");
    p!(w, "mul.wide.u32 %rd3, %r53, 4;");
    p!(w, "add.u64 %rd2, %rd2, %rd3;");
}

/// The range of K a workgroup covers: with `%r13` the first `k` of its slice and `%r26` one past its last on entry, `%r13` and
/// `%r14` (the length) on exit, cut to what the operand that is a triangle leaves non-zero for the tile `(%r11, %r12)` of edges `bm`
/// x `bn`: `tri` 1 (left operand lower: `k < bm0 + bm`), 2 (left upper: `k >= bm0`), 3 (right lower: `k >= bn0`), 4 (right
/// upper: `k < bn0 + bn`), 0 the whole range. The terms left out are products with zeros. Clobbers `%r27`, `%r28`, `%p15`.
pub(crate) fn k_range(w: &mut Ptx, bm: u32, bn: u32) {
    p!(w, "mov.u32 %r27, 0;");
    p!(w, "mov.u32 %r28, 4294967295;");
    p!(w, "setp.eq.u32 %p15, %r54, 1;");
    p!(w, "@%p15 add.u32 %r28, %r11, {bm};");
    p!(w, "setp.eq.u32 %p15, %r54, 2;");
    p!(w, "@%p15 mov.u32 %r27, %r11;");
    p!(w, "setp.eq.u32 %p15, %r54, 3;");
    p!(w, "@%p15 mov.u32 %r27, %r12;");
    p!(w, "setp.eq.u32 %p15, %r54, 4;");
    p!(w, "@%p15 add.u32 %r28, %r12, {bn};");
    p!(w, "max.u32 %r13, %r13, %r27;");
    p!(w, "min.u32 %r26, %r26, %r28;");
    p!(w, "max.u32 %r26, %r26, %r13;");
    p!(w, "sub.u32 %r14, %r26, %r13;");
}

/// The stores of the pair of accumulators `c0, c1` of one row (`%rd5` its address of column `col`, `%r32` the column, `%p4` whether the row
/// is inside) at byte `at`: `v2` where the pair is inside and the row's alignment allows it (`%p6` the output's `n` is even), else one
/// store each. Clobbers `%r36`, `%p5`, `%p7..%p11`.
pub(crate) fn guarded_pair(w: &mut Ptx, at: u32, c0: u32) {
    p!(w, "setp.lt.u32 %p5, %r32, %r1;");
    p!(w, "and.pred %p5, %p5, %p4;");
    p!(w, "add.u32 %r36, %r32, 1;");
    p!(w, "setp.lt.u32 %p7, %r36, %r1;");
    p!(w, "and.pred %p7, %p7, %p4;");
    p!(w, "and.pred %p8, %p7, %p6;");
    p!(w, "@%p8 st.global.v2.f32 [%rd5+{at}], {{%c{c0}, %c{}}};", c0 + 1);
    p!(w, "not.pred %p9, %p8;");
    p!(w, "and.pred %p10, %p5, %p9;");
    p!(w, "@%p10 st.global.f32 [%rd5+{at}], %c{c0};");
    p!(w, "and.pred %p11, %p7, %p9;");
    p!(w, "@%p11 st.global.f32 [%rd5+{}], %c{};", at + 4, c0 + 1);
}
