//! Tensor-core matrix product on `f16` / `bf16` operands: `mma.sync.aligned.m16n8k16` kernels written as PTX (W3-R7).
//!
//! The operands are 16-bit elements (two per 32-bit word, as the arrays store them), multiplied exactly and accumulated in `f32`: no
//! rounding of the operands, so no TF32 multiply. One workgroup computes a `rows x cols` tile of `C` and walks K in stages of 32
//! elements through a multi-stage `cp.async` pipeline of 16-byte copies (eight elements); the fragments come from shared memory with
//! `ldmatrix` (`.trans` for the layouts whose `k` pairs are not contiguous), every row of a staged tile padded by 8 elements (the stride
//! in bytes is an odd multiple of 16, so the eight rows of an 8x8 matrix a thread group reads fall into eight different 16-byte bank
//! groups). The launch, the uniform and the slices of K are those of [`crate::tensor`]; the lanes are in units of 8 elements:
//! [`luma_core::GemmLanes::Vec4`] needs `m`, `n`, `k`, the leading dimensions and the offsets multiples of 8, `Vec4Tails` the leading
//! dimensions and offsets (the chunks across an edge are cut, the rest zero), `Scalar` is not generated.
//!
//! The output is `f32` (also the partial sums of the slices of K), or `f16` / `bf16` rounded in the store (`cvt.rn.f16x2.f32`, two
//! adjacent columns per 32-bit store where aligned): a half output is only used with one slice of K.

use luma_core::{GemmOut, GemmSpec, GemmUnit};

use crate::gemm_copy::{self, Mode};
use crate::ptx::Ptx;

/// Elements of K per stage.
const BK: u32 = 32;
/// Padding of a staged row, elements.
const PAD: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config16 {
    pub bm: u32,
    pub bn: u32,
    pub wm: u32,
    pub wn: u32,
    pub stages: u32,
    pub lhs_t: bool,
    pub rhs_t: bool,
    pub group: u32,
    pub mode: Mode,
    pub bf16: bool,
    pub out: GemmOut,
}

impl Config16 {
    /// The kernel of `spec` (a `F16` or `Bf16` unit), `None` for a spec this generator does not take (the scalar lanes, half-work
    /// products, a tile it has no layout for).
    pub(crate) fn of(spec: &GemmSpec) -> Option<Config16> {
        if !matches!(spec.unit, GemmUnit::F16 | GemmUnit::Bf16) || spec.half || spec.hook.is_some() || spec.lanes == luma_core::GemmLanes::Scalar {
            return None;
        }
        let (wm, wn) = match (spec.rows, spec.cols) {
            (128, 128) => (2, 4),
            (128, 64) | (64, 128) | (64, 64) => (2, 2),
            _ => return None,
        };
        let stages = if spec.rows * spec.cols >= 128 * 128 { 3 } else { 4 };
        Some(Config16 { bm: spec.rows, bn: spec.cols, wm, wn, stages, lhs_t: spec.lhs_transposed, rhs_t: spec.rhs_transposed, group: 8, mode: Mode::of(spec.lanes), bf16: spec.unit == GemmUnit::Bf16, out: spec.out })
    }

    pub(crate) fn threads(&self) -> u32 {
        self.wm * self.wn * 32
    }

    /// Elements per staged row of `A`: `K` along rows (`A` transposed) or `M`.
    fn a_stride(&self) -> u32 {
        PAD + if self.lhs_t { self.bm } else { BK }
    }

    fn b_stride(&self) -> u32 {
        PAD + if self.rhs_t { BK } else { self.bn }
    }

    fn a_bytes(&self) -> u32 {
        2 * self.a_stride() * if self.lhs_t { BK } else { self.bm }
    }

    fn b_bytes(&self) -> u32 {
        2 * self.b_stride() * if self.rhs_t { self.bn } else { BK }
    }

    fn stage_bytes(&self) -> u32 {
        self.a_bytes() + self.b_bytes()
    }

    pub(crate) fn shared_bytes(&self) -> u32 {
        self.stages * self.stage_bytes()
    }
}

/// The `cp.async` copy of the chunk `j` of operand `x` (eight elements) to the shared address in `dst`, then the advance of its global
/// pointer by `step`. `k_contig` says the chunk runs along `k`. Clobbers `%r32`, `%r36`, `%p12`.
fn copy_chunk(w: &mut Ptx, mode: Mode, k_contig: bool, x: &str, j: u32, dst: &str, kleft: &str, step: &str) {
    match mode {
        Mode::Aligned => {
            p!(w, "setp.lt.s32 %p12, %k{x}{j}, {kleft};");
            p!(w, "and.pred %p12, %p12, %v{x}{j};");
            p!(w, "selp.b32 %r32, 16, 0, %p12;");
        }
        _ => {
            if k_contig {
                p!(w, "sub.s32 %r36, {kleft}, %k{x}{j};");
                p!(w, "max.s32 %r36, %r36, 0;");
                p!(w, "min.s32 %r36, %r36, 8;");
                p!(w, "shl.b32 %r36, %r36, 1;");
                p!(w, "selp.b32 %r32, %r36, 0, %v{x}{j};");
            } else {
                p!(w, "setp.lt.s32 %p12, %k{x}{j}, {kleft};");
                p!(w, "max.s32 %r36, %z{x}{j}, 0;");
                p!(w, "min.s32 %r36, %r36, 8;");
                p!(w, "shl.b32 %r36, %r36, 1;");
                p!(w, "selp.b32 %r32, %r36, 0, %p12;");
            }
        }
    }
    p!(w, "cp.async.cg.shared.global [{dst}], [%g{x}{j}], 16, %r32;");
    p!(w, "add.u64 %g{x}{j}, %g{x}{j}, {step};");
}

/// The PTX of the kernel `main`.
pub(crate) fn ptx(config: &Config16) -> String {
    let Config16 { bm, bn, wm, wn, stages, lhs_t, rhs_t, group, mode, bf16, out } = *config;
    let threads = config.threads();
    let (warp_rows, warp_cols) = (bm / wm, bn / wn);
    let (mt, nt) = (warp_rows / 16, warp_cols / 8);
    assert!(nt.is_multiple_of(2));
    let (sa, sb) = (config.a_stride(), config.b_stride());
    let (stage_bytes, a_bytes) = (config.stage_bytes(), config.a_bytes());
    let a_chunks = bm * BK / 8;
    let b_chunks = bn * BK / 8;
    assert!(a_chunks.is_multiple_of(threads) && b_chunks.is_multiple_of(threads));
    let (a_per, b_per) = (a_chunks / threads, b_chunks / threads);
    let kind = if bf16 { "bf16" } else { "f16" };

    let mut w = Ptx { text: String::new() };
    w.text.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n");
    w.text.push_str(".extern .shared .align 16 .b8 smem[];\n\n");
    w.text.push_str(".visible .entry main(\n    .param .align 4 .b8 luma[16],\n    .param .u64 luma_indirect,\n    .param .u64 pc,\n    .param .u64 pa,\n    .param .u64 pb,\n");
    w.text.push_str(&format!("    .param .align 16 .b8 pu[{}]\n)\n", gemm_copy::UNIFORM_BYTES));
    w.text.push_str(&format!(".maxntid {threads}, 1, 1\n{{\n"));
    w.text.push_str("    .reg .pred %p<16>;\n    .reg .b32 %r<64>;\n    .reg .b64 %rd<64>;\n    .reg .b16 %h<2>;\n");
    w.text.push_str(&format!("    .reg .f32 %c<{}>;\n", mt * nt * 4));
    w.text.push_str(&format!("    .reg .b32 %fa<{}>;\n", mt * 4));
    w.text.push_str(&format!("    .reg .b32 %fb<{}>;\n", nt * 2));
    gemm_copy::declare_chunks(&mut w.text, "a", a_per);
    gemm_copy::declare_chunks(&mut w.text, "b", b_per);
    // r0 m, r1 n, r2 k, r3 kc, r4 tid, r5 lane, r6 warp, r7 gid, r8 tg, r9 wm_i, r10 wn_i
    // r11 bm0, r12 bn0, r13 kbeg, r14 klen, r15 ktiles, r16 kt, r17 rd_off, r18 wr_off, r19 kleft
    // r20 smem base, r21 / r22 ldmatrix addresses of A / B, r23 / r24 a / b thread offset
    // r50 lda, r51 ldb (elements), r52 a_off, r53 b_off, r54 tri
    w.text.push('\n');
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
    p!(w, "mul.wide.u32 %rd3, %r52, 2;");
    p!(w, "add.u64 %rd1, %rd1, %rd3;");
    p!(w, "mul.wide.u32 %rd3, %r53, 2;");
    p!(w, "add.u64 %rd2, %rd2, %rd3;");
    p!(w, "mov.u32 %r4, %tid.x;");
    p!(w, "and.b32 %r5, %r4, 31;");
    p!(w, "shr.u32 %r6, %r4, 5;");
    p!(w, "shr.u32 %r7, %r5, 2;");
    p!(w, "and.b32 %r8, %r5, 3;");
    p!(w, "div.u32 %r9, %r6, {wn};");
    p!(w, "rem.u32 %r10, %r6, {wn};");
    if group == 0 {
        p!(w, "mov.u32 %r25, %ctaid.y;");
        p!(w, "mov.u32 %r26, %ctaid.x;");
    } else {
        p!(w, "mov.u32 %r25, %ctaid.y;");
        p!(w, "mov.u32 %r26, %nctaid.x;");
        p!(w, "mov.u32 %r27, %ctaid.x;");
        p!(w, "mad.lo.u32 %r28, %r25, %r26, %r27;");
        p!(w, "mul.lo.u32 %r29, %r26, {group};");
        p!(w, "div.u32 %r30, %r28, %r29;");
        p!(w, "mul.lo.u32 %r31, %r30, {group};");
        p!(w, "mov.u32 %r32, %nctaid.y;");
        p!(w, "sub.u32 %r32, %r32, %r31;");
        p!(w, "min.u32 %r32, %r32, {group};");
        p!(w, "rem.u32 %r33, %r28, %r29;");
        p!(w, "rem.u32 %r34, %r33, %r32;");
        p!(w, "add.u32 %r25, %r31, %r34;");
        p!(w, "div.u32 %r26, %r33, %r32;");
    }
    p!(w, "mul.lo.u32 %r11, %r25, {bm};");
    p!(w, "mul.lo.u32 %r12, %r26, {bn};");
    p!(w, "mov.u32 %r25, %ctaid.z;");
    p!(w, "mul.lo.u32 %r13, %r25, %r3;");
    p!(w, "add.u32 %r26, %r13, %r3;");
    p!(w, "min.u32 %r26, %r26, %r2;");
    gemm_copy::k_range(&mut w, bm, bn);
    p!(w, "add.u32 %r15, %r14, {};", BK - 1);
    p!(w, "shr.u32 %r15, %r15, {};", BK.trailing_zeros());
    p!(w, "mov.u32 %r20, smem;");
    for i in 0..mt * nt * 4 {
        p!(w, "mov.f32 %c{i}, 0f00000000;");
    }

    // Per-thread copy state. Chunk `c = tid + j * threads` of an operand tile, eight elements.
    for j in 0..a_per {
        p!(w, "add.u32 %r27, %r4, {};", j * threads);
        if !lhs_t {
            let cpr = BK / 8;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (m)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 3;"); // k offset in tile
            p!(w, "mov.u32 %ka{j}, %r29;");
            p!(w, "add.u32 %r30, %r11, %r28;");
            p!(w, "setp.lt.u32 %va{j}, %r30, %r0;");
            p!(w, "mad.lo.u32 %r31, %r30, %r50, %r13;");
            p!(w, "add.u32 %r31, %r31, %r29;");
            p!(w, "mul.wide.u32 %rd3, %r31, 2;");
            p!(w, "add.u64 %ga{j}, %rd1, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sa}, %r29;");
            p!(w, "shl.b32 %da{j}, %r31, 1;");
        } else {
            let cpr = bm / 8;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (k)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 3;"); // m offset in tile
            p!(w, "mov.u32 %ka{j}, %r28;");
            p!(w, "add.u32 %r30, %r11, %r29;");
            p!(w, "setp.lt.u32 %va{j}, %r30, %r0;");
            if mode.guarded() {
                p!(w, "sub.s32 %za{j}, %r0, %r30;");
            }
            p!(w, "add.u32 %r31, %r13, %r28;");
            p!(w, "mad.lo.u32 %r31, %r31, %r50, %r30;");
            p!(w, "mul.wide.u32 %rd3, %r31, 2;");
            p!(w, "add.u64 %ga{j}, %rd1, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sa}, %r29;");
            p!(w, "shl.b32 %da{j}, %r31, 1;");
        }
    }
    for j in 0..b_per {
        p!(w, "add.u32 %r27, %r4, {};", j * threads);
        if !rhs_t {
            let cpr = bn / 8;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (k)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 3;"); // n offset
            p!(w, "mov.u32 %kb{j}, %r28;");
            p!(w, "add.u32 %r30, %r12, %r29;");
            p!(w, "setp.lt.u32 %vb{j}, %r30, %r1;");
            if mode.guarded() {
                p!(w, "sub.s32 %zb{j}, %r1, %r30;");
            }
            p!(w, "add.u32 %r31, %r13, %r28;");
            p!(w, "mad.lo.u32 %r31, %r31, %r51, %r30;");
            p!(w, "mul.wide.u32 %rd3, %r31, 2;");
            p!(w, "add.u64 %gb{j}, %rd2, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sb}, %r29;");
            p!(w, "shl.b32 %db{j}, %r31, 1;");
        } else {
            let cpr = BK / 8;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (n)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 3;"); // k offset
            p!(w, "mov.u32 %kb{j}, %r29;");
            p!(w, "add.u32 %r30, %r12, %r28;");
            p!(w, "setp.lt.u32 %vb{j}, %r30, %r1;");
            p!(w, "mad.lo.u32 %r31, %r30, %r51, %r13;");
            p!(w, "add.u32 %r31, %r31, %r29;");
            p!(w, "mul.wide.u32 %rd3, %r31, 2;");
            p!(w, "add.u64 %gb{j}, %rd2, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sb}, %r29;");
            p!(w, "shl.b32 %db{j}, %r31, 1;");
        }
    }
    // Bytes the global pointers advance per stage.
    if !lhs_t {
        p!(w, "mov.u64 %rd10, {};", BK * 2);
    } else {
        p!(w, "mul.wide.u32 %rd10, %r50, {};", BK * 2);
    }
    if !rhs_t {
        p!(w, "mul.wide.u32 %rd11, %r51, {};", BK * 2);
    } else {
        p!(w, "mov.u64 %rd11, {};", BK * 2);
    }

    // ldmatrix thread offsets (bytes within a stage). Lane `l`: matrix `q = l / 8`, row `r = l % 8` of it.
    p!(w, "shr.u32 %r35, %r5, 3;"); // q
    p!(w, "and.b32 %r37, %r5, 7;"); // r
    p!(w, "and.b32 %r38, %r35, 1;"); // q & 1
    p!(w, "shr.u32 %r39, %r35, 1;"); // q >> 1
    p!(w, "mul.lo.u32 %r40, %r9, {warp_rows};"); // warp row base
    p!(w, "mul.lo.u32 %r41, %r10, {warp_cols};"); // warp column base
    if !lhs_t {
        // row = warp_row0 + (q & 1) * 8 + r, k = (q >> 1) * 8
        p!(w, "mad.lo.u32 %r23, %r38, 8, %r40;");
        p!(w, "add.u32 %r23, %r23, %r37;");
        p!(w, "mul.lo.u32 %r23, %r23, {sa};");
        p!(w, "mad.lo.u32 %r23, %r39, 8, %r23;");
    } else {
        // `.trans`: stored [k][m]: k = (q >> 1) * 8 + r, m = warp_row0 + (q & 1) * 8
        p!(w, "mad.lo.u32 %r23, %r39, 8, %r37;");
        p!(w, "mul.lo.u32 %r23, %r23, {sa};");
        p!(w, "add.u32 %r23, %r23, %r40;");
        p!(w, "mad.lo.u32 %r23, %r38, 8, %r23;");
    }
    p!(w, "shl.b32 %r23, %r23, 1;");
    if rhs_t {
        // stored [n][k]: n = warp_col0 + (q >> 1) * 8 + r, k = (q & 1) * 8
        p!(w, "mad.lo.u32 %r24, %r39, 8, %r41;");
        p!(w, "add.u32 %r24, %r24, %r37;");
        p!(w, "mul.lo.u32 %r24, %r24, {sb};");
        p!(w, "mad.lo.u32 %r24, %r38, 8, %r24;");
    } else {
        // `.trans`, stored [k][n]: k = (q & 1) * 8 + r, n = warp_col0 + (q >> 1) * 8
        p!(w, "mad.lo.u32 %r24, %r38, 8, %r37;");
        p!(w, "mul.lo.u32 %r24, %r24, {sb};");
        p!(w, "add.u32 %r24, %r24, %r41;");
        p!(w, "mad.lo.u32 %r24, %r39, 8, %r24;");
    }
    p!(w, "shl.b32 %r24, %r24, 1;");
    p!(w, "add.u32 %r24, %r24, {a_bytes};");

    let issue = |w: &mut Ptx| {
        for j in 0..a_per {
            p!(w, "add.u32 %r33, %r20, %r18;");
            p!(w, "add.u32 %r33, %r33, %da{j};");
            copy_chunk(w, mode, !lhs_t, "a", j, "%r33", "%r19", "%rd10");
        }
        for j in 0..b_per {
            p!(w, "add.u32 %r33, %r20, %r18;");
            p!(w, "add.u32 %r33, %r33, %db{j};");
            p!(w, "add.u32 %r33, %r33, {a_bytes};");
            copy_chunk(w, mode, rhs_t, "b", j, "%r33", "%r19", "%rd11");
        }
        p!(w, "sub.s32 %r19, %r19, {BK};");
        p!(w, "add.u32 %r18, %r18, {stage_bytes};");
        p!(w, "setp.ge.u32 %p2, %r18, {};", stages * stage_bytes);
        p!(w, "@%p2 mov.u32 %r18, 0;");
    };
    p!(w, "mov.u32 %r18, 0;");
    p!(w, "mov.s32 %r19, %r14;");
    for _ in 0..stages - 1 {
        issue(&mut w);
        p!(w, "cp.async.commit_group;");
    }
    p!(w, "mov.u32 %r17, 0;");
    p!(w, "mov.u32 %r16, 0;");
    p!(w, "setp.ge.u32 %p3, %r16, %r15;");
    p!(w, "@%p3 bra DONE;");
    w.text.push_str("LOOP:\n");
    p!(w, "cp.async.wait_group {};", stages - 2);
    p!(w, "bar.sync 0;");
    issue(&mut w);
    p!(w, "cp.async.commit_group;");
    p!(w, "add.u32 %r21, %r20, %r17;");
    p!(w, "add.u32 %r21, %r21, %r23;");
    p!(w, "add.u32 %r22, %r20, %r17;");
    p!(w, "add.u32 %r22, %r22, %r24;");
    for kk in 0..BK / 16 {
        for i in 0..mt {
            let off = if !lhs_t { (i * 16 * sa + kk * 16) * 2 } else { (kk * 16 * sa + i * 16) * 2 };
            let t = if lhs_t { ".trans" } else { "" };
            p!(w, "ldmatrix.sync.aligned.m8n8.x4{t}.shared.b16 {{%fa{}, %fa{}, %fa{}, %fa{}}}, [%r21+{off}];", i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3);
        }
        for jp in 0..nt / 2 {
            let off = if rhs_t { (jp * 16 * sb + kk * 16) * 2 } else { (kk * 16 * sb + jp * 16) * 2 };
            let t = if rhs_t { "" } else { ".trans" };
            p!(w, "ldmatrix.sync.aligned.m8n8.x4{t}.shared.b16 {{%fb{}, %fb{}, %fb{}, %fb{}}}, [%r22+{off}];", jp * 4, jp * 4 + 1, jp * 4 + 2, jp * 4 + 3);
        }
        for i in 0..mt {
            for j in 0..nt {
                let c = (i * nt + j) * 4;
                p!(
                    w,
                    "mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 {{%c{}, %c{}, %c{}, %c{}}}, {{%fa{}, %fa{}, %fa{}, %fa{}}}, {{%fb{}, %fb{}}}, {{%c{}, %c{}, %c{}, %c{}}};",
                    c, c + 1, c + 2, c + 3, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3, j * 2, j * 2 + 1, c, c + 1, c + 2, c + 3
                );
            }
        }
    }
    p!(w, "add.u32 %r17, %r17, {stage_bytes};");
    p!(w, "setp.ge.u32 %p2, %r17, {};", stages * stage_bytes);
    p!(w, "@%p2 mov.u32 %r17, 0;");
    p!(w, "add.u32 %r16, %r16, 1;");
    p!(w, "setp.lt.u32 %p3, %r16, %r15;");
    p!(w, "@%p3 bra LOOP;");
    w.text.push_str("DONE:\n");
    p!(w, "cp.async.wait_group 0;");

    // Epilogue: C[z * m * n + row * n + col].
    p!(w, "mov.u32 %r25, %ctaid.z;");
    p!(w, "mul.lo.u32 %r26, %r0, %r1;");
    p!(w, "mul.wide.u32 %rd4, %r26, %r25;");
    p!(w, "mul.lo.u32 %r27, %r9, {warp_rows};");
    p!(w, "add.u32 %r27, %r27, %r11;");
    p!(w, "add.u32 %r27, %r27, %r7;"); // row of fragment (0,0)
    p!(w, "mul.lo.u32 %r28, %r10, {warp_cols};");
    p!(w, "add.u32 %r28, %r28, %r12;");
    p!(w, "shl.b32 %r29, %r8, 1;");
    p!(w, "add.u32 %r28, %r28, %r29;"); // col of fragment (0,0)
    let shift = if out == GemmOut::F32 { 2 } else { 1 };
    if mode.guarded() {
        p!(w, "and.b32 %r35, %r1, 1;");
        p!(w, "setp.eq.u32 %p6, %r35, 0;"); // n even: a pair is 4-byte (half) / 8-byte (f32) aligned
    }
    for i in 0..mt {
        for h in 0..2u32 {
            p!(w, "add.u32 %r30, %r27, {};", i * 16 + h * 8);
            p!(w, "setp.lt.u32 %p4, %r30, %r0;");
            p!(w, "mad.lo.u32 %r31, %r30, %r1, %r28;");
            p!(w, "cvt.u64.u32 %rd5, %r31;");
            p!(w, "add.u64 %rd5, %rd5, %rd4;");
            p!(w, "shl.b64 %rd5, %rd5, {shift};");
            p!(w, "add.u64 %rd5, %rd5, %rd0;");
            for j in 0..nt {
                p!(w, "add.u32 %r32, %r28, {};", j * 8);
                let c = (i * nt + j) * 4 + h * 2;
                let at = j * 8 * (1 << shift);
                if out == GemmOut::F32 {
                    if mode.guarded() {
                        gemm_copy::guarded_pair(&mut w, at, c);
                    } else {
                        p!(w, "setp.lt.u32 %p5, %r32, %r1;");
                        p!(w, "and.pred %p5, %p5, %p4;");
                        p!(w, "@%p5 st.global.v2.f32 [%rd5+{at}], {{%c{c}, %c{}}};", c + 1);
                    }
                    continue;
                }
                let cvt = if out == GemmOut::Bf16 { "bf16" } else { "f16" };
                if mode.guarded() {
                    p!(w, "setp.lt.u32 %p5, %r32, %r1;");
                    p!(w, "and.pred %p5, %p5, %p4;");
                    p!(w, "add.u32 %r36, %r32, 1;");
                    p!(w, "setp.lt.u32 %p7, %r36, %r1;");
                    p!(w, "and.pred %p7, %p7, %p4;");
                    p!(w, "and.pred %p8, %p7, %p6;");
                    p!(w, "cvt.rn.{cvt}x2.f32 %r34, %c{}, %c{c};", c + 1);
                    p!(w, "@%p8 st.global.b32 [%rd5+{at}], %r34;");
                    p!(w, "not.pred %p9, %p8;");
                    p!(w, "and.pred %p10, %p5, %p9;");
                    p!(w, "cvt.rn.{cvt}.f32 %h0, %c{c};");
                    p!(w, "@%p10 st.global.b16 [%rd5+{at}], %h0;");
                    p!(w, "and.pred %p11, %p7, %p9;");
                    p!(w, "cvt.rn.{cvt}.f32 %h1, %c{};", c + 1);
                    p!(w, "@%p11 st.global.b16 [%rd5+{}], %h1;", at + 2);
                } else {
                    p!(w, "setp.lt.u32 %p5, %r32, %r1;");
                    p!(w, "and.pred %p5, %p5, %p4;");
                    p!(w, "cvt.rn.{cvt}x2.f32 %r34, %c{}, %c{c};", c + 1);
                    p!(w, "@%p5 st.global.b32 [%rd5+{at}], %r34;");
                }
            }
        }
    }
    p!(w, "ret;");
    w.text.push_str("}\n");
    w.text
}

#[cfg(test)]
mod tests {
    use super::*;
    use luma_core::GemmLanes;

    #[test]
    fn every_tile_layout_and_output_generates_a_kernel() {
        for unit in [GemmUnit::F16, GemmUnit::Bf16] {
            for (rows, cols) in [(128, 128), (128, 64), (64, 128), (64, 64)] {
                for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                    for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails] {
                        for out in [GemmOut::F32, GemmOut::F16, GemmOut::Bf16] {
                            let spec = GemmSpec::new(rows, cols, lhs_t, rhs_t, unit).with_lanes(lanes).with_out(out);
                            let config = Config16::of(&spec).expect("a tile of the table");
                            let text = ptx(&config);
                            assert!(text.contains(".entry main(") && text.trim_end().ends_with('}'));
                            let (warp_rows, warp_cols) = (rows / config.wm, cols / config.wn);
                            // One mma per fragment pair of each of the two k steps of a stage.
                            assert_eq!(text.matches("mma.sync").count() as u32, (warp_rows / 16) * (warp_cols / 8) * 2, "{rows}x{cols}");
                            assert!(text.contains(if lhs_t { "x4.trans.shared" } else { "x4.shared" }));
                            assert!(config.shared_bytes() <= 99 * 1024 && config.threads() <= 256, "{rows}x{cols}");
                        }
                    }
                }
            }
        }
        assert!(Config16::of(&GemmSpec::new(64, 64, false, false, GemmUnit::F16).with_lanes(GemmLanes::Scalar)).is_none());
        assert!(Config16::of(&GemmSpec::new(64, 64, false, false, GemmUnit::Tf32)).is_none());
    }

    #[test]
    fn staged_rows_fall_into_distinct_bank_groups() {
        // The eight rows an `ldmatrix` matrix reads (16 bytes each) must start in eight different 16-byte groups of the 128-byte
        // bank window, and every row must be 16-byte aligned.
        for (rows, cols) in [(128, 128), (128, 64), (64, 128), (64, 64)] {
            for (lhs_t, rhs_t) in [(false, false), (true, true), (true, false), (false, true)] {
                let config = Config16::of(&GemmSpec::new(rows, cols, lhs_t, rhs_t, GemmUnit::F16)).unwrap();
                for stride in [config.a_stride(), config.b_stride()] {
                    let bytes = stride * 2;
                    assert_eq!(bytes % 16, 0);
                    let groups: std::collections::HashSet<u32> = (0..8).map(|r| (r * bytes % 128) / 16).collect();
                    assert_eq!(groups.len(), 8, "stride {stride} halves");
                }
            }
        }
    }
}
