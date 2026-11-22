//! Tensor-core matrix product: TF32 `mma.sync` kernels written as PTX (no NVVM: the driver's JIT
//! compiles the text).
//!
//! The kernel computes `C = A B` for the row-major `M x K` and `K x N` matrices of
//! [`TensorSpec`] (or their transposes, stored as `K x M` / `N x K`), operands rounded to TF32
//! (10 bits of mantissa) and accumulated in `f32`. One workgroup computes a `rows x cols` tile of
//! `C` and slices K in steps of the stage depth `bk` through a multi-stage `cp.async` pipeline of shared
//! memory, each warp multiplying its sub-tile with `mma.sync.aligned.m16n8k8`. The launch is
//! that of `Builtin::Matmul`: grid `[column tiles, row tiles, slices of K]`, uniform
//! `{m, n, k, kc}`; the slice of grid layer `z` is written at `z * m * n` of `c`.
//! `m`, `n` and `k` must be multiples of 4 (16-byte `cp.async` rows).

use luma_core::TensorSpec;

/// The shape of a kernel: output tile, warps and pipeline depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    pub bm: u32,
    pub bn: u32,
    /// Warps along `M` and `N`.
    pub wm: u32,
    pub wn: u32,
    pub stages: u32,
    /// K elements staged per pipeline stage.
    pub bk: u32,
    pub lhs_t: bool,
    pub rhs_t: bool,
    /// Rows of tiles a group of consecutive workgroups covers (0: row-major order), so that the
    /// workgroups running together read the operand panels of a square of tiles.
    pub group: u32,
}

impl Config {
    pub(crate) fn of(spec: &TensorSpec) -> Config {
        let (wm, wn) = match (spec.rows, spec.cols) {
            (128, 128) => (2, 4),
            _ => (2, 2),
        };
        // Two stages of the big tile fit two workgroups in a multiprocessor (38 KB each), which hides
        // more latency than a third stage: 32.2 TFLOP/s against 29.1 at 4096^3 on an RTX 3090.
        let stages = if spec.rows * spec.cols >= 128 * 128 { 2 } else { 4 };
        Config { bm: spec.rows, bn: spec.cols, wm, wn, stages, bk: 16, lhs_t: spec.lhs_transposed, rhs_t: spec.rhs_transposed, group: 8 }
    }

    pub(crate) fn threads(&self) -> u32 {
        self.wm * self.wn * 32
    }

    /// Floats per row of the staged `A` tile (`K` along rows) / of the transposed one, padded so
    /// that the fragment loads of a warp touch 32 different banks.
    fn a_stride(&self) -> u32 {
        if self.lhs_t { self.bm + 8 } else { self.bk + 4 }
    }

    fn b_stride(&self) -> u32 {
        if self.rhs_t { self.bk + 4 } else { self.bn + 8 }
    }

    fn a_bytes(&self) -> u32 {
        4 * self.a_stride() * if self.lhs_t { self.bk } else { self.bm }
    }

    fn b_bytes(&self) -> u32 {
        4 * self.b_stride() * if self.rhs_t { self.bn } else { self.bk }
    }

    fn stage_bytes(&self) -> u32 {
        self.a_bytes() + self.b_bytes()
    }

    /// Dynamic shared memory of a workgroup.
    pub(crate) fn shared_bytes(&self) -> u32 {
        self.stages * self.stage_bytes()
    }
}

struct Ptx {
    text: String,
}

impl Ptx {
    fn line(&mut self, s: impl AsRef<str>) {
        self.text.push_str("    ");
        self.text.push_str(s.as_ref());
        self.text.push('\n');
    }
}

macro_rules! p {
    ($w:expr, $($arg:tt)*) => { $w.line(format!($($arg)*)) };
}

/// The PTX of the kernel `main` for `config`.
pub(crate) fn ptx(config: &Config) -> String {
    let Config { bm, bn, wm, wn, stages, bk, lhs_t, rhs_t, group } = *config;
    let threads = config.threads();
    let (warp_rows, warp_cols) = (bm / wm, bn / wn);
    let (mt, nt) = (warp_rows / 16, warp_cols / 8);
    let (sa, sb) = (config.a_stride(), config.b_stride());
    let stage_bytes = config.stage_bytes();
    let a_bytes = config.a_bytes();
    // Chunks (16 bytes) of each operand's stage, and what a thread copies.
    let a_chunks = bm * bk / 4;
    let b_chunks = bn * bk / 4;
    assert!(a_chunks.is_multiple_of(threads) && b_chunks.is_multiple_of(threads));
    let (a_per, b_per) = (a_chunks / threads, b_chunks / threads);

    let mut w = Ptx { text: String::new() };
    w.text.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n");
    w.text.push_str(".extern .shared .align 16 .b8 smem[];\n\n");
    w.text.push_str(".visible .entry main(\n    .param .align 4 .b8 luma[16],\n    .param .u64 luma_indirect,\n    .param .u64 pc,\n    .param .u64 pa,\n    .param .u64 pb,\n    .param .align 16 .b8 pu[16]\n)\n");
    w.text.push_str(&format!(".maxntid {threads}, 1, 1\n{{\n"));
    w.text.push_str(
        "    .reg .pred %p<16>;\n    .reg .b32 %r<64>;\n    .reg .b64 %rd<64>;\n    .reg .f32 %f<2>;\n",
    );
    // Accumulators, fragments, per-thread copy state.
    w.text.push_str(&format!("    .reg .f32 %c<{}>;\n", mt * nt * 4));
    w.text.push_str(&format!("    .reg .b32 %fa<{}>;\n", mt * 4));
    w.text.push_str(&format!("    .reg .b32 %fb<{}>;\n", nt * 2));
    w.text.push_str(&format!("    .reg .b64 %ga<{}>;\n    .reg .b64 %gb<{}>;\n", a_per, b_per));
    w.text.push_str(&format!("    .reg .b32 %da<{}>;\n    .reg .b32 %db<{}>;\n", a_per, b_per));
    w.text.push_str(&format!("    .reg .b32 %ka<{}>;\n    .reg .b32 %kb<{}>;\n", a_per, b_per));
    w.text.push_str(&format!("    .reg .pred %va<{}>;\n    .reg .pred %vb<{}>;\n", a_per, b_per));
    // r0 m, r1 n, r2 k, r3 kc, r4 tid, r5 lane, r6 warp, r7 gid, r8 tg, r9 wm_i, r10 wn_i
    // r11 bm0, r12 bn0, r13 kbeg, r14 klen, r15 ktiles, r16 kt, r17 rd_off, r18 wr_off, r19 kleft
    // r20 smem base, r21 pa (frag base), r22 pb, r23 a thread off, r24 b thread off
    w.text.push('\n');
    p!(w, "ld.param.u32 %r0, [pu];");
    p!(w, "ld.param.u32 %r1, [pu+4];");
    p!(w, "ld.param.u32 %r2, [pu+8];");
    p!(w, "ld.param.u32 %r3, [pu+12];");
    p!(w, "ld.param.u64 %rd0, [pc];");
    p!(w, "ld.param.u64 %rd1, [pa];");
    p!(w, "ld.param.u64 %rd2, [pb];");
    p!(w, "mov.u32 %r4, %tid.x;");
    p!(w, "and.b32 %r5, %r4, 31;");
    p!(w, "shr.u32 %r6, %r4, 5;");
    p!(w, "shr.u32 %r7, %r5, 2;");
    p!(w, "and.b32 %r8, %r5, 3;");
    // Warp position: warp = wm_i * wn + wn_i.
    p!(w, "div.u32 %r9, %r6, {wn};");
    p!(w, "rem.u32 %r10, %r6, {wn};");
    if group == 0 {
        p!(w, "mov.u32 %r25, %ctaid.y;");
        p!(w, "mov.u32 %r26, %ctaid.x;");
    } else {
        // Tiles in groups of `group` rows: the workgroups numbered `x + y * columns` walk down a
        // group's rows first, so those running together cover a square of tiles.
        p!(w, "mov.u32 %r25, %ctaid.y;");
        p!(w, "mov.u32 %r26, %nctaid.x;");
        p!(w, "mov.u32 %r27, %ctaid.x;");
        p!(w, "mad.lo.u32 %r28, %r25, %r26, %r27;"); // linear tile
        p!(w, "mul.lo.u32 %r29, %r26, {group};"); // tiles in a group
        p!(w, "div.u32 %r30, %r28, %r29;");
        p!(w, "mul.lo.u32 %r31, %r30, {group};"); // first row of the group
        p!(w, "mov.u32 %r32, %nctaid.y;");
        p!(w, "sub.u32 %r32, %r32, %r31;");
        p!(w, "min.u32 %r32, %r32, {group};"); // rows in the group
        p!(w, "rem.u32 %r33, %r28, %r29;");
        p!(w, "rem.u32 %r34, %r33, %r32;");
        p!(w, "add.u32 %r25, %r31, %r34;"); // row
        p!(w, "div.u32 %r26, %r33, %r32;"); // column
    }
    p!(w, "mul.lo.u32 %r11, %r25, {bm};");
    p!(w, "mul.lo.u32 %r12, %r26, {bn};");
    p!(w, "mov.u32 %r25, %ctaid.z;");
    p!(w, "mul.lo.u32 %r13, %r25, %r3;");
    // klen = min(k, kbeg + kc) - kbeg
    p!(w, "add.u32 %r26, %r13, %r3;");
    p!(w, "min.u32 %r26, %r26, %r2;");
    p!(w, "sub.u32 %r14, %r26, %r13;");
    p!(w, "add.u32 %r15, %r14, {};", bk - 1);
    p!(w, "shr.u32 %r15, %r15, {};", bk.trailing_zeros());
    p!(w, "mov.u32 %r20, smem;");

    // Zero the accumulators.
    for i in 0..mt * nt * 4 {
        p!(w, "mov.f32 %c{i}, 0f00000000;");
    }

    // Per-thread copy state. Chunk `c = tid + j * threads` of an operand tile.
    // Left operand: normal => tile rows are M (bm) with bk/4 chunks each; transposed => rows are K.
    for j in 0..a_per {
        p!(w, "add.u32 %r27, %r4, {};", j * threads);
        if !lhs_t {
            let cpr = bk / 4;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (m)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1); // chunk in row
            p!(w, "shl.b32 %r29, %r29, 2;"); // k offset in tile
            p!(w, "mov.u32 %ka{j}, %r29;");
            // global: A + ((bm0 + row) * k + kbeg + koff) * 4
            p!(w, "add.u32 %r30, %r11, %r28;");
            p!(w, "setp.lt.u32 %va{j}, %r30, %r0;");
            p!(w, "mad.lo.u32 %r31, %r30, %r2, %r13;");
            p!(w, "add.u32 %r31, %r31, %r29;");
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %ga{j}, %rd1, %rd3;");
            // smem: (row * sa + koff) * 4
            p!(w, "mad.lo.u32 %r31, %r28, {sa}, %r29;");
            p!(w, "shl.b32 %da{j}, %r31, 2;");
        } else {
            let cpr = bm / 4;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros()); // row (k)
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 2;"); // m offset in tile
            p!(w, "mov.u32 %ka{j}, %r28;");
            // global: A + ((kbeg + row) * m + bm0 + moff) * 4
            p!(w, "add.u32 %r30, %r11, %r29;");
            p!(w, "setp.lt.u32 %va{j}, %r30, %r0;");
            p!(w, "add.u32 %r31, %r13, %r28;");
            p!(w, "mad.lo.u32 %r31, %r31, %r0, %r30;");
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %ga{j}, %rd1, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sa}, %r29;");
            p!(w, "shl.b32 %da{j}, %r31, 2;");
        }
    }
    for j in 0..b_per {
        p!(w, "add.u32 %r27, %r4, {};", j * threads);
        if !rhs_t {
            // rows are K, bn/4 chunks each
            let cpr = bn / 4;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros());
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 2;"); // n offset in tile
            p!(w, "mov.u32 %kb{j}, %r28;");
            p!(w, "add.u32 %r30, %r12, %r29;");
            p!(w, "setp.lt.u32 %vb{j}, %r30, %r1;");
            p!(w, "add.u32 %r31, %r13, %r28;");
            p!(w, "mad.lo.u32 %r31, %r31, %r1, %r30;");
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %gb{j}, %rd2, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sb}, %r29;");
            p!(w, "shl.b32 %db{j}, %r31, 2;");
        } else {
            // rows are N, bk/4 chunks each
            let cpr = bk / 4;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros());
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 2;"); // k offset
            p!(w, "mov.u32 %kb{j}, %r29;");
            p!(w, "add.u32 %r30, %r12, %r28;");
            p!(w, "setp.lt.u32 %vb{j}, %r30, %r1;");
            p!(w, "mad.lo.u32 %r31, %r30, %r2, %r13;");
            p!(w, "add.u32 %r31, %r31, %r29;");
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %gb{j}, %rd2, %rd3;");
            p!(w, "mad.lo.u32 %r31, %r28, {sb}, %r29;");
            p!(w, "shl.b32 %db{j}, %r31, 2;");
        }
    }
    // Bytes the global pointers advance per stage.
    if !lhs_t {
        p!(w, "mov.u64 %rd10, {};", bk * 4);
    } else {
        p!(w, "mul.wide.u32 %rd10, %r0, {};", bk * 4);
    }
    if !rhs_t {
        p!(w, "mul.wide.u32 %rd11, %r1, {};", bk * 4);
    } else {
        p!(w, "mov.u64 %rd11, {};", bk * 4);
    }

    // Fragment base offsets (bytes, within a stage).
    // A: element (m, k) of the warp tile row group, see the module docs.
    p!(w, "mul.lo.u32 %r23, %r9, {warp_rows};"); // warp row base
    if !lhs_t {
        p!(w, "add.u32 %r23, %r23, %r7;");
        p!(w, "mul.lo.u32 %r23, %r23, {sa};");
        p!(w, "add.u32 %r23, %r23, %r8;");
    } else {
        p!(w, "add.u32 %r23, %r23, %r7;"); // m = warp base + gid
        p!(w, "mad.lo.u32 %r23, %r8, {sa}, %r23;"); // + tg * sa
    }
    p!(w, "shl.b32 %r23, %r23, 2;");
    p!(w, "mul.lo.u32 %r24, %r10, {warp_cols};");
    if !rhs_t {
        p!(w, "add.u32 %r24, %r24, %r7;"); // n = warp base + gid
        p!(w, "mad.lo.u32 %r24, %r8, {sb}, %r24;");
    } else {
        p!(w, "add.u32 %r24, %r24, %r7;");
        p!(w, "mul.lo.u32 %r24, %r24, {sb};");
        p!(w, "add.u32 %r24, %r24, %r8;");
    }
    p!(w, "shl.b32 %r24, %r24, 2;");
    p!(w, "add.u32 %r24, %r24, {a_bytes};");

    // Issue the copies of tile `t`: into stage offset %r18 with `kleft` %r19 elements left.
    let issue = |w: &mut Ptx| {
        for j in 0..a_per {
            p!(w, "setp.lt.s32 %p1, %ka{j}, %r19;");
            p!(w, "and.pred %p1, %p1, %va{j};");
            p!(w, "selp.b32 %r32, 16, 0, %p1;");
            p!(w, "add.u32 %r33, %r20, %r18;");
            p!(w, "add.u32 %r33, %r33, %da{j};");
            p!(w, "cp.async.cg.shared.global [%r33], [%ga{j}], 16, %r32;");
            p!(w, "add.u64 %ga{j}, %ga{j}, %rd10;");
        }
        for j in 0..b_per {
            p!(w, "setp.lt.s32 %p1, %kb{j}, %r19;");
            p!(w, "and.pred %p1, %p1, %vb{j};");
            p!(w, "selp.b32 %r32, 16, 0, %p1;");
            p!(w, "add.u32 %r33, %r20, %r18;");
            p!(w, "add.u32 %r33, %r33, %db{j};");
            p!(w, "add.u32 %r33, %r33, {a_bytes};");
            p!(w, "cp.async.cg.shared.global [%r33], [%gb{j}], 16, %r32;");
            p!(w, "add.u64 %gb{j}, %gb{j}, %rd11;");
        }
        p!(w, "sub.s32 %r19, %r19, {bk};");
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
    // Compute on the stage at %r17.
    p!(w, "add.u32 %r21, %r20, %r17;");
    p!(w, "add.u32 %r21, %r21, %r23;");
    p!(w, "add.u32 %r22, %r20, %r17;");
    p!(w, "add.u32 %r22, %r22, %r24;");
    for kk in 0..bk / 8 {
        // A fragments.
        for i in 0..mt {
            let regs: [(u32, u32); 4] = [(0, 0), (8, 0), (0, 4), (8, 4)]; // (m delta, k delta)
            for (q, (dm, dk)) in regs.into_iter().enumerate() {
                let off = if !lhs_t { ((i * 16 + dm) * sa + kk * 8 + dk) * 4 } else { ((kk * 8 + dk) * sa + i * 16 + dm) * 4 };
                p!(w, "ld.shared.b32 %fa{}, [%r21+{off}];", i * 4 + q as u32);
            }
        }
        for j in 0..nt {
            for (q, dk) in [0u32, 4].into_iter().enumerate() {
                let off = if !rhs_t { ((kk * 8 + dk) * sb + j * 8) * 4 } else { (j * 8 * sb + kk * 8 + dk) * 4 };
                p!(w, "ld.shared.b32 %fb{}, [%r22+{off}];", j * 2 + q as u32);
            }
        }
        // Bring the operands to TF32 (the tensor core drops the low 13 bits): multiplying by
        // `1 + 0.7 * 2^-11` before the truncation rounds to within 0.7 ulp, unbiased, with an RMS
        // error of 2.1e-4 (0.7 is the best factor; round to nearest has 2.0e-4, truncation 4.0e-4
        // and a bias of -3.4e-4). One multiplication per register, where adding half an ulp to
        // the bits (round to nearest) turns an infinity into a NaN and needs a guard that costs
        // 5 % of the product, and converting in shared memory exposes the latency of the copy.
        // Infinities and NaNs stay what they are; a magnitude within 3.4e-4 of the largest `f32`
        // overflows to an infinity.
        for reg in (0..mt * 4).map(|i| format!("%fa{i}")).chain((0..nt * 2).map(|j| format!("%fb{j}"))) {
            p!(w, "mov.b32 %f0, {reg};");
            p!(w, "mul.f32 %f1, %f0, 0f3F800B33;");
            p!(w, "mov.b32 {reg}, %f1;");
        }
        for i in 0..mt {
            for j in 0..nt {
                let c = (i * nt + j) * 4;
                p!(
                    w,
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{%c{}, %c{}, %c{}, %c{}}}, {{%fa{}, %fa{}, %fa{}, %fa{}}}, {{%fb{}, %fb{}}}, {{%c{}, %c{}, %c{}, %c{}}};",
                    c, c + 1, c + 2, c + 3, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3, j * 2, j * 2 + 1, c, c + 1, c + 2, c + 3
                );
            }
        }
    }
    // Next stage to read.
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
    p!(w, "mul.wide.u32 %rd4, %r26, %r25;"); // elements before the slice
    p!(w, "mul.lo.u32 %r27, %r9, {warp_rows};");
    p!(w, "add.u32 %r27, %r27, %r11;");
    p!(w, "add.u32 %r27, %r27, %r7;"); // row of fragment (0,0)
    p!(w, "mul.lo.u32 %r28, %r10, {warp_cols};");
    p!(w, "add.u32 %r28, %r28, %r12;");
    p!(w, "shl.b32 %r29, %r8, 1;");
    p!(w, "add.u32 %r28, %r28, %r29;"); // col of fragment (0,0)
    for i in 0..mt {
        for h in 0..2u32 {
            let row = i * 16 + h * 8;
            p!(w, "add.u32 %r30, %r27, {row};");
            p!(w, "setp.lt.u32 %p4, %r30, %r0;");
            p!(w, "mad.lo.u32 %r31, %r30, %r1, %r28;");
            p!(w, "cvt.u64.u32 %rd5, %r31;");
            p!(w, "add.u64 %rd5, %rd5, %rd4;");
            p!(w, "shl.b64 %rd5, %rd5, 2;");
            p!(w, "add.u64 %rd5, %rd5, %rd0;");
            for j in 0..nt {
                p!(w, "add.u32 %r32, %r28, {};", j * 8);
                p!(w, "setp.lt.u32 %p5, %r32, %r1;");
                p!(w, "and.pred %p5, %p5, %p4;");
                let c = (i * nt + j) * 4 + h * 2;
                p!(w, "@%p5 st.global.v2.f32 [%rd5+{}], {{%c{}, %c{}}};", j * 32, c, c + 1);
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

    #[test]
    fn every_tile_and_layout_generates_a_kernel() {
        for (rows, cols) in [(128, 128), (128, 64), (64, 128), (64, 64)] {
            for (lhs_transposed, rhs_transposed) in [(false, false), (true, false), (false, true), (true, true)] {
                let config = Config::of(&TensorSpec { rows, cols, lhs_transposed, rhs_transposed });
                let text = ptx(&config);
                assert!(text.contains(".entry main(") && text.trim_end().ends_with('}'));
                let (warp_rows, warp_cols) = (rows / config.wm, cols / config.wn);
                // One mma per fragment pair of each of the two k steps of a stage.
                let expected = (warp_rows / 16) * (warp_cols / 8) * (config.bk / 8);
                assert_eq!(text.matches("mma.sync").count() as u32, expected, "{rows}x{cols}");
                assert!(config.shared_bytes() <= 99 * 1024, "{rows}x{cols}: {} bytes of shared memory", config.shared_bytes());
                assert!(config.threads() <= 256);
            }
        }
    }

    #[test]
    fn staged_rows_are_padded_to_distinct_banks() {
        // The fragment loads of a warp (`gid` 0..8 across, `tg` 0..4 down) must hit 32 banks.
        let banks = |stride: u32, gid_step: u32, tg_step: u32| {
            let mut seen = std::collections::HashSet::new();
            (0..8).all(|gid| (0..4).all(|tg| seen.insert((gid * gid_step + tg * tg_step) % 32))) && stride > 0
        };
        for (rows, cols) in [(128, 128), (64, 64)] {
            for (lhs_transposed, rhs_transposed) in [(false, false), (true, true)] {
                let config = Config::of(&TensorSpec { rows, cols, lhs_transposed, rhs_transposed });
                let (sa, sb) = (config.a_stride(), config.b_stride());
                assert!(if lhs_transposed { banks(sa, 1, sa) } else { banks(sa, sa, 1) }, "A stride {sa}");
                assert!(if rhs_transposed { banks(sb, sb, 1) } else { banks(sb, 1, sb) }, "B stride {sb}");
            }
        }
    }
}
