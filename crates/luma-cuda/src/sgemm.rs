//! `f32` matrix product on the CUDA cores: SIMT kernels written as PTX (no NVVM), `cp.async`
//! pipelined like the tensor-core ones of [`crate::tensor`].
//!
//! The launch is that of `Builtin::Matmul` (grid `[column tiles, row tiles, slices of K]`, uniform
//! `{m, n, k, kc}`, the slice of grid layer `z` written at `z * m * n` of `c`) for the row-major
//! `M x K` and `K x N` matrices of the spec or their transposes. A workgroup computes a `bm x bn`
//! tile of `C`; each thread owns `tm x tn` outputs in registers, as groups of four consecutive
//! rows and columns so that every fragment load is one `ld.shared.v4`. Every output is a chain of
//! `fma.rn.f32` over `k` in ascending order from zero (within a slice of K), so results are
//! reproducible. `m`, `n` and `k` are multiples of 4.
//!
//! Shared memory holds each stage as `A[k][m]` and `B[k][n]` (`m` and `n` contiguous). An operand
//! stored along its rows (`A` as `M x K`, `B` as `N x K`) is copied with 4-byte `cp.async`
//! (transposing), the others with 16-byte ones; the rows of the transposing copies are padded by
//! 4 floats so that the 8 `k` of a warp's copies fall in different banks.

use luma_core::GemmSpec;

use crate::ptx::Ptx;

/// The shape of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    pub bm: u32,
    pub bn: u32,
    /// Outputs a thread owns along `M` and `N` (4 or 8).
    pub tm: u32,
    pub tn: u32,
    /// Warps along `M` and `N`.
    pub wm: u32,
    pub wn: u32,
    pub stages: u32,
    /// K elements staged per stage.
    pub bk: u32,
    pub lhs_t: bool,
    pub rhs_t: bool,
    /// Rows of tiles a group of consecutive workgroups covers (0: row-major order).
    pub group: u32,
    /// Workgroups per multiprocessor the registers are limited to fit (0: no limit).
    pub min_ctas: u32,
    /// Whether the fragments of step `k + 1` are loaded before the multiply-adds of step `k`.
    pub prefetch: bool,
}

/// The tiles of the kernels, largest first: `(rows, cols)`.
pub(crate) const TILES: [(u32, u32); 7] = [(128, 128), (128, 64), (64, 128), (64, 64), (64, 32), (32, 64), (32, 32)];

impl Config {
    pub(crate) fn of(spec: &GemmSpec) -> Config {
        // (tm, tn, warps along M, warps along N): a warp covers `8 tm` rows and `4 tn` columns.
        let (tm, tn, wm, wn) = match (spec.rows, spec.cols) {
            (128, 128) => (8, 8, 2, 4),
            (128, 64) => (8, 8, 2, 2),
            (64, 128) => (8, 8, 1, 4),
            (64, 64) => (8, 8, 1, 2),
            (64, 32) => (8, 4, 1, 2),
            (32, 64) => (4, 8, 1, 2),
            (32, 32) => (4, 4, 1, 2),
            (rows, cols) => panic!("no f32 kernel of {rows}x{cols}"),
        };
        // Debug: a thread owns 8 x 16 outputs (a warp covers 64 x 64) in the big tile.
        let (tm, tn, wm, wn) = if (spec.rows, spec.cols) == (128, 128) && std::env::var_os("LUMA_SG_WIDE").is_some() { (8, 16, 2, 2) } else { (tm, tn, wm, wn) };
        let mut config = Config { bm: spec.rows, bn: spec.cols, tm, tn, wm, wn, stages: 3, bk: 8, lhs_t: spec.lhs_transposed, rhs_t: spec.rhs_transposed, group: 8, min_ctas: 0, prefetch: true };
        if let Ok(text) = std::env::var("LUMA_SG_CFG") {
            // Debug override: `stages,bk,group,min_ctas,prefetch`.
            let parts: Vec<u32> = text.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if let [s, k, g, c, f] = parts[..] {
                config.stages = s;
                config.bk = k;
                config.group = g;
                config.min_ctas = c;
                config.prefetch = f != 0;
            }
        }
        config
    }

    pub(crate) fn threads(&self) -> u32 {
        self.wm * self.wn * 32
    }

    /// Floats per row (`k`) of the staged `A`: padded when it is copied element by element.
    fn a_stride(&self) -> u32 {
        if self.lhs_t { self.bm } else { self.bm + 4 }
    }

    fn b_stride(&self) -> u32 {
        if self.rhs_t { self.bn + 4 } else { self.bn }
    }

    fn a_bytes(&self) -> u32 {
        4 * self.bk * self.a_stride()
    }

    fn b_bytes(&self) -> u32 {
        4 * self.bk * self.b_stride()
    }

    fn stage_bytes(&self) -> u32 {
        self.a_bytes() + self.b_bytes()
    }

    /// Dynamic shared memory of a workgroup.
    pub(crate) fn shared_bytes(&self) -> u32 {
        self.stages * self.stage_bytes()
    }
}

/// One operand of the product as a thread stages it.
#[derive(Clone, Copy)]
struct Operand {
    /// `a` or `b`.
    name: &'static str,
    /// Rows (`m`) or columns (`n`) of the output tile it covers.
    extent: u32,
    /// Floats per row (`k`) of its staged tile.
    stride: u32,
    /// Stored along `k` (`A` as `M x K`, `B` as `N x K`): loaded 4 `k` at a time into registers and
    /// stored transposed; otherwise copied by 16-byte `cp.async`.
    staged: bool,
    /// Registers holding the base pointer, the tile origin and the extent of the whole matrix.
    base: &'static str,
    origin: &'static str,
    dim: &'static str,
    /// Copies (chunks of 4 floats) per thread.
    per_thread: u32,
}

/// The state of every copy of an operand a thread makes: global pointer, validity of its row (or
/// column), its `k` within the tile, and its offset in the staged tile (in bytes; for a staged
/// operand the offset of its first float, the next ones `stride` floats on).
fn setup(w: &mut Ptx, o: &Operand, threads: u32, bk: u32) {
    let (x, extent, stride) = (o.name, o.extent, o.stride);
    for j in 0..o.per_thread {
        p!(w, "add.u32 %r27, %r4, {};", j * threads);
        if o.staged {
            // Chunk of 4 consecutive `k` of row `r`: neighbouring threads take the two halves of
            // a row (bk 8), the rows follow, and the upper `k` of a deeper stage come last.
            p!(w, "and.b32 %r29, %r27, 1;");
            if bk > 8 {
                p!(w, "div.u32 %r28, %r27, {};", 2 * extent);
                p!(w, "shl.b32 %r28, %r28, 1;");
                p!(w, "or.b32 %r29, %r29, %r28;");
            }
            p!(w, "shr.u32 %r28, %r27, 1;");
            if bk > 8 {
                p!(w, "rem.u32 %r28, %r28, {extent};");
            }
            p!(w, "shl.b32 %r29, %r29, 2;"); // k of the chunk
            p!(w, "mov.u32 %k{x}{j}, %r29;");
            p!(w, "add.u32 %r30, {}, %r28;", o.origin);
            p!(w, "setp.lt.u32 %v{x}{j}, %r30, {};", o.dim);
            p!(w, "mad.lo.u32 %r31, %r30, %r2, %r13;");
            p!(w, "add.u32 %r31, %r31, %r29;");
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %g{x}{j}, {}, %rd3;", o.base);
            p!(w, "mad.lo.u32 %r31, %r29, {stride}, %r28;");
            p!(w, "shl.b32 %d{x}{j}, %r31, 2;");
        } else {
            // Chunk of 4 consecutive `m` (`n`) of row `k`.
            let cpr = extent / 4;
            p!(w, "shr.u32 %r28, %r27, {};", cpr.trailing_zeros());
            p!(w, "and.b32 %r29, %r27, {};", cpr - 1);
            p!(w, "shl.b32 %r29, %r29, 2;");
            p!(w, "mov.u32 %k{x}{j}, %r28;");
            p!(w, "add.u32 %r30, {}, %r29;", o.origin);
            p!(w, "setp.lt.u32 %v{x}{j}, %r30, {};", o.dim);
            p!(w, "add.u32 %r31, %r13, %r28;");
            p!(w, "mad.lo.u32 %r31, %r31, {}, %r30;", o.dim);
            p!(w, "mul.wide.u32 %rd3, %r31, 4;");
            p!(w, "add.u64 %g{x}{j}, {}, %rd3;", o.base);
            p!(w, "mad.lo.u32 %r31, %r28, {stride}, %r29;");
            p!(w, "shl.b32 %d{x}{j}, %r31, 2;");
        }
    }
}

/// The PTX of the kernel `main` for `config`.
pub(crate) fn ptx(config: &Config) -> String {
    let Config { bm, bn, tm, tn, wm: _, wn, stages, bk, lhs_t, rhs_t, group, min_ctas, prefetch } = *config;
    assert!(bk == 8 || bk == 16);
    let threads = config.threads();
    let (warp_rows, warp_cols) = (8 * tm, 4 * tn);
    let (ga, gb) = (tm / 4, tn / 4);
    let (sa, sb) = (config.a_stride(), config.b_stride());
    let (a_bytes, stage_bytes) = (config.a_bytes(), config.stage_bytes());
    let chunks = |extent: u32| extent * bk / 4;
    let a = Operand { name: "a", extent: bm, stride: sa, staged: !lhs_t, base: "%rd1", origin: "%r11", dim: "%r0", per_thread: chunks(bm) / threads };
    let b = Operand { name: "b", extent: bn, stride: sb, staged: rhs_t, base: "%rd2", origin: "%r12", dim: "%r1", per_thread: chunks(bn) / threads };
    assert!(a.per_thread >= 1 && b.per_thread >= 1 && a.per_thread * threads == chunks(bm) && b.per_thread * threads == chunks(bn), "{config:?}");
    let ops = [a, b];

    let mut w = Ptx { text: String::new() };
    w.text.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n");
    w.text.push_str(".extern .shared .align 16 .b8 smem[];\n\n");
    w.text.push_str(".visible .entry main(\n    .param .align 4 .b8 luma[16],\n    .param .u64 luma_indirect,\n    .param .u64 pc,\n    .param .u64 pa,\n    .param .u64 pb,\n    .param .align 16 .b8 pu[16]\n)\n");
    w.text.push_str(&format!(".maxntid {threads}, 1, 1\n"));
    if min_ctas > 0 {
        w.text.push_str(&format!(".minnctapersm {min_ctas}\n"));
    }
    w.text.push_str("{\n    .reg .pred %p<16>;\n    .reg .b32 %r<80>;\n    .reg .b64 %rd<64>;\n");
    w.text.push_str(&format!("    .reg .f32 %c<{}>;\n", tm * tn));
    for buffer in 0..2 {
        w.text.push_str(&format!("    .reg .f32 %fa{buffer}x<{tm}>;\n    .reg .f32 %fb{buffer}x<{tn}>;\n"));
    }
    for o in &ops {
        let (x, n) = (o.name, o.per_thread);
        w.text.push_str(&format!("    .reg .b64 %g{x}<{n}>;\n    .reg .b32 %d{x}<{n}>;\n    .reg .b32 %k{x}<{n}>;\n    .reg .pred %v{x}<{n}>;\n"));
        if o.staged {
            // Registers of the tile in flight, `s{x}{j}x{q}`.
            for j in 0..n {
                w.text.push_str(&format!("    .reg .f32 %s{x}{j}x<4>;\n"));
            }
        }
    }
    // r0 m, r1 n, r2 k, r3 kc, r4 tid, r5 lane, r6 warp, r7 lane row, r8 lane column, r9 warp row,
    // r10 warp column, r11 bm0, r12 bn0, r13 kbeg, r14 klen, r15 ktiles, r16 kt, r17 read stage,
    // r18 write stage of the `cp.async` copies, r19 kleft of those, r20 smem base, r21 A fragment
    // base, r22 B fragment base, r23 A thread offset, r24 B thread offset, r38 k left of the
    // staged tile, r39 stage the staged tile goes to.
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
    p!(w, "sub.u32 %r14, %r26, %r13;");
    p!(w, "add.u32 %r15, %r14, {};", bk - 1);
    p!(w, "shr.u32 %r15, %r15, {};", bk.trailing_zeros());
    p!(w, "mov.u32 %r20, smem;");
    for i in 0..tm * tn {
        p!(w, "mov.f32 %c{i}, 0f00000000;");
    }
    for o in &ops {
        setup(&mut w, o, threads, bk);
    }
    // Bytes the global pointers advance per tile.
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
    // Fragment bases (bytes within a stage): `A[k][warp row + 4 lane row]`, `B[k][warp column + 4 lane column]`.
    p!(w, "mul.lo.u32 %r23, %r9, {warp_rows};");
    p!(w, "mad.lo.u32 %r23, %r7, 4, %r23;");
    p!(w, "shl.b32 %r23, %r23, 2;");
    p!(w, "mul.lo.u32 %r24, %r10, {warp_cols};");
    p!(w, "mad.lo.u32 %r24, %r8, 4, %r24;");
    p!(w, "shl.b32 %r24, %r24, 2;");
    p!(w, "add.u32 %r24, %r24, {a_bytes};");

    // `cp.async` copies of the operands that are not staged, into the write stage %r18 with %r19
    // elements of K left.
    let issue = |w: &mut Ptx| {
        p!(w, "add.u32 %r35, %r20, %r18;");
        for o in ops.iter().filter(|o| !o.staged) {
            let (x, step) = (o.name, if o.name == "a" { "%rd10" } else { "%rd11" });
            let region = if o.name == "a" { 0 } else { a_bytes };
            for j in 0..o.per_thread {
                p!(w, "setp.lt.s32 %p1, %k{x}{j}, %r19;");
                p!(w, "and.pred %p1, %p1, %v{x}{j};");
                p!(w, "selp.b32 %r32, 16, 0, %p1;");
                p!(w, "add.u32 %r33, %r35, %d{x}{j};");
                if region != 0 {
                    p!(w, "add.u32 %r33, %r33, {region};");
                }
                p!(w, "cp.async.cg.shared.global [%r33], [%g{x}{j}], 16, %r32;");
                p!(w, "add.u64 %g{x}{j}, %g{x}{j}, {step};");
            }
        }
        p!(w, "sub.s32 %r19, %r19, {bk};");
        p!(w, "add.u32 %r18, %r18, {stage_bytes};");
        p!(w, "setp.ge.u32 %p2, %r18, {};", stages * stage_bytes);
        p!(w, "@%p2 mov.u32 %r18, 0;");
    };
    // The next tile of the staged operands into registers (zeros past the slice or the matrix).
    let load = |w: &mut Ptx| {
        for o in ops.iter().filter(|o| o.staged) {
            let (x, step) = (o.name, if o.name == "a" { "%rd10" } else { "%rd11" });
            for j in 0..o.per_thread {
                for q in 0..4 {
                    p!(w, "mov.f32 %s{x}{j}x{q}, 0f00000000;");
                }
                p!(w, "setp.lt.s32 %p1, %k{x}{j}, %r38;");
                p!(w, "and.pred %p1, %p1, %v{x}{j};");
                p!(w, "@%p1 ld.global.v4.f32 {{%s{x}{j}x0, %s{x}{j}x1, %s{x}{j}x2, %s{x}{j}x3}}, [%g{x}{j}];");
                p!(w, "add.u64 %g{x}{j}, %g{x}{j}, {step};");
            }
        }
        p!(w, "sub.s32 %r38, %r38, {bk};");
    };
    // Stores them transposed into the stage at %r39.
    let store = |w: &mut Ptx| {
        p!(w, "add.u32 %r40, %r20, %r39;");
        for o in ops.iter().filter(|o| o.staged) {
            let (x, stride) = (o.name, o.stride);
            let region = if o.name == "a" { 0 } else { a_bytes };
            for j in 0..o.per_thread {
                p!(w, "add.u32 %r41, %r40, %d{x}{j};");
                for q in 0..4 {
                    p!(w, "st.shared.f32 [%r41+{}], %s{x}{j}x{q};", region + q * stride * 4);
                }
            }
        }
    };
    let staged = ops.iter().any(|o| o.staged);
    p!(w, "mov.u32 %r18, 0;");
    p!(w, "mov.s32 %r19, %r14;");
    p!(w, "mov.s32 %r38, %r14;");
    for _ in 0..stages - 1 {
        issue(&mut w);
        p!(w, "cp.async.commit_group;");
    }
    if staged {
        // Tile 0 of the staged operands goes to stage 0.
        load(&mut w);
        p!(w, "mov.u32 %r39, 0;");
        store(&mut w);
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
    if staged {
        load(&mut w);
    }
    p!(w, "add.u32 %r21, %r20, %r17;");
    p!(w, "add.u32 %r22, %r21, %r24;");
    p!(w, "add.u32 %r21, %r21, %r23;");
    // Fragment loads of step `kk` into register set `kk % 2`; those of the next step are issued
    // before the multiply-adds of this one.
    let frags = |w: &mut Ptx, kk: u32| {
        let set = kk % 2;
        for g in 0..ga {
            let off = (kk * sa + g * 32) * 4;
            let r = g * 4;
            p!(w, "ld.shared.v4.f32 {{%fa{set}x{}, %fa{set}x{}, %fa{set}x{}, %fa{set}x{}}}, [%r21+{off}];", r, r + 1, r + 2, r + 3);
        }
        for h in 0..gb {
            let off = (kk * sb + h * 16) * 4;
            let r = h * 4;
            p!(w, "ld.shared.v4.f32 {{%fb{set}x{}, %fb{set}x{}, %fb{set}x{}, %fb{set}x{}}}, [%r22+{off}];", r, r + 1, r + 2, r + 3);
        }
    };
    if prefetch {
        frags(&mut w, 0);
    }
    for kk in 0..bk {
        if prefetch {
            if kk + 1 < bk {
                frags(&mut w, kk + 1);
            }
        } else {
            frags(&mut w, kk);
        }
        let set = kk % 2;
        for i in 0..tm {
            for j in 0..tn {
                p!(w, "fma.rn.f32 %c{0}, %fa{set}x{i}, %fb{set}x{j}, %c{0};", i * tn + j);
            }
        }
    }
    if staged {
        // The next tile goes to the stage after the one just read.
        p!(w, "add.u32 %r39, %r17, {stage_bytes};");
        p!(w, "setp.ge.u32 %p2, %r39, {};", stages * stage_bytes);
        p!(w, "@%p2 mov.u32 %r39, 0;");
        store(&mut w);
    }
    p!(w, "add.u32 %r17, %r17, {stage_bytes};");
    p!(w, "setp.ge.u32 %p2, %r17, {};", stages * stage_bytes);
    p!(w, "@%p2 mov.u32 %r17, 0;");
    p!(w, "add.u32 %r16, %r16, 1;");
    p!(w, "setp.lt.u32 %p3, %r16, %r15;");
    p!(w, "@%p3 bra LOOP;");
    w.text.push_str("DONE:\n");
    p!(w, "cp.async.wait_group 0;");

    // Epilogue: C[z * m * n + row * n + col], rows in groups of four, columns in groups of four.
    p!(w, "mov.u32 %r25, %ctaid.z;");
    p!(w, "mul.lo.u32 %r26, %r0, %r1;");
    p!(w, "mul.wide.u32 %rd4, %r26, %r25;");
    p!(w, "mul.lo.u32 %r27, %r9, {warp_rows};");
    p!(w, "add.u32 %r27, %r27, %r11;");
    p!(w, "mad.lo.u32 %r27, %r7, 4, %r27;");
    p!(w, "mul.lo.u32 %r28, %r10, {warp_cols};");
    p!(w, "add.u32 %r28, %r28, %r12;");
    p!(w, "mad.lo.u32 %r28, %r8, 4, %r28;");
    for g in 0..ga {
        for ii in 0..4u32 {
            let i = g * 4 + ii;
            p!(w, "add.u32 %r30, %r27, {};", g * 32 + ii);
            p!(w, "setp.lt.u32 %p4, %r30, %r0;");
            p!(w, "mad.lo.u32 %r31, %r30, %r1, %r28;");
            p!(w, "cvt.u64.u32 %rd5, %r31;");
            p!(w, "add.u64 %rd5, %rd5, %rd4;");
            p!(w, "shl.b64 %rd5, %rd5, 2;");
            p!(w, "add.u64 %rd5, %rd5, %rd0;");
            for h in 0..gb {
                p!(w, "add.u32 %r32, %r28, {};", h * 16);
                p!(w, "setp.lt.u32 %p5, %r32, %r1;");
                p!(w, "and.pred %p5, %p5, %p4;");
                let c = i * tn + h * 4;
                p!(w, "@%p5 st.global.v4.f32 [%rd5+{}], {{%c{}, %c{}, %c{}, %c{}}};", h * 64, c, c + 1, c + 2, c + 3);
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
        for (rows, cols) in TILES {
            for (lhs_transposed, rhs_transposed) in [(false, false), (true, false), (false, true), (true, true)] {
                let config = Config::of(&GemmSpec { rows, cols, lhs_transposed, rhs_transposed, exact: true });
                let text = ptx(&config);
                assert!(text.contains(".entry main(") && text.trim_end().ends_with('}'));
                // One multiply-add per output per `k` of a stage.
                assert_eq!(text.matches("fma.rn.f32").count() as u32, config.tm * config.tn * config.bk, "{rows}x{cols}");
                assert!(config.shared_bytes() <= 99 * 1024, "{rows}x{cols}: {} bytes of shared memory", config.shared_bytes());
                assert_eq!(config.bm, config.wm * 8 * config.tm);
                assert_eq!(config.bn, config.wn * 4 * config.tn);
            }
        }
    }
}
