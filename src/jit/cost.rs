//! The cost model: what a launch, a byte of memory traffic and an arithmetic operation cost on a
//! device. Measured once per device (see [`calibrate`]) or taken from the tuning profile's fixed
//! numbers ([`CostModel::fallback`]); the fusion decisions of [`super::inline`] weigh
//! recomputation against storing with it.

use std::time::Instant;

use crate::builtin::{df_library, BinOp, CmpOp, Part, UnOp};
use crate::context::{Context, RawArg, STORAGE_USAGE};
use crate::error::Result;
use crate::kernel::{build_core, KernelKey};
use crate::ops::launch_1d;

pub(crate) use luma_core::{CostModel, MemoryModel};

use super::{Edge, ElemTy, Gen, Op, OpKind, ShiftKind};

/// The cost of `op` per element in f32-operation units.
pub(crate) fn op_weight(op: &Op, cost: &CostModel) -> f64 {
    let base = match op.kind {
        OpKind::Unary(u) => match u {
            UnOp::Neg | UnOp::Conj | UnOp::Abs | UnOp::Sign | UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Round | UnOp::Id | UnOp::BitNot | UnOp::Shl(_) | UnOp::Shr(_) | UnOp::Rotl(_) => 1.0,
            UnOp::Sqrt => 4.0,
            UnOp::Exp | UnOp::Ln | UnOp::Sin | UnOp::Cos => 8.0,
            UnOp::Tan | UnOp::Tanh => 16.0,
            UnOp::Asin | UnOp::Acos | UnOp::Atan | UnOp::Sinh | UnOp::Cosh | UnOp::Asinh | UnOp::Acosh | UnOp::Atanh => 16.0,
            UnOp::Exp2 | UnOp::Log2 | UnOp::Log10 | UnOp::Rsqrt | UnOp::Sigmoid => 8.0,
            UnOp::Cbrt => 12.0,
            UnOp::Erf | UnOp::Erfc => 24.0,
            UnOp::Lgamma | UnOp::Tgamma => 32.0,
            UnOp::Log1p | UnOp::Expm1 => 20.0,
        },
        OpKind::Binary(b) => match b {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Min | BinOp::Max | BinOp::Copysign => 1.0,
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr | BinOp::Rotl => 1.0,
            BinOp::Div | BinOp::Rem => 4.0,
            BinOp::Mod => 8.0,
            BinOp::Hypot => 6.0,
            BinOp::Atan2 => 16.0,
        },
        OpKind::Powf => 16.0,
        OpKind::Compare(c) => match c {
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge | CmpOp::Eq | CmpOp::Ne => 1.0,
        },
        OpKind::Select => 1.0,
        OpKind::Part(part) => match part {
            Part::Real | Part::Imag => 1.0,
            Part::NormSqr => 2.0,
            Part::Abs => 10.0,
            Part::Arg => 16.0,
        },
        OpKind::Make => 1.0,
        OpKind::Classify(_) => 2.0,
        OpKind::Clamp => 2.0,
        OpKind::Cast(to) => {
            if to == ElemTy::Df32 || op.ty == ElemTy::Df32 {
                4.0
            } else {
                1.0
            }
        }
        OpKind::Shift(ShiftKind::Dims { .. } | ShiftKind::ParamDims { .. } | ShiftKind::Linear) => 4.0,
        OpKind::Shift(ShiftKind::View { edge, .. }) => 6.0 + if edge != Edge::None { 2.0 } else { 0.0 },
        OpKind::Shift(ShiftKind::ParamView { spec, .. }) => 6.0 + if spec.edge != Edge::None { 2.0 } else { 0.0 },
        OpKind::Shift(ShiftKind::Take { .. } | ShiftKind::ParamTake { .. }) => 2.0,
        OpKind::Generate(Gen::Index | Gen::RangePow2 | Gen::RangeConst { .. } | Gen::ParamRange { .. } | Gen::Linspace) => 2.0,
        OpKind::Generate(Gen::Bits | Gen::Uniform | Gen::Normal) => 60.0,
    };
    // The double-float math library takes about this many double-float operations (see
    // `builtin::DF_MATH_LIBRARY`).
    let base = match (op.ty, op.kind) {
        (ElemTy::Df32, OpKind::Unary(UnOp::Exp)) => 20.0,
        (ElemTy::Df32, OpKind::Unary(UnOp::Ln)) => 16.0,
        (ElemTy::Df32, OpKind::Unary(UnOp::Tanh)) => 24.0,
        (ElemTy::Df32, OpKind::Unary(UnOp::Log1p | UnOp::Expm1)) => 24.0,
        (ElemTy::Df32, OpKind::Powf) => 40.0,
        // The components of a complex number are separate `f32` operations.
        (ElemTy::C32, OpKind::Unary(UnOp::Neg | UnOp::Conj) | OpKind::Binary(BinOp::Add | BinOp::Sub)) => 2.0,
        (ElemTy::C32, OpKind::Binary(BinOp::Mul)) => 4.0,
        (ElemTy::C32, OpKind::Binary(BinOp::Div)) => 14.0,
        (ElemTy::C32, OpKind::Unary(UnOp::Exp)) => 24.0,
        (ElemTy::C32, OpKind::Unary(UnOp::Sqrt)) => 18.0,
        _ => base,
    };
    let arithmetic = matches!(op.kind, OpKind::Unary(_) | OpKind::Binary(_) | OpKind::Clamp | OpKind::Compare(_) | OpKind::Powf)
        && !matches!(op.kind, OpKind::Unary(UnOp::Id));
    match op.ty {
        ElemTy::Df32 if arithmetic => base * cost.df32_factor,
        ElemTy::F64 if arithmetic => base * cost.f64_factor.unwrap_or(32.0),
        _ => base,
    }
}

/// Estimated nanoseconds of one kernel moving `bytes` and doing `ops` f32-operation units of work.
pub(crate) fn kernel_ns(cost: &CostModel, bytes: f64, ops: f64) -> f64 {
    cost.launch_us * 1e3 + (bytes / cost.bandwidth_gbs).max(ops / cost.f32_ops_per_ns)
}

/// Invocations per f32 operation per nanosecond of throughput at which a kernel keeps the
/// device's arithmetic units busy (a device that does more per nanosecond needs more invocations
/// in flight: 2 x 16 000 for an RTX 3090). Fewer leave them idle: the work of every invocation
/// runs in sequence, so a kernel of 20 000 cells does not run at the throughput calibration
/// measures over a million, and a kernel that staged a region in workgroup tiles of 512 cells
/// has half the invocations of one of 256.
const SATURATION_PER_OP_RATE: f64 = 2.0;

/// [`kernel_ns`] for a kernel that runs `invocations` invocations (see [`SATURATION_PER_OP_RATE`]).
pub(crate) fn kernel_ns_on(cost: &CostModel, bytes: f64, ops: f64, invocations: f64) -> f64 {
    let busy = (invocations / (SATURATION_PER_OP_RATE * cost.f32_ops_per_ns)).clamp(1e-3, 1.0);
    cost.launch_us * 1e3 + (bytes / cost.bandwidth_gbs).max(ops / (cost.f32_ops_per_ns * busy))
}

/// The 2-D reads of a kernel that could go through the texture units, as the decision of
/// [`read_path`] sees them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureRead {
    /// How the kernel finds the texels it reads.
    pub pattern: ReadPattern,
    /// Source texels an output element combines (4 for a bilinear method, 16 for a bicubic one).
    pub taps: u32,
    /// Output elements.
    pub outputs: u64,
    /// Bytes of the source image.
    pub src_bytes: u64,
    /// Whether a texture over the source's own buffer is possible: no copy to make it.
    pub zero_copy: bool,
}

/// Where the 2-D position of a read comes from, which decides how the reads of neighbouring
/// invocations overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadPattern {
    /// A position array: any pattern, with the locality of the data (`approx2`).
    Positions,
    /// A map of the output pixel (`transform`, `rotate`): neighbours read neighbouring texels.
    Warp,
    /// The output pixel over a scale (`resize`): the taps of a column are shared by all its rows.
    Scaled,
}

/// How the source of a [`TextureRead`] is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadPath {
    /// Loads from the buffer (on NVIDIA GPUs through the read-only data cache, which a
    /// `read`-only binding already gets).
    Plain,
    /// Gathers (`textureGather`: the four texels of a bilinear footprint in one fetch) from a
    /// texture; unfiltered, so exact.
    Texture,
}

/// The share of a kernel's time that gathers save over loads on the texture-capable NVIDIA parts
/// measured (RTX 3090 and RTX 3080 Laptop, the sampling kernel of `crate::sampler` with its
/// per-method specialisation, best of interleaved runs; differences within 2% are noise and count
/// as 0). A warp reads the four taps of a bilinear footprint with one gather instead of four
/// loads (4K: `rotate` 5%, `transform` 7-13%) and sixteen bicubic taps with four (26-28%, 1080p
/// 25-27%); scattered `approx2` positions (within 48 pixels of the pixel, the reads of neighbouring
/// invocations overlap only in the cache) gain 28-36% at bicubic and 1-3% at bilinear. Smooth
/// `approx2` positions and `resize` are bound by DRAM, which a texture does not touch: no
/// difference at any method.
///
/// The same measurements for the kernels the JIT generates (hand-written equivalents of
/// `take` with 2-D data-dependent coordinates, 4-tap bilinear gathers from integer or computed
/// positions, lookup tables, clamped 3 x 3 windows, 512x512 to 4096x4096 sources, 1 to 16 outputs per
/// texel, texture point loads and gathers against buffer loads): they are bound by DRAM or by
/// the index arithmetic, texture reads are between 0.84 and 1.0 of the buffer's (windows: 0.71-0.97,
/// the texture's out-of-range read is 0, so the clamp stays in the kernel) with one outlier, 1.13
/// on a 512x512 source of the laptop part. The JIT never reads through a texture.
fn gather_saving(pattern: ReadPattern, taps: u32) -> f64 {
    match (pattern, taps) {
        (ReadPattern::Positions, 16) => 0.25,
        (ReadPattern::Warp, 4) => 0.05,
        (ReadPattern::Warp, 16) => 0.25,
        _ => 0.0,
    }
}

/// Whether `read` goes through a texture: the device must have one for the data, the kernel's
/// saving must beat the cost of making the texture. A texture over the source's own buffer costs
/// nothing (and is kept for the array's next reader); one that copies the source costs a read and
/// a write of it and a launch, and is made per use, so it pays only when the saving exceeds that.
/// The saving is the table of [`gather_saving`] where measured (NVIDIA); every other device reads
/// from the buffer.
pub(crate) fn read_path(ctx: &Context, read: &TextureRead) -> ReadPath {
    if ctx.hal().texture_caps().is_none() || ctx.tuning().profile != luma_core::TuningProfile::Nvidia {
        return ReadPath::Plain;
    }
    let cost = ctx.cost_model();
    // The kernel's traffic: the source once, the output, and positions where there are some.
    let out_bytes = read.outputs as f64 * 4.0;
    let pos_bytes = if read.pattern == ReadPattern::Positions { read.outputs as f64 * 8.0 } else { 0.0 };
    let work_ns = (read.src_bytes as f64 + out_bytes + pos_bytes) / cost.bandwidth_gbs;
    let saving_ns = gather_saving(read.pattern, read.taps) * work_ns;
    let view_ns = if read.zero_copy { 0.0 } else { cost.launch_us * 1e3 + 2.0 * read.src_bytes as f64 / cost.bandwidth_gbs };
    if saving_ns > view_ns && saving_ns > 0.0 { ReadPath::Texture } else { ReadPath::Plain }
}

/// Times of calibration kernels are the best of this many measurements.
const BEST_OF: usize = 5;
/// Launches recorded before one `sync` in a measurement, amortising submission and the wait.
const REPS: u32 = 4;
/// Launches of the launch-overhead measurement.
const LAUNCHES: u32 = 64;
/// Invocations per workgroup of the calibration kernels.
const WORKGROUP: u32 = 64;

/// WGSL of a calibration kernel: reads `in0`, writes `out0`, loops `p.k` times over `loop_body`
/// on a value `x` of type `ty` initialised from `init`, stores `store`.
fn kernel_source(prelude: &str, ty: &str, init: &str, body: &str, store: &str) -> String {
    format!(
        "struct Params {{ k: u32, g: u32 }}
@group(1) @binding(0) var<storage, read> in0: array<f32>;
@group(1) @binding(1) var<storage, read_write> out0: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
{prelude}
@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    var x: {ty} = {init};
    for (var k = 0u; k < p.k; k++) {{ {body} }}
    out0[i] = {store};
}}
"
    )
}

/// A calibration kernel with its buffers, so that several parameter values are timed without
/// compiling or allocating again.
struct Rig {
    core: crate::kernel::KernelCore,
    input: luma_core::Buffer,
    output: luma_core::Buffer,
    threads: u32,
    bytes: u64,
}

impl Rig {
    /// `source` over `threads` invocations with whole device buffers of `elems` elements (not
    /// array storage, so the measurement does not depend on the context's chunk size).
    fn new(ctx: &Context, source: String, label: &str, threads: u32, elems: u32) -> Result<Rig> {
        let core = build_core(ctx, &KernelKey::new(source, "main", None), Some(label), luma_core::KernelTrust::Trusted)?;
        let bytes = u64::from(elems) * 4;
        let input = ctx.alloc_buffer(bytes, STORAGE_USAGE)?;
        let output = match ctx.alloc_buffer(bytes, STORAGE_USAGE) {
            Ok(output) => output,
            Err(err) => {
                ctx.recycle(input);
                return Err(err);
            }
        };
        Ok(Rig { core, input, output, threads, bytes })
    }

    /// Best wall-clock nanoseconds per launch with uniform `{k, g}`: one warm-up measurement,
    /// then the best of [`BEST_OF`], each `reps` launches and one `sync`.
    fn time(&self, ctx: &Context, k: u32, g: u32, reps: u32) -> Result<f64> {
        self.time_best_of(ctx, k, g, reps, BEST_OF)
    }

    /// [`Rig::time`] with the best of `best_of` measurements.
    fn time_best_of(&self, ctx: &Context, k: u32, g: u32, reps: u32, best_of: usize) -> Result<f64> {
        let words = [k, g];
        let raw = [RawArg::read(&self.input, self.bytes), RawArg::write(&self.output, self.bytes), RawArg::Uniform(bytemuck::cast_slice(&words))];
        let mut best = f64::INFINITY;
        for round in 0..=best_of {
            let start = Instant::now();
            for _ in 0..reps {
                launch_1d(ctx, &self.core, self.threads, &raw)?;
            }
            ctx.sync()?;
            let ns = start.elapsed().as_secs_f64() * 1e9 / f64::from(reps);
            if round > 0 {
                best = best.min(ns);
            }
        }
        Ok(best)
    }

    fn release(self, ctx: &Context) {
        ctx.recycle(self.input);
        ctx.recycle(self.output);
    }
}

/// Best wall-clock nanoseconds per launch of a kernel over `n` elements with uniform `{k, 0}`
/// (see [`Rig::time`]); the input and output are `n` elements.
fn time_kernel(ctx: &Context, source: String, label: &str, n: u32, k: u32, reps: u32) -> Result<f64> {
    let rig = Rig::new(ctx, source, label, n, n)?;
    let best = rig.time(ctx, k, 0, reps);
    rig.release(ctx);
    best
}

/// The coalesced, strided, gathered and transposed copy of the memory measurement: `out0[i] =
/// in0[j]`, with `j` chosen by `p.g` (a stride, `0` for a hash of `i`, `0xffffffff` for the
/// transposed index of a `p.k` x `p.k` matrix). One kernel for all, so the factors compare
/// kernels that differ only in the index.
const PATTERN: &str = "
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    let n = luma.size.x;
    if (i >= n) { return; }
    var j = i;
    if (p.g == 0u) {
        var h = i * 747796405u + 2891336453u;
        h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        j = ((h >> 22u) ^ h) & (n - 1u);
    } else if (p.g == 0xffffffffu) {
        j = (i % p.k) * p.k + i / p.k;
    } else {
        j = (i * p.g) & (n - 1u);
    }
    out0[i] = in0[j];
}
";

/// A copy over a working set of `p.g` words: every invocation sweeps the set `p.k` times with a
/// stride of the number of invocations, so the invocations together read the whole set once per
/// pass and a word is read again only after the rest of the set (the reuse distance is the set
/// size, which is what makes the bandwidth drop where the set leaves L2). Each invocation writes
/// its sum to its own word of `out0` (invocations writing one word would make the CPU backend's
/// threads fight over its line).
const WORKING_SET: &str = "
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.x + gid.y * nwg.x * 64u;
    if (t >= luma.size.x) { return; }
    let stride = luma.size.x;
    var sum = 0.0;
    for (var rep = 0u; rep < p.k; rep++) {
        for (var j = t; j < p.g; j += stride) { sum += in0[j]; }
    }
    out0[t] = sum;
}
";

/// One workgroup of 256 per compute unit reads a 16 KiB workgroup array `4 * p.k` times per
/// invocation, four independent reads per step, lanes on consecutive words (no bank conflicts).
const SHARED: &str = "
var<workgroup> tile: array<f32, 4096>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    for (var j = lid.x; j < 4096u; j += 256u) { tile[j] = in0[j]; }
    workgroupBarrier();
    var a = 0.0; var b = 0.0; var c = 0.0; var d = 0.0;
    for (var k = 0u; k < p.k; k++) {
        let at = (lid.x + k * 33u) & 4095u;
        a += tile[at];
        b += tile[(at + 1024u) & 4095u];
        c += tile[(at + 2048u) & 4095u];
        d += tile[(at + 3072u) & 4095u];
    }
    if (i < luma.size.x) { out0[i] = a + b + c + d; }
}
";

/// A chain of `p.k` dependent loads from the first 1024 words; `p.g` is 0 at run time, so the
/// next index depends on the loaded word without the compiler seeing it.
const CHAIN: &str = "
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    var idx = 0u;
    for (var k = 0u; k < p.k; k++) { idx = (idx + 1u + (bitcast<u32>(in0[idx]) & p.g)) & 1023u; }
    out0[gid.x] = f32(idx);
}
";

/// A working set inside L2 moves at least this much faster than DRAM.
const L2_FASTER: f64 = 1.5;
/// Invocations of the L2 sweep, and the launches and measurements per working set.
const SWEEP_THREADS: u32 = 1 << 17;
const SWEEP_REPS: u32 = 16;
/// Dependent loads of the latency chain.
const CHAIN_STEPS: u32 = 4096;
/// Reads per invocation of the workgroup-memory kernel (times 4).
const SHARED_STEPS: u32 = 2048;
/// The strides of [`MemoryModel::strided_factor`], in elements.
const STRIDES: [u32; 3] = [2, 8, 32];

fn with_header(body: &str) -> String {
    format!(
        "struct Params {{ k: u32, g: u32 }}
@group(1) @binding(0) var<storage, read> in0: array<f32>;
@group(1) @binding(1) var<storage, read_write> out0: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
{body}"
    )
}

/// Measures how bytes move inside the device (see [`MemoryModel`]): the copy bandwidth over
/// working sets around the L2 size, workgroup-memory bandwidth, the cost of strided, gathered and
/// transposed reads relative to coalesced ones, and the latency of a dependent load. `dram_gbs` is
/// the bandwidth measured over a large array and `launch_ns` the cost of an empty launch.
fn measure_memory(ctx: &Context, dram_gbs: f64, launch_ns: f64) -> Result<MemoryModel> {
    let cpu = ctx.is_cpu();
    let cap = u32::try_from(ctx.max_buffer_bytes() / 4).unwrap_or(u32::MAX);

    // The L2 sweep: `SWEEP_THREADS` invocations read a working set of `ws` words over and over,
    // `traffic` words in all (at least one pass).
    let traffic: u32 = if cpu { 1 << 22 } else { 1 << 24 }.min(cap);
    let attribute = ctx.info().l2_cache_bytes;
    let sizes: Vec<u64> = match attribute {
        Some(l2) => [l2 / 4, l2 / 2, l2, l2 * 2, l2 * 4].to_vec(),
        None => (20..=26).map(|shift| 1u64 << shift).collect(),
    };
    let rig = Rig::new(ctx, with_header(WORKING_SET), "CostL2", SWEEP_THREADS, traffic)?;
    let mut curve = Vec::new();
    let mut failed = None;
    for bytes in sizes {
        let words = bytes / 4;
        if words < u64::from(SWEEP_THREADS) || words > u64::from(traffic) {
            continue;
        }
        let passes = (u64::from(traffic) / words).max(1) as u32;
        match rig.time_best_of(ctx, passes, words as u32, SWEEP_REPS, 3) {
            Ok(t) => curve.push((bytes, 4.0 * words as f64 * f64::from(passes) / t)),
            Err(err) => {
                failed = Some(err);
                break;
            }
        }
    }
    rig.release(ctx);
    if let Some(err) = failed {
        return Err(err);
    }
    let inside: Vec<usize> = (0..curve.len()).filter(|&i| curve[i].1 >= L2_FASTER * dram_gbs).collect();
    let l2_gbs = match inside.last() {
        Some(&i) => curve[i].1,
        None => curve.iter().map(|(_, gbs)| *gbs).fold(0.0, f64::max),
    };
    // Without a device attribute the size is the knee, taken only where it is sharp: the next
    // size up (a doubling) is at least `L2_FASTER` times slower.
    let l2_bytes = attribute.or_else(|| {
        let &i = inside.last()?;
        let next = curve.get(i + 1)?;
        (next.0 == 2 * curve[i].0 && next.1 * L2_FASTER <= curve[i].1).then_some(curve[i].0)
    });

    // Access patterns over one array of a power-of-four number of words.
    let n_pat = 1u32 << (traffic.ilog2() & !1);
    let side = 1u32 << (n_pat.ilog2() / 2);
    let rig = Rig::new(ctx, with_header(PATTERN), "CostPattern", n_pat, n_pat)?;
    let patterns = (|| -> Result<[f64; 5]> {
        let before = rig.time_best_of(ctx, side, 1, 2, 3)?;
        let mut times = [0.0; 5];
        for (slot, stride) in times.iter_mut().zip(STRIDES) {
            *slot = rig.time_best_of(ctx, side, stride, 2, 3)?;
        }
        times[3] = rig.time_best_of(ctx, side, 0, 2, 3)?;
        times[4] = rig.time_best_of(ctx, side, u32::MAX, 2, 3)?;
        // Coalesced again at the end: a device (or the CPU backend) that is still warming up at
        // the start measures the first kernel slow.
        let coalesced = before.min(rig.time_best_of(ctx, side, 1, 2, 3)?);
        Ok(times.map(|t| t / coalesced))
    })();
    rig.release(ctx);
    let patterns = patterns?;

    // Workgroup memory: one workgroup per compute unit.
    let shared_gbs = if cpu {
        None
    } else {
        let groups = ctx.info().compute_units.unwrap_or(16).max(1);
        let threads = groups * 256;
        let rig = Rig::new(ctx, with_header(SHARED), "CostShared", threads, threads.max(4096))?;
        let t = rig.time(ctx, SHARED_STEPS, 0, REPS);
        rig.release(ctx);
        let t = (t? - launch_ns).max(1.0);
        Some(f64::from(threads) * f64::from(SHARED_STEPS) * 16.0 / t)
    };

    // Latency: one workgroup, one dependent chain per invocation.
    let rig = Rig::new(ctx, with_header(CHAIN), "CostChain", 64, 1024)?;
    let t = rig.time_best_of(ctx, CHAIN_STEPS, 0, 8, 3);
    rig.release(ctx);
    let latency_ns = (t? - launch_ns).max(launch_ns * 0.1) / f64::from(CHAIN_STEPS);

    Ok(MemoryModel::new(l2_bytes, attribute.is_some() && l2_bytes.is_some(), l2_gbs, shared_gbs, [patterns[0], patterns[1], patterns[2]], patterns[3], patterns[4], latency_ns))
}

/// Measures the device: bandwidth, simple f32 throughput, the cost of `Df32` and `f64`
/// arithmetic relative to it, and the cost of a launch.
pub(crate) fn calibrate(ctx: &Context) -> Result<CostModel> {
    let cpu = ctx.is_cpu();
    let copy = || kernel_source("", "f32", "in0[i]", "", "x");

    // One buffer must hold a whole calibration array: stay within what the device can bind.
    let cap = u32::try_from(ctx.max_buffer_bytes() / 4).unwrap_or(u32::MAX);
    let n_copy: u32 = if cpu { 1 << 20 } else { 1 << 24 }.min(cap);
    let t_copy = time_kernel(ctx, copy(), "CostCopy", n_copy, 0, REPS)?;
    let bandwidth_gbs = 8.0 * f64::from(n_copy) / t_copy;

    let n_alu: u32 = if cpu { 1 << 16 } else { 1 << 20 }.min(cap);
    let (k_alu, k_df, k_f64) = (512u32, 64u32, 128u32);
    let alu = kernel_source("", "f32", "in0[i]", "x = x * 1.0001 + 0.5;", "x");
    let t_alu = time_kernel(ctx, alu, "CostAlu", n_alu, k_alu, REPS)?;
    let f32_ops_per_ns = 2.0 * f64::from(k_alu) * f64::from(n_alu) / t_alu;
    let per_op_f32 = t_alu / (2.0 * f64::from(k_alu) * f64::from(n_alu));

    let df = kernel_source(
        &df_library("p.g", ctx.tuning().df_fma, false),
        "vec2<f32>",
        "vec2<f32>(in0[i], 0.0)",
        "x = df_add(df_mul(x, vec2<f32>(1.0001, 0.0)), vec2<f32>(0.5, 0.0));",
        "x.x + x.y",
    );
    let t_df = time_kernel(ctx, df, "CostDf32", n_alu, k_df, REPS)?;
    let df32_factor = t_df / (2.0 * f64::from(k_df) * f64::from(n_alu)) / per_op_f32;

    let f64_factor = if ctx.features().contains(luma_core::Features::SHADER_F64) {
        let wide = kernel_source("", "f64", "f64(in0[i])", "x = x * f64(1.0001) + f64(0.5);", "f32(x)");
        let t_f64 = time_kernel(ctx, wide, "CostF64", n_alu, k_f64, REPS)?;
        Some(t_f64 / (2.0 * f64::from(k_f64) * f64::from(n_alu)) / per_op_f32)
    } else {
        None
    };

    let t_launch = time_kernel(ctx, copy(), "CostLaunch", 1, 0, LAUNCHES)?;
    let memory = match measure_memory(ctx, bandwidth_gbs, t_launch) {
        Ok(memory) => Some(memory),
        Err(err) => {
            log::debug!("luma: the memory hierarchy calibration failed: {err}");
            None
        }
    };
    Ok(CostModel::new(t_launch / 1000.0, bandwidth_gbs, f32_ops_per_ns, df32_factor, f64_factor).with(|m| m.memory = memory))
}

/// How much worse than the profile's numbers a measurement may be before it counts as taken
/// under contention (another process using the device) and is not believed.
const CONTENDED: f64 = 4.0;

/// Whether `measured` is worse than `reference` (the profile's fixed numbers) by more than
/// [`CONTENDED`] in any number: a slower launch, less bandwidth or arithmetic throughput, or a
/// costlier `Df32` or `f64` operation. Better than the reference is never suspicious (the
/// reference is a typical device, not a limit).
pub(crate) fn contended(measured: &CostModel, reference: &CostModel) -> bool {
    let slower = |m: f64, r: f64| m > r * CONTENDED;
    let lower = |m: f64, r: f64| m < r / CONTENDED;
    slower(measured.launch_us, reference.launch_us)
        || lower(measured.bandwidth_gbs, reference.bandwidth_gbs)
        || lower(measured.f32_ops_per_ns, reference.f32_ops_per_ns)
        || slower(measured.df32_factor, reference.df32_factor)
        || matches!((measured.f64_factor, reference.f64_factor), (Some(m), Some(r)) if slower(m, r))
        || matches!((&measured.memory, &reference.memory), (Some(m), Some(r)) if lower(m.l2_gbs, r.l2_gbs) || slower(m.latency_ns, r.latency_ns))
}

/// [`calibrate`], measured again once if the result looks contended (see [`contended`]):
/// `None` if both measurements do, so that the caller neither uses nor keeps them.
pub(crate) fn calibrate_clean(ctx: &Context, reference: &CostModel) -> Result<Option<CostModel>> {
    for _ in 0..2 {
        let model = calibrate(ctx)?;
        if !contended(&model, reference) {
            return Ok(Some(model));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_follow_the_table() {
        let cost = CostModel::TEST.with(|m| { m.df32_factor = 20.0; m.f64_factor = Some(3.0); });
        let mul = |ty| Op::new(OpKind::Binary(BinOp::Mul), ty);
        assert_eq!(op_weight(&mul(ElemTy::F32), &cost), 1.0);
        assert_eq!(op_weight(&mul(ElemTy::Df32), &cost), 20.0);
        assert_eq!(op_weight(&mul(ElemTy::F64), &cost), 3.0);
        assert_eq!(op_weight(&mul(ElemTy::F64), &cost.with(|m| { m.f64_factor = None; })), 32.0);
        assert_eq!(op_weight(&Op::new(OpKind::Unary(UnOp::Sqrt), ElemTy::F32), &cost), 4.0);
        assert_eq!(op_weight(&Op::new(OpKind::Unary(UnOp::Tanh), ElemTy::Df32), &cost), 480.0);
        assert_eq!(op_weight(&Op::new(OpKind::Unary(UnOp::Exp), ElemTy::Df32), &cost), 400.0);
        assert_eq!(op_weight(&Op::new(OpKind::Unary(UnOp::Ln), ElemTy::Df32), &cost), 320.0);
        assert_eq!(op_weight(&Op::new(OpKind::Powf, ElemTy::Df32), &cost), 800.0);
        assert_eq!(op_weight(&Op::new(OpKind::Powf, ElemTy::F32), &cost), 16.0);
        assert_eq!(op_weight(&Op::new(OpKind::Cast(ElemTy::Df32), ElemTy::F32), &cost), 4.0);
        assert_eq!(op_weight(&Op::new(OpKind::Generate(Gen::Normal), ElemTy::F32), &cost), 60.0);
    }

    #[test]
    fn kernel_time_is_launch_plus_the_slower_of_memory_and_arithmetic() {
        let cost = CostModel::TEST.with(|m| { m.launch_us = 5.0; m.bandwidth_gbs = 100.0; m.f32_ops_per_ns = 1000.0; });
        assert_eq!(kernel_ns(&cost, 1e6, 0.0), 5000.0 + 1e4);
        assert_eq!(kernel_ns(&cost, 1e3, 1e8), 5000.0 + 1e5);
    }

    #[test]
    fn only_a_measurement_much_worse_than_the_profile_counts_as_contended() {
        let reference = CostModel::NVIDIA;
        assert!(!contended(&reference, &reference));
        let better = CostModel::new(1.0, 5000.0, 1e5, 5.0, Some(2.0));
        assert!(!contended(&better, &reference), "a faster device is no sign of contention");
        let slightly_worse = reference.with(|m| { m.launch_us = reference.launch_us * 3.9; m.bandwidth_gbs = reference.bandwidth_gbs / 3.9; });
        assert!(!contended(&slightly_worse, &reference));
        let worse = |change: fn(&mut CostModel)| {
            let mut model = reference;
            change(&mut model);
            contended(&model, &reference)
        };
        assert!(worse(|m| m.launch_us *= 4.5), "launches 4.5 times slower");
        assert!(worse(|m| m.bandwidth_gbs /= 4.5), "a quarter of the bandwidth");
        assert!(worse(|m| m.f32_ops_per_ns /= 4.5), "a quarter of the arithmetic");
        assert!(worse(|m| m.df32_factor *= 4.5), "Df32 4.5 times costlier");
        assert!(worse(|m| m.f64_factor = m.f64_factor.map(|f| f * 4.5)), "f64 4.5 times costlier");
        assert!(!worse(|m| m.f64_factor = None), "a device without f64 has nothing to compare");
        // The spread HangHunt saw between processes sharing one GPU.
        let busy = reference.with(|m| { m.launch_us = 57.9; m.bandwidth_gbs = 139.0; m.f32_ops_per_ns = 1282.0; });
        assert!(contended(&busy, &reference));
    }

    #[test]
    fn the_memory_calibration_on_the_cpu_gives_sane_factors() {
        let options = crate::ContextOptions::default().backend(luma_core::BackendKind::Cpu).calibrate(false);
        let ctx = Context::with_options(options).unwrap();
        let cost = calibrate(&ctx).unwrap();
        let memory = cost.memory.expect("the CPU backend measures the memory model");
        assert_eq!(memory.shared_gbs, None);
        let factors = [memory.strided_factor[0], memory.strided_factor[1], memory.strided_factor[2], memory.gather_factor, memory.transpose_factor];
        for factor in factors {
            assert!(factor.is_finite() && factor >= 0.9, "{memory}");
        }
        assert!(memory.l2_gbs.is_finite() && memory.l2_gbs > 0.0 && memory.latency_ns.is_finite() && memory.latency_ns > 0.0, "{memory}");
    }

    #[test]
    fn a_much_slower_l2_or_latency_counts_as_contended_when_both_have_memory_numbers() {
        let memory = MemoryModel::new(Some(6 << 20), true, 2000.0, Some(5000.0), [1.0, 2.0, 4.0], 8.0, 4.0, 30.0);
        let reference = CostModel::NVIDIA.with(|m| m.memory = Some(memory));
        assert!(!contended(&reference, &reference));
        assert!(contended(&reference.with(|m| m.memory = Some(MemoryModel::new(Some(6 << 20), true, 400.0, Some(5000.0), [1.0, 2.0, 4.0], 8.0, 4.0, 30.0))), &reference));
        assert!(contended(&reference.with(|m| m.memory = Some(MemoryModel::new(Some(6 << 20), true, 2000.0, Some(5000.0), [1.0, 2.0, 4.0], 8.0, 4.0, 130.0))), &reference));
        assert!(!contended(&reference.with(|m| m.memory = None), &reference), "nothing to compare");
        assert!(!contended(&reference, &CostModel::NVIDIA), "the presets have no memory numbers");
    }

    #[test]
    fn calibrating_a_context_with_small_chunks_leaves_it_usable() {
        let options = crate::ContextOptions::default().calibrate(false).chunk_bytes(1024);
        let ctx = Context::with_options(options).unwrap();
        let cost = calibrate(&ctx).unwrap();
        for number in [cost.launch_us, cost.bandwidth_gbs, cost.f32_ops_per_ns, cost.df32_factor] {
            assert!(number.is_finite() && number > 0.0, "{cost:?}");
        }
        let xs: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let x = crate::Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
        let y = &x * 2.0 + 1.0;
        let expected: Vec<f32> = xs.iter().map(|v| v * 2.0 + 1.0).collect();
        assert_eq!(y.to_vec().unwrap(), expected);
    }
}
