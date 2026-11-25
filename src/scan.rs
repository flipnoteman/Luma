//! Scans and reductions along one axis. A line is split in blocks that run in parallel: block
//! totals, a scan of the totals (recursively) and a second pass that seeds every block. The
//! results are recorded, not waited for.
//!
//! Memory is only fast when the invocations of a warp touch adjacent addresses at the same time,
//! so the decomposition depends on the layout of the line:
//!
//! - *Contiguous lines* (`stride == 1`): a workgroup owns a chunk of the line. It loads the chunk
//!   cooperatively (coalesced) into workgroup memory, every invocation folds its own run of
//!   [`PER_THREAD`] cells, the partial results are scanned in workgroup memory with a
//!   Hillis-Steele ladder and every invocation rescans its run seeded with its prefix. Where
//!   the device orders memory across workgroups (`Tuning::lookback_scan`), a line of several
//!   chunks is one kernel that reads the input once instead of twice when the result does not
//!   depend on how the operands are grouped (integers, `min`, `max`): the chunks wait for each
//!   other's prefix (decoupled look-back, see [`Pass::ChunkLookback`]).
//! - *Many contiguous lines* (at least [`WARP_MIN_LINES`] on the reference device) on devices
//!   whose subgroups have 32 invocations (`Tuning::subgroup32`): one subgroup per line scans it
//!   in registers, shuffling the totals of its vectors, in a single kernel that reads and writes
//!   every cell once, at the bandwidth of a copy whatever the length of the line (see
//!   [`Pass::Warp`]). The grouping of the operands is fixed by the vector width and the subgroup
//!   width, so `f32` sums are reproducible.
//! - *Strided lines*: invocations are numbered `block * lines + line`, so neighbours read
//!   neighbouring cells of the same row; each walks one block of the line sequentially.

use std::fmt::Write;
use std::panic::Location;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use luma_core::Buffer;

use crate::jit::structured::boundary::{ScanAttrs, ScanOperand};
use crate::jit::structured::emit::Emit;
use crate::jit::structured::strategy::{first_candidate, note, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};

use crate::array::{Array, Failure};
use crate::builtin::{BinOp, Builtin, ReduceElem, ReduceList, ReduceOp, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, Numeric, ScalarType};
use crate::jit::{self, operator_source, ElemTy, Node, Operator};
use crate::kernel::KernelCore;
use crate::ops::{fold_workgroups, launch_1d, Outcome, Rhs};
use crate::reduce::Reduction;
use crate::shape::Shape;

// Measured on an RTX 3090 (`bench_pde --only scan` and `reduce/sum_axis`, 2^26 `f32`): every
// kernel below runs at about 800 GB/s, the bandwidth of the passes (a scan reads the input twice
// and writes it once), so the block shapes are not sensitive: workgroups of 64 to 256 and 4 to 16
// cells per invocation all give `scan/cumsum_*` 0.93-0.95 ms (before: 22-29 ms with one
// invocation per 1024 cells, 1.55 ms for the strided axis). `sum_axis` takes 0.33 ms (was 1.7).

/// Invocations of a workgroup of the contiguous-line kernels.
pub(crate) const WORKGROUP: u32 = 128;
/// Contiguous cells every invocation of a chunk scan folds (a chunk is `WORKGROUP * PER_THREAD`).
const PER_THREAD: u32 = 8;
/// Cells of a contiguous line one workgroup scans.
pub(crate) const SCAN_CHUNK: u32 = WORKGROUP * PER_THREAD;
/// Contiguous cells every invocation of a [`Pass::ChunkLookback`] chunk folds. A look-back walks
/// over the chunks that started while it waited and have no inclusive prefix yet, a window of
/// [`WORKGROUP`] chunks at a time, and takes longer the more of them there are. At the bandwidth
/// of an RTX 3090 chunks of 1024 cells go by at 100 per microsecond, more than a window per
/// microsecond; chunks of 4096 cells (16 KiB of workgroup memory, 18 KiB with the rest of the
/// kernel) go by four times slower, which a window covers. Chosen, not measured. The loads of
/// such a chunk are unrolled so that an invocation has all of them in flight (see
/// [`chunk_source`]).
const LOOKBACK_PER_THREAD: u32 = 32;
/// Cells of a contiguous line one workgroup of [`Pass::ChunkLookback`] scans.
pub(crate) const LOOKBACK_CHUNK: u32 = WORKGROUP * LOOKBACK_PER_THREAD;
/// Cells of a contiguous line one workgroup reduces (its invocations stride over them).
pub(crate) const REDUCE_CHUNK: u32 = 16384;
/// Contiguous lines up to this many cells are handled a group of lines per workgroup. One
/// workgroup per line (128 invocations for 33 cells) was 6 ms for 2^26 cells, a group of lines
/// 0.4 ms; from 256 cells on a workgroup per line is as fast.
pub(crate) const SHORT_LINE: u32 = 255;
/// Cells of workgroup memory a group of short lines is staged in (16 KiB, the smallest limit
/// WebGPU guarantees).
pub(crate) const ROWS_TILE: u32 = 4096;
/// Invocations of a workgroup of [`Pass::Warp`]: one subgroup, one line. Measured on an RTX 3090
/// (`cumsum` of 4096 lines of 4096 `f32`): workgroups of 32 to 512 invocations and 1 to 8 steps in
/// flight all take 0.169-0.175 ms of GPU time (an elementwise copy of the same bytes 0.166 ms),
/// but the call (launch to sync) takes 0.175 ms with one subgroup per workgroup and one step in
/// flight, and 0.18-0.19 ms with more.
const WARP_GROUP: u32 = 32;
/// Steps (of 32 vectors, one per invocation) a [`Pass::Warp`] subgroup loads ahead of the one it
/// scans.
const WARP_AHEAD: u32 = 1;
/// Fewest lines (on the reference device, scaled to the multiprocessors of the device, see
/// `Tuning::scaled`) for which a subgroup per line keeps memory busy and beats the other passes.
/// `cumsum` of 2^24 cells in lines on an RTX 3090 (u32 / f32, ms): 128 lines 0.200 against
/// 0.210 / 0.271 for the other passes, 256 lines 0.184 against 0.204 / 0.263, 64 lines 0.33
/// against 0.214 / 0.255; on an RTX 3080 Laptop (48 multiprocessors) 128 lines already run at
/// the bandwidth of a copy and 64 lines beat the other passes too.
pub(crate) const WARP_MIN_LINES: u32 = 128;
/// Strided lines up to this many cells are walked by one invocation each.
const SHORT_STRIDED: u32 = 256;
/// A strided pass with this many lines or more has enough invocations without blocks.
const WIDE_LINES: u32 = 1 << 16;
/// Invocations per compute unit a blocked strided pass aims for.
const THREADS_PER_UNIT: u64 = 1536;

/// The invocations a blocked strided pass aims for: [`THREADS_PER_UNIT`] per compute unit (about the
/// threads a unit keeps resident). Fewer, longer blocks mean fewer partial results for the second
/// pass to read back: a strided sum, product or mean of 4096 x 4096 over axis 1 takes 0.101 ms on an
/// RTX 3090 with a million invocations, 0.094 ms with 262144 (ArrayFire 0.096), and on an RTX 3080
/// Laptop 0.221 ms with a million, 0.195 ms with 65536 (ArrayFire 0.203).
pub(crate) fn target_threads(tuning: &crate::tuning::Tuning) -> u64 {
    u64::from(tuning.compute_units.max(1)) * THREADS_PER_UNIT
}
/// Fewest cells of a line one invocation of a blocked strided pass walks.
const MIN_BLOCK: u32 = 16;

/// One kernel of the scans and reductions along an axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Pass {
    /// Strided lines, one invocation per line: the inclusive scan of the whole line.
    Seq,
    /// Strided lines, one invocation per block (`block * lines + line`): the fold of the block
    /// into `out[block * lines + line]`; one block per line writes `out[line]`.
    Fold,
    /// Strided lines, same invocations: the scan of the block, seeded with the scan of the
    /// totals of the blocks before it.
    Walk,
    /// Short contiguous lines, one workgroup per `nb` lines: the workgroup loads its lines
    /// cooperatively into workgroup memory and each invocation scans (`fold == false`) or folds
    /// one line from there, so the global accesses stay coalesced.
    Rows { fold: bool, width: u32 },
    /// Contiguous lines, one workgroup per line and chunk: the ordered fold of the chunk into
    /// `out[line * nb + chunk]`.
    ChunkFold,
    /// Contiguous lines, same workgroups: the scan of the chunk, seeded (when the line has
    /// several chunks) with the scan of the totals of the chunks before it.
    ChunkScan { seeded: bool },
    /// Contiguous lines of more than [`LOOKBACK_CHUNK`] cells, one workgroup per chunk of that
    /// size, the scan of the whole line in one kernel that reads the input once (decoupled
    /// look-back, Merrill & Garland 2016). A workgroup takes its chunk from an atomic counter, so
    /// the chunks it waits for have all started (forward progress needs no particular scheduling
    /// order); it scans its chunk in workgroup memory, publishes the chunk's aggregate, looks
    /// back over the chunks before it until one has published its inclusive prefix, publishes
    /// its own and writes the scan of the chunk seeded with the prefix before it. The status of
    /// the chunks lives in the buffer that [`chunk_source`] describes, which must be zero when
    /// the kernel starts; `k` is the number of chunks it reads the status of at a time. Only for
    /// scans whose result does not depend on the grouping of the operands ([`Lines::grouping_free`]):
    /// the prefix of a chunk is the fold of the chunks the look-back found, not of a fixed set.
    ChunkLookback,
    /// Contiguous lines, same workgroups: the fold of the chunk with the invocations striding
    /// over it, so the operator must be commutative as well as associative.
    ChunkReduce,
    /// Contiguous lines of more than [`SHORT_LINE`] cells, a workgroup of one subgroup of 32
    /// invocations per line, the whole scan in one kernel that reads the
    /// input once and writes the result once, in registers, with no workgroup memory or barrier.
    /// The subgroup walks the line 32 vectors of `vec` cells at a time, one vector per invocation
    /// (so the global loads and stores are coalesced and `vec * 4` bytes wide): every invocation
    /// scans its vector, the subgroup scans the totals of the vectors with shuffles (a
    /// Hillis-Steele ladder) and the carry of the line so far comes from the last cell written
    /// by the step before. `vec` divides the line. The grouping of the operands (and so the
    /// rounding of `f32` sums) is fixed by `vec` and the width of the subgroup: the cells of a
    /// vector, then the vectors of a step, then the steps, left to right. Needs
    /// `Tuning::subgroup32` and enough lines (see [`WARP_MIN_LINES`]).
    Warp { vec: u32 },
    /// Segmented reduction over entries cut in tiles by offsets (see [`crate::segments`]): one workgroup per tile.
    Segments,
}

impl Pass {
    fn label(self) -> &'static str {
        match self {
            Pass::Seq => "ScanSeq",
            Pass::Rows { fold: false, .. } => "ScanRows",
            Pass::Rows { fold: true, .. } => "ReduceRows",
            Pass::Fold => "ScanFold",
            Pass::Walk => "ScanWalk",
            Pass::ChunkFold => "ScanChunkFold",
            Pass::ChunkScan { .. } => "ScanChunk",
            Pass::ChunkLookback => "ScanLookback",
            Pass::ChunkReduce => "ReduceChunk",
            Pass::Warp { .. } => "ScanWarp",
            Pass::Segments => "SegmentsTile",
        }
    }

    /// Whether the pass runs one workgroup (instead of one invocation) per unit of work.
    fn per_workgroup(self) -> bool {
        matches!(self, Pass::Rows { .. } | Pass::Warp { .. } | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkLookback | Pass::ChunkReduce | Pass::Segments)
    }

    pub(crate) fn seeded(self) -> bool {
        matches!(self, Pass::Walk | Pass::ChunkScan { seeded: true })
    }
}

/// A reducing [`Pass`]: the kernels of `reduce_axis`, which the buffer path of this module and the
/// fused kernels of `jit` (`Tail::AxisReduce`) both generate with [`reduce_body`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct AxisPass {
    /// `Fold`, `Rows { fold: true, .. }` or `ChunkReduce`.
    pub(crate) pass: Pass,
    /// The reductions, computed together from one read of every cell.
    pub(crate) ops: ReduceList,
    /// The cells are the partial results of an earlier pass: op `j` reads its own `p.n * p.lines`
    /// cells, which start at `j * p.n * p.lines`. Otherwise every op reads the same cells.
    pub(crate) partial: bool,
    /// Op `j` writes its own buffer `out{j}` (the last pass of several ops). Otherwise it writes
    /// `out[j * count + i]` of the one buffer `out`, `count` being the results per op.
    pub(crate) split: bool,
    /// A `Fold` that leaves several partial results per line writes them line-major, `out[l * nb + b]`
    /// instead of `out[b * lines + l]`, so that the next pass is a contiguous reduction of
    /// lines of `nb` cells (see [`line_major`]).
    pub(crate) line_major: bool,
    /// A scan, not a reduction: `pass` is one of the scan passes of [`Lines::scan`], reading its operand through `load` (the fused
    /// first read of a `cumsum`, see [`scan_body`]); `ops` is then the one sum that stands for the operator.
    pub(crate) scan: bool,
}

/// Whether the `Fold` of `nb` blocks per line writes line-major partial results: when one
/// workgroup of the contiguous pass reduces a whole line of them, that pass is the last.
fn line_major(pass: Pass, nb: u32) -> bool {
    pass == Pass::Fold && nb > 1 && nb <= REDUCE_CHUNK
}

/// The fields of the `Params` uniform of every reducing pass: the geometry of [`Lines`].
pub(crate) const AXIS_PARAMS: &str = "n: u32, stride: u32, lines: u32, nb: u32, k: u32, ";

impl AxisPass {
    /// The profile label of the kernel.
    pub(crate) fn label(self) -> &'static str {
        if self.scan {
            return self.pass.label();
        }
        let arg = self.ops.as_slice().first().is_some_and(|op| op.is_arg());
        match (self.pass, arg) {
            (Pass::Fold, false) => "ReduceFold",
            (Pass::Fold, true) => "ArgFold",
            (Pass::Rows { .. }, true) => "ArgRows",
            (Pass::ChunkReduce, true) => "ArgChunk",
            (pass, _) => pass.label(),
        }
    }

    /// How many buffers the pass writes (`@binding(0..)`).
    pub(crate) fn outputs(self) -> usize {
        if self.pass == Pass::Segments { 2 } else if self.split { self.ops.as_slice().len() } else { 1 }
    }

    /// The declarations of the output buffers, of elements of `ty` (`u32`, the indices and the
    /// bits of the values, for an arg reduction).
    pub(crate) fn output_bindings(self, ty: &str) -> String {
        if self.pass == Pass::Segments {
            return crate::segments::tile_outputs(ty);
        }
        if let (true, Pass::Warp { vec }) = (self.scan, self.pass)
            && vec > 1
        {
            return format!("@group(1) @binding(0) var<storage, read_write> out: array<vec{vec}<{ty}>>;\n");
        }
        let ty = if self.ops.as_slice().first().is_some_and(|op| op.is_arg()) { "u32" } else { ty };
        let mut src = String::new();
        for j in 0..self.outputs() {
            let name = if self.split { format!("out{j}") } else { "out".to_owned() };
            let _ = writeln!(src, "@group(1) @binding({j}) var<storage, read_write> {name}: array<{ty}>;");
        }
        src
    }
}

/// The kernel of `pass` over a buffer `a`.
fn reduce_source(pass: AxisPass, ty: &str, workgroup: u32) -> String {
    if pass.ops.as_slice().first().is_some_and(|op| op.is_arg()) {
        return crate::arg::source(pass, ty, workgroup);
    }
    let outputs = pass.outputs();
    let mut src = format!("struct Params {{ {AXIS_PARAMS}}}\n{}", pass.output_bindings(ty));
    let _ = writeln!(src, "@group(1) @binding({outputs}) var<storage, read> a: array<{ty}>;");
    let _ = writeln!(src, "@group(1) @binding({}) var<uniform> p: Params;", outputs + 1);
    let _ = writeln!(src, "fn ld(q: u32) -> {ty} {{ return a[q]; }}");
    src.push_str(&reduce_body(pass, ty, "ld", workgroup));
    src
}

/// `line_base`, the combine functions `c{j}` and the entry point of the reducing pass, with the
/// cells of the operand from `{load}(q)`. Needs `Params` (see [`AXIS_PARAMS`]) as `p` and the
/// output buffers of [`AxisPass::output_bindings`]. The combine of an op is that of the whole
/// array reductions (`ReduceElem::combine`), so a NaN is handled as `Array::min` and `max` do;
/// `SumSquares` squares the cells it reads.
pub(crate) fn reduce_body(pass: AxisPass, ty: &str, load: &str, workgroup: u32) -> String {
    if pass.scan {
        return fused_scan_body(pass.pass, ty, load, workgroup);
    }
    if pass.pass == Pass::Segments {
        return crate::segments::tile_body(ty, pass.ops.as_slice()[0], load);
    }
    if pass.ops.as_slice().first().is_some_and(|op| op.is_arg()) {
        return crate::arg::body(pass, ty, load, workgroup);
    }
    let elem = ReduceElem::Plain(ty);
    let ops = pass.ops.as_slice();
    let mut src = String::from("fn line_base(l: u32) -> u32 { return (l % p.stride) + (l / p.stride) * p.stride * p.n; }\n");
    for (j, &op) in ops.iter().enumerate() {
        let _ = writeln!(src, "fn c{j}(u: {ty}, v: {ty}) -> {ty} {{ return {}; }}", elem.combine(op));
    }
    let cin = if pass.partial { "    let cin = p.n * p.lines;\n" } else { "" };
    // The statement binding the cell at `q`: `{name}` for all ops, or `{name}{j}` for op `j`.
    let read = |name: &str, q: &str| -> String {
        if pass.partial {
            (0..ops.len()).map(|j| format!("let {name}{j} = {load}({j}u * cin + {q});")).collect::<Vec<_>>().join(" ")
        } else {
            format!("let {name} = {load}({q});")
        }
    };
    // A first pass of ops that reduce different roots of the program reads one array of the roots' values per cell.
    let roots = if pass.partial { None } else { pass.ops.roots() };
    let cell = |name: &str, j: usize| match &roots {
        _ if pass.partial => format!("{name}{j}"),
        Some(roots) => format!("{name}[{}u]", roots[j]),
        None => name.to_owned(),
    };
    let root_of = |j: usize| roots.as_ref().map_or(0, |roots| usize::from(roots[j]));
    let tiles = roots.as_ref().map_or(1, |_| root_count(&pass.ops) as usize);
    let mapped = |j: usize, cell: String| if ops[j] == ReduceOp::SumSquares { elem.square(&cell) } else { cell };
    let store = |j: usize, at: &str, count: &str, value: String| {
        if pass.split { format!("out{j}[{at}] = {value};") } else { format!("out[{j}u * {count} + {at}] = {value};") }
    };
    let w = WORKGROUP;
    match pass.pass {
        Pass::Fold => {
            let _ = write!(
                src,
                "@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {workgroup}u;
    if (i >= p.lines * p.nb) {{ return; }}
    let l = i % p.lines;
    let b = i / p.lines;
    let base = line_base(l);
    let s0 = b * p.k;
    let s1 = min(p.n, s0 + p.k);
{cin}    {}
",
                read("f", "base + s0 * p.stride")
            );
            for j in 0..ops.len() {
                let _ = writeln!(src, "    var acc{j} = {};", mapped(j, cell("f", j)));
            }
            let _ = writeln!(src, "    for (var s = s0 + 1u; s < s1; s++) {{\n        {}", read("e", "base + s * p.stride"));
            for j in 0..ops.len() {
                let _ = writeln!(src, "        acc{j} = c{j}(acc{j}, {});", mapped(j, cell("e", j)));
            }
            src.push_str("    }\n");
            for j in 0..ops.len() {
                let at = if pass.line_major { "l * p.nb + b" } else { "i" };
                let _ = writeln!(src, "    {}", store(j, at, "p.lines * p.nb", format!("acc{j}")));
            }
            src.push_str("}\n");
        }
        Pass::Rows { width, .. } => {
            for r in 0..tiles {
                let name = if roots.is_some() { format!("tile{r}") } else { "tile".to_owned() };
                let _ = writeln!(src, "var<workgroup> {name}: array<{ty}, {}>;", ROWS_TILE as usize / tiles);
            }
            let _ = write!(
                src,
                "@compute @workgroup_size({width})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let first = (wg.x + wg.y * nwg.x) * p.nb;
    if (first >= p.lines) {{ return; }}
    let cnt = min(p.nb, p.lines - first);
    let pitch = p.n | 1u;
    let base = first * p.n;
    let cells = cnt * p.n;
{cin}"
            );
            for j in 0..ops.len() {
                let _ = writeln!(src, "    var acc{j}: {ty};");
            }
            // Staged cells are shared by all ops, but the partial results of every op are their
            // own cells: those take a round each through the one tile.
            let rounds: Vec<Vec<usize>> = if pass.partial { (0..ops.len()).map(|j| vec![j]).collect() } else { vec![(0..ops.len()).collect()] };
            for (r, round) in rounds.iter().enumerate() {
                let offset = if pass.partial { format!("{r}u * cin + ") } else { String::new() };
                let _ = write!(
                    src,
                    "    for (var j = t; j < cells; j += {width}u) {{
        let row = j / p.n;
"
                );
                if roots.is_some() {
                    let _ = writeln!(src, "        let v = {load}({offset}base + j);");
                    for k in 0..tiles {
                        let _ = writeln!(src, "        tile{k}[row * pitch + (j - row * p.n)] = v[{k}u];");
                    }
                } else {
                    let _ = writeln!(src, "        tile[row * pitch + (j - row * p.n)] = {load}({offset}base + j);");
                }
                let _ = write!(
                    src,
                    "    }}
    workgroupBarrier();
    if (t < cnt) {{
        let row = t * pitch;
"
                );
                for &j in round {
                    let tile = if roots.is_some() { format!("tile{}", root_of(j)) } else { "tile".to_owned() };
                    let _ = writeln!(src, "        acc{j} = {};", mapped(j, format!("{tile}[row]")));
                }
                src.push_str("        for (var s = 1u; s < p.n; s++) {\n");
                for &j in round {
                    let tile = if roots.is_some() { format!("tile{}", root_of(j)) } else { "tile".to_owned() };
                    let _ = writeln!(src, "            acc{j} = c{j}(acc{j}, {});", mapped(j, format!("{tile}[row + s]")));
                }
                src.push_str("        }\n    }\n");
                if r + 1 < rounds.len() {
                    src.push_str("    workgroupBarrier();\n");
                }
            }
            src.push_str("    if (t < cnt) {\n");
            for j in 0..ops.len() {
                let _ = writeln!(src, "        {}", store(j, "first + t", "p.lines", format!("acc{j}")));
            }
            src.push_str("    }\n}\n");
        }
        Pass::ChunkReduce => {
            for j in 0..ops.len() {
                let _ = writeln!(src, "var<workgroup> part{j}: array<{ty}, {w}>;");
            }
            let _ = write!(
                src,
                "@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wid = wg.x + wg.y * nwg.x;
    if (wid >= p.lines * p.nb) {{ return; }}
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = line_base(l);
{cin}    let s0 = c * {REDUCE_CHUNK}u;
    let s1 = min(p.n, s0 + {REDUCE_CHUNK}u);
"
            );
            for j in 0..ops.len() {
                let _ = writeln!(src, "    var acc{j}: {ty};");
            }
            let _ = writeln!(src, "    var s = s0 + t;\n    if (s < s1) {{\n        {}", read("f", "base + s"));
            for j in 0..ops.len() {
                let _ = writeln!(src, "        acc{j} = {};", mapped(j, cell("f", j)));
            }
            let _ = writeln!(src, "        s += {w}u;\n        for (; s < s1; s += {w}u) {{\n            {}", read("e", "base + s"));
            for j in 0..ops.len() {
                let _ = writeln!(src, "            acc{j} = c{j}(acc{j}, {});", mapped(j, cell("e", j)));
            }
            src.push_str("        }\n    }\n");
            for j in 0..ops.len() {
                let _ = writeln!(src, "    part{j}[t] = acc{j};");
            }
            let _ = writeln!(src, "    let valid = min({w}u, s1 - s0);\n    for (var d = {w}u / 2u; d > 0u; d >>= 1u) {{\n        workgroupBarrier();\n        if (t < d && t + d < valid) {{");
            for j in 0..ops.len() {
                let _ = writeln!(src, "            part{j}[t] = c{j}(part{j}[t], part{j}[t + d]);");
            }
            src.push_str("        }\n    }\n    if (t == 0u) {\n");
            for j in 0..ops.len() {
                let _ = writeln!(src, "        {}", store(j, "wid", "p.lines * p.nb", format!("part{j}[0]")));
            }
            src.push_str("    }\n}\n");
        }
        Pass::Seq | Pass::Walk | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkLookback | Pass::Warp { .. } | Pass::Segments => unreachable!("scans are not reductions"),
    }
    src
}

/// The kernel of a scan `pass` of `+` whose operand is computed in it: `load(q)` is cell `q` (see [`reduce_body`], whose fused
/// kernels this is one of). A warp pass of vectors loads `vec` consecutive cells.
fn fused_scan_body(pass: Pass, ty: &str, load: &str, workgroup: u32) -> String {
    let mut src = String::new();
    let mut load = load.to_owned();
    if let Pass::Warp { vec } = pass
        && vec > 1
    {
        let cells: Vec<String> = (0..vec).map(|c| format!("{load}(q * {vec}u + {c}u)")).collect();
        let _ = writeln!(src, "fn luma_scan_vec(q: u32) -> vec{vec}<{ty}> {{ return vec{vec}<{ty}>({}); }}", cells.join(", "));
        load = "luma_scan_vec".to_owned();
    }
    src.push_str(&scan_body(pass, ty, &format!("fn op(x0: {ty}, x1: {ty}) -> {ty} {{ return x0 + x1; }}\n"), &load, workgroup));
    src
}

fn source(pass: Pass, ty: &str, op: &Operator, workgroup: u32) -> String {
    let mut src = String::from("struct Params { n: u32, stride: u32, lines: u32, nb: u32, k: u32, ");
    for (k, scalar) in op.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", scalar.wgsl());
    }
    src.push_str("}\n");
    let elem = match pass {
        Pass::Warp { vec } if vec > 1 => format!("vec{vec}<{ty}>"),
        _ => ty.to_owned(),
    };
    let _ = writeln!(src, "@group(1) @binding(0) var<storage, read_write> out: array<{elem}>;");
    let _ = writeln!(src, "@group(1) @binding(1) var<storage, read> a: array<{elem}>;");
    let uniform = if pass.seeded() {
        let _ = writeln!(src, "@group(1) @binding(2) var<storage, read> pre: array<{ty}>;");
        3
    } else if pass == Pass::ChunkLookback {
        let _ = writeln!(src, "@group(1) @binding(2) var<storage, read_write> st: array<atomic<u32>>;");
        3
    } else {
        2
    };
    let _ = writeln!(src, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    let _ = writeln!(src, "fn lda(q: u32) -> {elem} {{ return a[q]; }}");
    src.push_str(&scan_body(pass, ty, &op.source, "lda", workgroup));
    src
}

/// The operator `op`, `line_base` and the entry point of a scan `pass`, with the cells of the operand from `{load}(q)` (a vector of
/// `vec` cells for [`Pass::Warp`]). Needs `Params` (see [`AXIS_PARAMS`]) as `p`, the output `out` (an array of vectors for a warp
/// pass of `vec > 1`), and `pre` (seeded passes) or `st` (the look-back) as the kernel's reads. The buffer kernels
/// of [`Lines`] and the fused ones of `jit` (a pending operand of a `cumsum`, `Tail::AxisReduce` with `scan` set) both generate with it.
pub(crate) fn scan_body(pass: Pass, ty: &str, op_source: &str, load: &str, workgroup: u32) -> String {
    let mut src = String::from(op_source);
    src.push_str("fn line_base(l: u32) -> u32 { return (l % p.stride) + (l / p.stride) * p.stride * p.n; }\n");
    if let Pass::Warp { vec } = pass {
        src.push_str(&warp_source(vec, ty, load));
        return src;
    }
    if pass.per_workgroup() {
        src.push_str(&chunk_source(pass, ty, load));
        return src;
    }
    let _ = writeln!(
        src,
        "@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {workgroup}u;"
    );
    let main = match pass {
        Pass::Seq => {
            "    if (i >= p.lines) { return; }
    let base = line_base(i);
    var acc = __LOAD__(base);
    out[base] = acc;
    for (var s = 1u; s < p.n; s++) {
        let q = base + s * p.stride;
        acc = op(acc, __LOAD__(q));
        out[q] = acc;
    }
}
"
        }
        Pass::Fold => {
            "    if (i >= p.lines * p.nb) { return; }
    let l = i % p.lines;
    let b = i / p.lines;
    let base = line_base(l);
    let s0 = b * p.k;
    let s1 = min(p.n, s0 + p.k);
    var acc = __LOAD__(base + s0 * p.stride);
    for (var s = s0 + 1u; s < s1; s++) { acc = op(acc, __LOAD__(base + s * p.stride)); }
    out[i] = acc;
}
"
        }
        Pass::Walk => {
            "    if (i >= p.lines * p.nb) { return; }
    let l = i % p.lines;
    let b = i / p.lines;
    let base = line_base(l);
    let s0 = b * p.k;
    let s1 = min(p.n, s0 + p.k);
    var acc: __T__;
    var have = false;
    if (b > 0u) { acc = pre[i - p.lines]; have = true; }
    for (var s = s0; s < s1; s++) {
        let q = base + s * p.stride;
        let v = __LOAD__(q);
        if (have) { acc = op(acc, v); } else { acc = v; have = true; }
        out[q] = acc;
    }
}
"
        }
        Pass::Rows { .. } | Pass::Warp { .. } | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkLookback | Pass::ChunkReduce | Pass::Segments => unreachable!("workgroup passes return above"),
    };
    src.push_str(&main.replace("__T__", ty).replace("__LOAD__", load));
    src
}

/// The entry point of [`Pass::Warp`]: `vec` cells per invocation (`a` and `out` are arrays of
/// vectors of that many cells, or of cells when it is 1), steps of 32 vectors. Every step the
/// subgroup has loaded [`WARP_AHEAD`] steps ahead is scanned within the invocation, then across
/// the invocations with a ladder of shuffles (every invocation takes part in every shuffle: the
/// control flow is uniform over the subgroup), then seeded with the last cell of the step before
/// (`carry`, which the first step of a line does not have: its first invocation has no prefix).
/// Lanes past the end of the line load zeros and write nothing: they only sit above the lanes
/// that matter, which no lane reads.
fn warp_source(vec: u32, ty: &str, load: &str) -> String {
    let vt = if vec > 1 { format!("vec{vec}<{ty}>") } else { ty.to_owned() };
    let comps: Vec<String> = if vec > 1 { ["x", "y", "z", "w"][..vec as usize].iter().map(|c| format!("v.{c}")).collect() } else { vec!["v".to_owned()] };
    let last = vec as usize - 1;
    let mut src = String::new();
    let _ = writeln!(src, "fn ld(vbase: u32, nv: u32, lane: u32, s: u32) -> {vt} {{ let q = s * 32u + lane; if (q < nv) {{ return {load}(vbase + q); }} return {vt}(); }}");
    // One step: `step` is the expression of its number, `cur` the vector the invocation loaded.
    let process = |step: &str, cur: &str| -> String {
        let mut s = String::new();
        let _ = writeln!(s, "        if ({step} < steps) {{");
        let _ = writeln!(s, "            let v = {cur};");
        for (k, comp) in comps.iter().enumerate() {
            if k == 0 {
                let _ = writeln!(s, "            let x0 = {comp};");
            } else {
                let _ = writeln!(s, "            let x{k} = op(x{}, {comp});", k - 1);
            }
        }
        let _ = writeln!(s, "            var incl = x{last};");
        for off in [1u32, 2, 4, 8, 16] {
            let _ = writeln!(s, "            {{ let o = subgroupShuffle(incl, select(lane, lane - {off}u, lane >= {off}u)); if (lane >= {off}u) {{ incl = op(o, incl); }} }}");
        }
        let _ = writeln!(s, "            let ex = subgroupShuffle(incl, select(0u, lane - 1u, lane > 0u));");
        let _ = writeln!(s, "            var pre = ex;");
        let _ = writeln!(s, "            if ({step} > 0u) {{ if (lane > 0u) {{ pre = op(carry, ex); }} else {{ pre = carry; }} }}");
        let _ = writeln!(s, "            let has = lane > 0u || {step} > 0u;");
        for k in 0..comps.len() {
            let _ = writeln!(s, "            let y{k} = select(x{k}, op(pre, x{k}), has);");
        }
        let _ = writeln!(s, "            carry = subgroupShuffle(y{last}, 31u);");
        let all: Vec<String> = (0..comps.len()).map(|k| format!("y{k}")).collect();
        let value = if vec > 1 { format!("{vt}({})", all.join(", ")) } else { "y0".to_owned() };
        let _ = writeln!(s, "            let q = ({step}) * 32u + lane;\n            if (q < nv) {{ out[vbase + q] = {value}; }}");
        s.push_str("        }\n");
        s
    };
    let u = WARP_AHEAD;
    let _ = write!(
        src,
        "@compute @workgroup_size({WARP_GROUP})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    let line = wg.x + wg.y * nwg.x;
    if (line >= p.lines) {{ return; }}
    let nv = p.n / {vec}u;
    let vbase = line * nv;
    let steps = (nv + 31u) / 32u;
    var carry: {ty};
"
    );
    for j in 0..u {
        let _ = writeln!(src, "    var c{j} = ld(vbase, nv, lane, {j}u);");
    }
    src.push_str("    for (var g = 0u; g < steps; g += ");
    let _ = writeln!(src, "{u}u) {{");
    for j in 0..u {
        let _ = writeln!(src, "        let n{j} = ld(vbase, nv, lane, g + {}u);", u + j);
    }
    for j in 0..u {
        src.push_str(&process(&format!("(g + {j}u)"), &format!("c{j}")));
    }
    for j in 0..u {
        let _ = writeln!(src, "        c{j} = n{j};");
    }
    src.push_str("    }\n}\n");
    src
}

/// The entry point of the passes that run a workgroup per chunk of a contiguous line. `acc` and
/// `run` of invocations past the end of a short chunk are never read by invocations before them.
///
/// [`Pass::ChunkLookback`] binds `st: array<atomic<u32>>` of `1 + 3 * lines * nb` words, whose
/// first `1 + lines * nb` are zero when the kernel starts (see [`Lines::lookback`]): word 0 counts
/// the chunks taken so far (the ticket of a chunk is `line * nb + chunk`, so the chunks before it
/// in its line have smaller tickets), word `1 + ticket` is the status of the chunk (0: nothing
/// yet, 1: its aggregate is published, 2: its inclusive prefix is), and the aggregate and the
/// inclusive prefix themselves (the bits of the element) are words `1 + lines * nb + ticket` and
/// `1 + 2 * lines * nb + ticket`. A value is stored, fenced (`storageBarrier`) and only then the
/// status; a reader loads the status, fences and only then the value.
fn chunk_source(pass: Pass, ty: &str, load: &str) -> String {
    let w = WORKGROUP;
    let mut src = String::new();
    let head = "@compute @workgroup_size(__W__)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let wid = wg.x + wg.y * nwg.x;
    if (wid >= p.lines * p.nb) { return; }
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = line_base(l);
";
    if let Pass::Rows { width, .. } = pass {
        let (tile, r) = (ROWS_TILE, width);
        let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {tile}>;");
        src.push_str(&format!(
            "@compute @workgroup_size({r})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let first = (wg.x + wg.y * nwg.x) * p.nb;
    if (first >= p.lines) {{ return; }}
    let cnt = min(p.nb, p.lines - first);
    let ld = p.n | 1u;
    let base = first * p.n;
    let cells = cnt * p.n;
    for (var j = t; j < cells; j += {r}u) {{
        let r = j / p.n;
        tile[r * ld + (j - r * p.n)] = {load}(base + j);
    }}
    workgroupBarrier();
    var acc: {ty};
    if (t < cnt) {{
        let row = t * ld;
        acc = tile[row];
        for (var s = 1u; s < p.n; s++) {{
            acc = op(acc, tile[row + s]);
            tile[row + s] = acc;
        }}
    }}
    workgroupBarrier();
    for (var j = t; j < cells; j += {r}u) {{
        let r = j / p.n;
        out[base + j] = tile[r * ld + (j - r * p.n)];
    }}
}}
"
        ));
        return src;
    }
    let lookback = pass == Pass::ChunkLookback;
    let (k, chunk) = if lookback { (LOOKBACK_PER_THREAD, LOOKBACK_CHUNK) } else { (PER_THREAD, SCAN_CHUNK) };
    let tile = chunk + chunk / 32 + 1;
    let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {tile}>;");
    let _ = writeln!(src, "var<workgroup> part: array<{ty}, {w}>;");
    src.push_str("fn ix(j: u32) -> u32 { return j + (j >> 5u); }\n");
    if lookback {
        src.push_str(&lookback_declarations(ty));
        src.push_str(
            "@compute @workgroup_size(__W__)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total = p.lines * p.nb;
    if (wg.x + wg.y * nwg.x >= total) { return; }
    if (t == 0u) { ticket = atomicAdd(&st[0], 1u); }
    let wid = workgroupUniformLoad(&ticket);
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = line_base(l);
",
        );
    } else {
        src.push_str(head);
    }
    let _ = write!(src, "    let s0 = c * {chunk}u;\n    let len = min(p.n - s0, {chunk}u);\n");
    if lookback {
        // The loads of an invocation do not depend on one another: all of them are issued before
        // the first is waited for.
        for i in 0..k {
            let _ = writeln!(src, "    {{ let j = {}u + t; if (j < len) {{ tile[ix(j)] = {load}(base + s0 + j); }} }}", i * w);
        }
    } else {
        let _ = writeln!(src, "    for (var j = t; j < len; j += {w}u) {{ tile[ix(j)] = {load}(base + s0 + j); }}");
    }
    src.push_str(&format!(
        "    workgroupBarrier();
    let lo = t * {k}u;
    let hi = min(len, lo + {k}u);
    var acc: {ty};
    if (lo < hi) {{
        acc = tile[ix(lo)];
        for (var j = lo + 1u; j < hi; j++) {{ acc = op(acc, tile[ix(j)]); }}
    }}
    part[t] = acc;
"
    ));
    if pass == Pass::ChunkFold {
        src.push_str(&format!(
            "    let valid = (len + {k}u - 1u) / {k}u;
    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        if (t % (2u * d) == 0u && t + d < valid) {{ part[t] = op(part[t], part[t + d]); }}
    }}
    if (t == 0u) {{ out[wid] = part[0]; }}
}}
"
        ));
    } else {
        src.push_str(&format!(
            "    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        var v = part[t];
        if (t >= d) {{ v = op(part[t - d], v); }}
        workgroupBarrier();
        part[t] = v;
    }}
    workgroupBarrier();
"
        ));
        if lookback {
            src.push_str(&lookback_block(ty));
        }
        src.push_str(&format!(
            "    var run: {ty};
    var have = false;
    if (t > 0u) {{ run = part[t - 1u]; have = true; }}
"
        ));
        if pass.seeded() {
            src.push_str(
                "    if (c > 0u) {
        let seed = pre[wid - 1u];
        if (have) { run = op(seed, run); } else { run = seed; }
        have = true;
    }
",
            );
        }
        if lookback {
            src.push_str(
                "    if (c > 0u) {
        if (have) { run = op(excl, run); } else { run = excl; }
        have = true;
    }
",
            );
        }
        src.push_str(&format!(
            "    for (var j = lo; j < hi; j++) {{
        let v = tile[ix(j)];
        if (have) {{ run = op(run, v); }} else {{ run = v; have = true; }}
        tile[ix(j)] = run;
    }}
    workgroupBarrier();
    for (var j = t; j < len; j += {w}u) {{ out[base + s0 + j] = tile[ix(j)]; }}
}}
"
        ));
    }
    src.replace("__W__", &w.to_string())
}

/// The workgroup variables and helpers of the look-back kernel: the status words hold the bits of
/// an element, whatever its type.
fn lookback_declarations(ty: &str) -> String {
    let (to_bits, from_bits) = if ty == "u32" { ("v".to_owned(), "u".to_owned()) } else { ("bitcast<u32>(v)".to_owned(), format!("bitcast<{ty}>(u)")) };
    let w = WORKGROUP;
    format!(
        "var<workgroup> lb_v: array<{ty}, {w}>;
var<workgroup> lb_s: array<u32, {w}>;
var<workgroup> ticket: u32;
fn lb_bits(v: {ty}) -> u32 {{ return {to_bits}; }}
fn lb_value(u: u32) -> {ty} {{ return {from_bits}; }}
"
    )
}

/// What a chunk of [`Pass::ChunkLookback`] does between the scan of its chunk in workgroup memory
/// (`part[t]` is the inclusive scan of the runs of invocations up to `t`) and writing the result:
/// publish the aggregate of the chunk, find the exclusive prefix `excl` of the chunk, publish the
/// inclusive prefix.
///
/// The first chunk of a line publishes its aggregate as its inclusive prefix and has no `excl`.
/// Any other chunk looks back over a window of `p.k` chunks (at most `WORKGROUP`) before it at a
/// time, an invocation per chunk, nearest chunk last. Every invocation makes a leaf of the status it read
/// (`lb_s`: bit 0 an inclusive prefix, bit 1 not ready, bit 2 no chunk) and the value (the
/// inclusive prefix or the aggregate), and the workgroup folds the leaves in order, oldest first:
/// a range that holds an inclusive prefix is the part of the window from its nearest one (what
/// came before is in that prefix), a range with a chunk that is not ready (nearer than the
/// nearest inclusive prefix) cannot be used yet, and the window is read again. Without an
/// inclusive prefix in the window, the fold of the aggregates is kept (`carry`) and the window
/// moves back. The first chunk of the line always has an inclusive prefix, so the walk ends.
/// Every decision is uniform (read with `workgroupUniformLoad`), so the barriers are too.
fn lookback_block(ty: &str) -> String {
    const BLOCK: &str = "    let agg = part[(len + __K__u - 1u) / __K__u - 1u];
    var excl: __T__;
    if (c == 0u) {
        if (t == 0u) { atomicStore(&st[2u * total + 1u + wid], lb_bits(agg)); }
        storageBarrier();
        if (t == 0u) { atomicStore(&st[1u + wid], 2u); }
    } else {
        if (t == 0u) { atomicStore(&st[total + 1u + wid], lb_bits(agg)); }
        storageBarrier();
        if (t == 0u) { atomicStore(&st[1u + wid], 1u); }
        var back = 0u;
        var carry: __T__;
        loop {
            let off = __W__u - 1u - t;
            let dist = back + off;
            var f = 3u;
            if (off < p.k && dist < c) { f = atomicLoad(&st[wid - dist]); }
            storageBarrier();
            var s = 4u;
            var v: __T__;
            if (f == 0u) { s = 2u; }
            if (f == 1u) { s = 0u; v = lb_value(atomicLoad(&st[total + wid - dist])); }
            if (f == 2u) { s = 1u; v = lb_value(atomicLoad(&st[2u * total + wid - dist])); }
            lb_s[t] = s;
            lb_v[t] = v;
            for (var d = 1u; d < __W__u; d <<= 1u) {
                workgroupBarrier();
                if (t % (2u * d) == 0u && t + d < __W__u) {
                    let o = lb_s[t];
                    let n = lb_s[t + d];
                    if ((n & 4u) == 0u) {
                        if ((o & 4u) != 0u || (n & 1u) != 0u) {
                            lb_s[t] = n;
                            lb_v[t] = lb_v[t + d];
                        } else {
                            lb_s[t] = o | (n & 2u);
                            lb_v[t] = op(lb_v[t], lb_v[t + d]);
                        }
                    }
                }
            }
            let state = workgroupUniformLoad(&lb_s[0]);
            if ((state & 2u) == 0u) {
                var got = lb_v[0];
                if (back > 0u) { got = op(got, carry); }
                if ((state & 1u) != 0u) { excl = got; break; }
                carry = got;
                back += p.k;
            }
        }
        let incl = op(excl, agg);
        if (t == 0u) { atomicStore(&st[2u * total + 1u + wid], lb_bits(incl)); }
        storageBarrier();
        if (t == 0u) { atomicStore(&st[1u + wid], 2u); }
    }
";
    BLOCK.replace("__K__", &LOOKBACK_PER_THREAD.to_string()).replace("__W__", &WORKGROUP.to_string()).replace("__T__", ty)
}

/// The kernels of one operator and what they run on.
pub(crate) struct Lines<'a> {
    ctx: &'a Context,
    op: &'a Operator,
    scratch: Scratch<'a>,
    cores: Vec<(Pass, Arc<KernelCore>)>,
    reduce_cores: Vec<(AxisPass, Arc<KernelCore>)>,
    ty: &'static str,
    /// Where the passes launch.
    emit: Emit<'a>,
}

/// A pending operand evaluated into a buffer for the scans that read it more than once, or that cannot compute it.
struct Evaluated {
    pin: crate::gpu::pager::Pin,
    bytes: u64,
}

impl Evaluated {
    fn bound(&self) -> Bound<'_> {
        Bound { buffer: self.pin.buffer(), bytes: self.bytes }
    }
}

#[inline(never)]
fn evaluate_operand(node: &Arc<Node>) -> Outcome<Evaluated> {
    let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
    Ok(Evaluated { pin: storage.pin_single_read()?, bytes: storage.byte_len() })
}

/// A buffer and the bytes bound of it.
#[derive(Clone, Copy)]
pub(crate) struct Bound<'b> {
    pub(crate) buffer: &'b Buffer,
    pub(crate) bytes: u64,
}

/// The lines a reduction runs over: `lines` lines of `n` cells, `stride` apart.
#[derive(Clone, Copy)]
struct Cells {
    n: u32,
    stride: u32,
    lines: u32,
}

/// What the first pass of a reduction reads its cells from.
#[derive(Clone, Copy)]
pub(crate) enum Input<'b> {
    Buffer(Bound<'b>),
    /// A lazy expression: computed inside the pass when it can be (see [`jit::axis_reduce`]),
    /// else evaluated into a buffer first.
    Pending(&'b Arc<Node>),
    /// Several lazy operands computed together in the pass (see [`jit::axis_plan`]); the ops of the first pass are
    /// [merged](ReduceList::merged) over its roots.
    Plan(&'b jit::AxisPlan),
}

/// The cells one invocation walks in a blocked pass over strided lines, or `None` when the lines
/// are short or numerous enough for one invocation each.
pub(crate) fn strided_block(n: u32, lines: u32, threads: u64) -> Option<u32> {
    if lines >= WIDE_LINES || n <= SHORT_STRIDED {
        return None;
    }
    let block = (u64::from(n) * u64::from(lines)).div_ceil(threads).clamp(u64::from(MIN_BLOCK), u64::from(n));
    Some(block as u32)
}

/// Strided lines of at most [`SHORT_STRIDED`] cells are walked by one invocation each (see
/// [`strided_block`]), which leaves the device idle when there are few of them: the bias gradient
/// of a batch of 256 over 10 classes is 10 invocations of 256 cells each and took 38 us on an
/// RTX 3090 with its operand recomputed inside (the weights of 512 units, 512 invocations: 11 us),
/// against a block of 16 cells and a second pass that folds the 16 partial results of a line. Lines
/// of up to [`FEW_LINES_MIN_CELLS`] cells (and the many-lined ones) keep one invocation each.
const FEW_LINES: u32 = 1024;
const FEW_LINES_MIN_CELLS: u32 = 64;

/// [`strided_block`] for a reduction: blocked also for short lines when there are few of them.
fn reduce_block(n: u32, lines: u32, threads: u64) -> Option<u32> {
    strided_block(n, lines, threads).or_else(|| {
        (lines < FEW_LINES && n > FEW_LINES_MIN_CELLS).then(|| (u64::from(n) * u64::from(lines)).div_ceil(threads).clamp(u64::from(MIN_BLOCK), u64::from(n)) as u32)
    })
}

/// The invocations of a workgroup of the short-line pass, which is the most lines it stages at
/// once: 64 lines of up to 63 cells fill the tile, longer lines leave room for fewer.
fn rows_width(n: u32) -> u32 {
    if n < 64 { 64 } else { 32 }
}

/// Lines of `n` cells a workgroup of the short-line pass stages at once (each row is padded to
/// an odd length, so the invocations read different banks); `roots` tiles share the workgroup memory.
fn rows_per_group(n: u32, roots: u32) -> u32 {
    (ROWS_TILE / roots / (n | 1)).min(rows_width(n))
}

/// The roots of the program a first pass of `ops` reduces (1 unless the list is merged over several).
pub(crate) fn root_count(ops: &ReduceList) -> u32 {
    ops.roots().map_or(1, |roots| roots.iter().map(|&root| u32::from(root)).max().unwrap_or(0) + 1)
}

/// Most reductions along an axis recorded together in one kernel (see [`reduce_axis_planned`]): every op keeps an accumulator
/// per invocation and a workgroup array, and every distinct operand a tile of the short-line pass.
pub(crate) const AXIS_MULTI: usize = 8;

/// The operator of the reductions, which have none of their own.
fn no_operator() -> Operator {
    Operator { source: String::new(), scalar_tys: Vec::new(), scalars: Vec::new(), extremum: false }
}

/// Reduces the lines `geom` of the `f32` expression `node` with `op` into `out`: the passes of [`Array::reduce_axis`] of one op,
/// the first computing the expression where it can.
pub(crate) fn reduce_axis_node(emit: &mut Emit<'_>, node: &Arc<Node>, geom: crate::reduce::AxisGeom, op: ReduceOp, out: Bound<'_>) -> Outcome<()> {
    let operator = no_operator();
    let mut run = Lines::new(emit.ctx(), &operator, "f32");
    run.emit = emit.clone();
    let result = run.reduce(Input::Pending(node), Cells { n: geom.n, stride: geom.stride, lines: geom.lines }, ReduceList::new(&[op]), false, &[out]);
    emit.absorb(&run.emit);
    result
}

/// Reduces the lines `geom` of the operands of `plan` in the passes of [`Array::reduce_axis`]: op `j` of `ops` reduces its own
/// operand (see [`jit::AxisPlan::roots`]) into `outs[j]`, every op combining the cells of its lines in the order a reduction
/// of that operand alone does, so the values are the same bits. One kernel computes the operands together (once what they share).
pub(crate) fn reduce_axis_planned(emit: &mut Emit<'_>, plan: &jit::AxisPlan, geom: crate::reduce::AxisGeom, ops: &[ReduceOp], outs: &[Bound<'_>]) -> Outcome<()> {
    let operator = no_operator();
    let mut run = Lines::new(emit.ctx(), &operator, "f32");
    run.emit = emit.clone();
    let list = ReduceList::merged(ops, plan.roots(), &vec![crate::builtin::Epilogue::default(); ops.len()]);
    let result = run.reduce(Input::Plan(plan), Cells { n: geom.n, stride: geom.stride, lines: geom.lines }, list, false, outs);
    emit.absorb(&run.emit);
    result
}

/// Chunks a workgroup of [`Pass::ChunkLookback`] reads the status of at a time, one per
/// invocation. Tests narrow it: on a GPU hundreds of chunks are in flight and most look-backs walk
/// several windows, which a CPU running a few workgroups at a time hardly ever does.
fn lookback_window() -> u32 {
    #[cfg(test)]
    if let Some(window) = tests::WINDOW.get() {
        return window;
    }
    WORKGROUP
}

/// The operator `+` on `u32`, for [`Lines::words`].
pub(crate) fn add_operator(ctx: &Context) -> Outcome<Operator> {
    Array::<u32>::alloc(ctx, Shape::from(1usize))?.trace_operator(&add)
}

impl<'a> Lines<'a> {
    fn new(ctx: &'a Context, op: &'a Operator, ty: &'static str) -> Self {
        Lines { ctx, op, scratch: Scratch::new(ctx), cores: Vec::new(), reduce_cores: Vec::new(), ty, emit: Emit::now(ctx) }
    }

    /// The scan kernels of `op`, which must be [`add_operator`] or another operator on `u32`.
    pub(crate) fn words(ctx: &'a Context, op: &'a Operator) -> Self {
        Lines::new(ctx, op, "u32")
    }

    fn core(&mut self, pass: Pass) -> Result<Arc<KernelCore>> {
        if let Some((_, core)) = self.cores.iter().find(|(p, _)| *p == pass) {
            return Ok(Arc::clone(core));
        }
        let workgroup = self.ctx.tuning().elementwise_workgroup;
        let key = format!("{pass:?} {} {workgroup} {:?} {}", self.ty, self.op.scalar_tys, self.op.source);
        let core = self.ctx.jit().scan_kernel(self.ctx, key, pass.label(), || source(pass, self.ty, self.op, workgroup))?;
        self.cores.push((pass, Arc::clone(&core)));
        Ok(core)
    }

    fn reduce_core(&mut self, pass: AxisPass) -> Result<Arc<KernelCore>> {
        if let Some((_, core)) = self.reduce_cores.iter().find(|(p, _)| *p == pass) {
            return Ok(Arc::clone(core));
        }
        let workgroup = self.ctx.tuning().elementwise_workgroup;
        let core = self.ctx.jit().stencil_kernel(self.ctx, reduce_source(pass, self.ty, workgroup), pass.label())?;
        self.reduce_cores.push((pass, Arc::clone(&core)));
        Ok(core)
    }

    /// The workgroups of `pass` over lines of the `geometry` and the `luma.size` of the launch.
    fn grid(&self, pass: Pass, geometry: [u32; 5]) -> Result<([u32; 3], [u32; 3])> {
        let (lines, nb) = (geometry[2], geometry[3]);
        let groups = match pass {
            Pass::Rows { .. } | Pass::Warp { .. } => lines.div_ceil(nb),
            _ if pass.per_workgroup() => lines.checked_mul(nb).ok_or(LumaError::InvalidShape("too many chunks".into()))?,
            _ => {
                let count = if pass == Pass::Seq { lines } else { lines * nb };
                let workgroups = fold_workgroups(self.ctx, count.div_ceil(self.ctx.tuning().elementwise_workgroup))?;
                return Ok((workgroups, [count, 1, 1]));
            }
        };
        Ok((fold_workgroups(self.ctx, groups)?, [groups, 1, 1]))
    }

    fn launch(
        &mut self,
        pass: Pass,
        geometry: [u32; 5],
        out: Bound<'_>,
        input: Bound<'_>,
        pre: Option<Bound<'_>>,
    ) -> Result<()> {
        let core = self.core(pass)?;
        let mut words = geometry.to_vec();
        words.extend_from_slice(&self.op.scalars);
        let mut args = vec![RawArg::write(out.buffer, out.bytes), RawArg::read(input.buffer, input.bytes)];
        if let Some(pre) = pre {
            // The statuses of a look-back are written by the chunks that wait for each other.
            args.push(if pass == Pass::ChunkLookback { RawArg::write(pre.buffer, pre.bytes) } else { RawArg::read(pre.buffer, pre.bytes) });
        }
        args.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
        let (workgroups, size) = self.grid(pass, geometry)?;
        self.emit.launch(&core, workgroups, size, &args)
    }

    /// Whether a scan with this operator gives the same result however it groups its operands, so
    /// that the chunks may take the prefix of whichever chunk before them is ready first (see
    /// [`Pass::ChunkLookback`]) and a scan stays reproducible: the integer types wrap, so every
    /// associative operator on them is exact, and `min` and `max` of floats pick one of their
    /// operands. The sums and products of floats round differently in other groupings and keep
    /// the fixed grouping of the two passes.
    fn grouping_free(&self) -> bool {
        matches!(self.ty, "u32" | "i32") || self.op.extremum
    }

    /// The inclusive scan of `lines` lines of `n` cells (more than one chunk of
    /// [`Pass::ChunkLookback`]) in one pass over `input` (see there): a launch that zeroes the
    /// counter and the statuses of the chunks, then the scan. Both are launches, so a recording
    /// or a graph of the caller replays them together and every replay starts from zero.
    fn lookback(&mut self, n: u32, lines: u32, out: Bound<'_>, input: Input<'_>) -> Outcome<()> {
        let nb = n.div_ceil(LOOKBACK_CHUNK);
        // `k`, the chunk size of the other passes, is the window of the look-back here.
        let geometry = [n, 1, lines, nb, lookback_window()];
        let chunks = u64::from(geometry[2]) * u64::from(geometry[3]);
        let (status, bytes) = self.temp(1 + 3 * chunks)?;
        let zeroed = u32::try_from(1 + chunks).map_err(|_| LumaError::InvalidShape("too many chunks".into()))?;
        let zero = self.ctx.builtin(Builtin::Fill, "u32")?;
        launch_1d(self.ctx, &zero, zeroed, &[RawArg::write(&status, u64::from(zeroed) * 4), RawArg::Uniform(bytemuck::bytes_of(&0u32))])?;
        self.launch_scan(Pass::ChunkLookback, geometry, out, input, Some(Bound { buffer: &status, bytes }))
    }

    /// One scan pass over `input` into `out`: a buffer is read as is; a pending operand is computed in the kernel (a fused first
    /// read, see [`scan_body`]) unless it cannot be (see [`jit::axis_reduce`]), when it is evaluated first. `pre` is the prefix of the
    /// blocks before (seeded passes) or the statuses of a look-back. Only `+` of a pending operand is fused: that is the only
    /// operator the fused kernels know, and the only one `run_scan` leaves pending.
    fn launch_scan(&mut self, pass: Pass, geometry: [u32; 5], out: Bound<'_>, input: Input<'_>, pre: Option<Bound<'_>>) -> Outcome<()> {
        match input {
            Input::Buffer(input) => Ok(self.launch(pass, geometry, out, input, pre)?),
            Input::Pending(node) => self.launch_pending(pass, geometry, out, node, pre),
            Input::Plan(_) => unreachable!("scans have no planned operands"),
        }
    }

    /// [`Lines::launch_scan`] of a pending operand (kept out of the path of buffers, which a scan of a few cells makes hot).
    #[inline(never)]
    fn launch_pending(&mut self, pass: Pass, geometry: [u32; 5], out: Bound<'_>, node: &Arc<Node>, pre: Option<Bound<'_>>) -> Outcome<()> {
        let (workgroups, size) = self.grid(pass, geometry)?;
        let scan = AxisPass { pass, ops: ReduceList::new(&[ReduceOp::Sum]), partial: false, split: false, line_major: false, scan: true };
        let writes = [RawArg::write(out.buffer, out.bytes)];
        let reads: Vec<RawArg<'_>> = pre.iter().map(|pre| if pass == Pass::ChunkLookback { RawArg::write(pre.buffer, pre.bytes) } else { RawArg::read(pre.buffer, pre.bytes) }).collect();
        if jit::axis_reduce(&mut self.emit, node, scan, geometry, workgroups, size, &writes, &reads)? {
            return Ok(());
        }
        let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
        let evaluated = storage.pin_single_read()?;
        Ok(self.launch(pass, geometry, out, Bound { buffer: evaluated.buffer(), bytes: storage.byte_len() }, pre)?)
    }

    /// One reducing pass of `ops` over `input` into `outs`: a buffer for every op if there are
    /// several, else the one buffer that holds op `j` at `j * count` (see [`AxisPass::split`]).
    fn launch_reduce(
        &mut self,
        pass: Pass,
        ops: ReduceList,
        partial: bool,
        geometry: [u32; 5],
        outs: &[Bound<'_>],
        input: Input<'_>,
    ) -> Outcome<()> {
        let pass = AxisPass { pass, ops, partial, split: outs.len() > 1, line_major: line_major(pass, geometry[3]), scan: false };
        let (workgroups, size) = self.grid(pass.pass, geometry)?;
        let writes: Vec<RawArg<'_>> = outs.iter().map(|out| RawArg::write(out.buffer, out.bytes)).collect();
        let evaluated;
        let input = match input {
            Input::Buffer(input) => input,
            Input::Plan(plan) => {
                jit::axis_reduce_planned(&mut self.emit, plan, pass, geometry, workgroups, size, &writes)?;
                return Ok(());
            }
            Input::Pending(node) => {
                if jit::axis_reduce(&mut self.emit, node, pass, geometry, workgroups, size, &writes, &[])? {
                    return Ok(());
                }
                let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
                evaluated = storage.pin_single_read()?;
                Bound { buffer: evaluated.buffer(), bytes: storage.byte_len() }
            }
        };
        let core = self.reduce_core(pass)?;
        let mut args = writes;
        args.push(RawArg::read(input.buffer, input.bytes));
        args.push(RawArg::Uniform(bytemuck::cast_slice(&geometry)));
        self.emit.launch(&core, workgroups, size, &args)?;
        Ok(())
    }

    fn temp(&mut self, elements: u64) -> Result<(Buffer, u64)> {
        let bytes = elements * 4;
        Ok((self.scratch.take(bytes)?, bytes))
    }

    /// The inclusive scan of every line of `input` into `out`.
    #[track_caller]
    pub(crate) fn scan(&mut self, input: Bound<'_>, n: u32, stride: u32, lines: u32, out: Bound<'_>) -> Result<()> {
        self.scan_input(Input::Buffer(input), n, stride, lines, out).map_err(Failure::into_error)
    }

    /// [`Lines::scan`] of an operand that may be pending (`+` only, see [`Lines::launch_scan`]). The strategy comes from the registry
    /// (`families::scan`): the order of its table is the order of the conditions this function had. A pending operand is computed
    /// in the kernels that read it when the strategy takes a prologue (`Strategy::prologue`) and recomputing it beats storing it.
    #[track_caller]
    pub(crate) fn scan_input(&mut self, input: Input<'_>, n: u32, stride: u32, lines: u32, out: Bound<'_>) -> Outcome<()> {
        let operand = match input {
            Input::Pending(node) if crate::jit::worth_storing(node) => ScanOperand::Costly,
            Input::Pending(_) => ScanOperand::Cheap,
            Input::Buffer(_) | Input::Plan(_) => ScanOperand::Buffer,
        };
        let boundary = Boundary::new(FamilyTag::Scan, "scan", Location::caller(), [n, stride, lines, 0], BoundaryAttrs::Scan(ScanAttrs { grouping_free: self.grouping_free(), by_key: false, operand }));
        let (id, params) = first_candidate(self.ctx, &boundary).expect("a scan strategy always fits");
        note(self.ctx, &boundary, id, &params, &[]);
        let Params::Scan(params) = params else { unreachable!("a scan strategy runs with scan parameters") };
        // The operand the kernels read: the pending expression where they compute it, else its buffer (evaluated once for all of them).
        let evaluated;
        let input = match input {
            Input::Pending(node) if !params.fused => {
                evaluated = evaluate_operand(node)?;
                Input::Buffer(evaluated.bound())
            }
            other => other,
        };
        match id {
            StrategyId::SCAN_ROWS => self.launch_scan(Pass::Rows { fold: false, width: rows_width(n) }, [n, 1, lines, rows_per_group(n, 1), n], out, input, None),
            StrategyId::SCAN_WARP => self.launch_scan(Pass::Warp { vec: params.vec }, [n, 1, lines, 1, n], out, input, None),
            StrategyId::SCAN_LOOKBACK => self.lookback(n, lines, out, input),
            StrategyId::SCAN_CHUNK_SINGLE => self.launch_scan(Pass::ChunkScan { seeded: false }, [n, 1, lines, 1, SCAN_CHUNK], out, input, None),
            StrategyId::SCAN_CHUNK_THREE_PASS => {
                let nb = params.nb;
                let geometry = [n, 1, lines, nb, SCAN_CHUNK];
                let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
                let totals = Bound { buffer: &totals, bytes };
                self.launch_scan(Pass::ChunkFold, geometry, totals, input, None)?;
                let (prefix, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
                let prefix = Bound { buffer: &prefix, bytes };
                self.scan_input(Input::Buffer(totals), nb, 1, lines, prefix)?;
                self.launch_scan(Pass::ChunkScan { seeded: true }, geometry, out, input, Some(prefix))
            }
            StrategyId::SCAN_SEQ => self.launch_scan(Pass::Seq, [n, stride, lines, 1, n], out, input, None),
            StrategyId::SCAN_FOLD_WALK => {
                let (nb, k) = (params.nb, params.k);
                let geometry = [n, stride, lines, nb, k];
                let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
                let totals = Bound { buffer: &totals, bytes };
                self.launch_scan(Pass::Fold, geometry, totals, input, None)?;
                let (prefix, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
                let prefix = Bound { buffer: &prefix, bytes };
                self.scan_input(Input::Buffer(totals), nb, lines, lines, prefix)?;
                self.launch_scan(Pass::Walk, geometry, out, input, Some(prefix))
            }
            other => unreachable!("{other:?} is not a strategy of a scan along an axis"),
        }
    }

    /// The reductions `ops` of every line of `input` into `outs` (see [`Lines::launch_reduce`]);
    /// `partial` if `input` holds the partial results of an earlier pass. The ops are commutative
    /// as well as associative. The first pass reads the expression behind a pending input
    /// directly; the passes over partial results read buffers.
    fn reduce(&mut self, input: Input<'_>, cells: Cells, ops: ReduceList, partial: bool, outs: &[Bound<'_>]) -> Outcome<()> {
        let Cells { n, stride, lines } = cells;
        if stride == 1 && n <= SHORT_LINE {
            return self.launch_reduce(Pass::Rows { fold: true, width: rows_width(n) }, ops, partial, [n, 1, lines, rows_per_group(n, root_count(&ops)), n], outs, input);
        }
        let (pass, geometry, nb) = if stride == 1 {
            let nb = n.div_ceil(REDUCE_CHUNK);
            (Pass::ChunkReduce, [n, 1, lines, nb, REDUCE_CHUNK], nb)
        } else {
            let k = reduce_block(n, lines, target_threads(self.ctx.tuning())).unwrap_or(n);
            let nb = n.div_ceil(k);
            (Pass::Fold, [n, stride, lines, nb, k], nb)
        };
        if nb == 1 {
            return self.launch_reduce(pass, ops, partial, geometry, outs, input);
        }
        let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb) * ops.cell_words() as u64)?;
        let totals = Bound { buffer: &totals, bytes };
        self.launch_reduce(pass, ops, partial, geometry, &[totals], input)?;
        // The partial results of contiguous lines (and line-major ones) are lines of `nb` cells,
        // those of strided ones `nb` rows of `lines` cells.
        let next = if line_major(pass, nb) {
            Cells { n: nb, stride: 1, lines }
        } else {
            Cells { n: nb, stride: if stride == 1 { 1 } else { lines }, lines }
        };
        self.reduce(Input::Buffer(totals), next, ops.rest(), true, outs)
    }
}

/// The traced operator `+` of every element type (see [`Array::sum_operator`]).
type SumOperators = Mutex<Vec<(&'static str, Arc<Operator>)>>;

/// The operator of a scan: an elementwise closure of two arguments.
pub(crate) type ScanOp<'a, T> = dyn Fn(&Array<T>, &Array<T>) -> Array<T> + 'a;

fn not_elementwise() -> Failure {
    LumaError::InvalidShape("the scan operator must be an elementwise expression of its two arguments".into()).into()
}

impl<T: ScalarType> Array<T> {
    /// `op` traced on two one-element placeholders as a WGSL function.
    pub(crate) fn trace_operator(&self, op: &ScanOp<'_, T>) -> Outcome<Operator> {
        let ctx = self.context();
        let a = Array::<T>::alloc(ctx, Shape::from(1usize))?;
        let b = Array::<T>::alloc(ctx, Shape::from(1usize))?;
        let r = op(&a, &b);
        r.src()?;
        match r.pending() {
            Some(node) if r.len() == 1 => operator_source(node, a.storage()?, b.storage()?)?.ok_or_else(not_elementwise),
            _ => Err(not_elementwise()),
        }
    }

    /// `+` as an [`Operator`], traced on the first call for the element type: the trace allocates
    /// two arrays and builds, flattens and prints an expression, which takes a few microseconds
    /// of a `cumsum` that a bandwidth-bound kernel of 0.17 ms leaves idle. The operator is
    /// the same text on every device.
    fn sum_operator(&self) -> Outcome<Arc<Operator>> {
        static SUMS: LazyLock<SumOperators> = LazyLock::new(Mutex::default);
        let ty = T::wgsl_type();
        let known = SUMS.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(known, _)| *known == ty).map(|(_, op)| Arc::clone(op));
        if let Some(op) = known {
            return Ok(op);
        }
        let op = Arc::new(self.trace_operator(&add)?);
        SUMS.lock().unwrap_or_else(PoisonError::into_inner).push((ty, Arc::clone(&op)));
        Ok(op)
    }

    /// Runs the scan of `op` along `axis`; `sum` if `op` is `+` (see [`Array::sum_operator`]).
    fn run_scan(&self, axis: usize, op: &ScanOp<'_, T>, sum: bool) -> Outcome<Array<T>> {
        let ctx = self.context();
        if ElemTy::of::<T>()?.layout() != ElemLayout::plain(4) {
            self.ready()?;
            return Err(LumaError::Unsupported(format!("scans and reductions along an axis are not available for {} arrays", ElemTy::of::<T>()?.display())).into());
        }
        let shape = self.shape();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        let operator = if sum { self.sum_operator()? } else { Arc::new(self.trace_operator(op)?) };
        let out = Array::<T>::alloc(ctx, shape)?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let (n, stride, lines) = self.line_geometry(axis)?;
        let mut run = Lines::new(ctx, &operator, T::wgsl_type());
        // The sum of a pending operand computes it in the scan's own kernels (a mask or a cast feeding a `cumsum` is never written);
        // any other operator needs the operand in a buffer.
        let ready;
        let input = match self.pending() {
            Some(node) if sum => Input::Pending(node),
            _ => {
                let (pin, bytes) = self.ready()?;
                ready = pin;
                Input::Buffer(Bound { buffer: ready.buffer(), bytes })
            }
        };
        let target = Bound { buffer: out_pin.buffer(), bytes: out.byte_len() };
        run.scan_input(input, n, stride, lines, target)?;
        drop(run);
        crate::jit::check_finite(ctx, &out_storage, ElemTy::of::<T>()?)?;
        Ok(out)
    }

    /// The extent of `axis`, the distance of the cells of a line and the number of lines.
    fn line_geometry(&self, axis: usize) -> Outcome<(u32, u32, u32)> {
        let dims = self.shape().dims();
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: dims }));
        let stride: usize = dims[..axis].iter().product();
        Ok((to_u32(dims[axis])?, to_u32(stride)?, to_u32(self.len() / dims[axis])?))
    }

    /// Runs the reductions `ops` along `axis`: one array per op. A pending operand is computed
    /// inside the first pass unless it cannot be (see [`jit::axis_reduce`]).
    fn run_axis_reduce(&self, axis: usize, ops: ReduceList) -> Outcome<Vec<Array<T>>> {
        let ctx = self.context();
        let ty = ElemTy::of::<T>()?;
        if ty.layout() != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported(format!("scans and reductions along an axis are not available for {} arrays", ty.display())).into());
        }
        let shape = self.shape();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        let mut out_dims = shape.dims();
        out_dims[axis] = 1;
        if ctx.can_run_on_host() && self.len() <= crate::host::MAX_HOST_ELEMS && self.error().is_none() {
            let (n, stride, lines) = self.line_geometry(axis)?;
            if let Some(outs) = crate::host::try_axis_reduce(self, [n, stride, lines], out_dims, ops) {
                return Ok(outs);
            }
        }
        let outs = ops.as_slice().iter().map(|_| Array::<T>::alloc(ctx, Shape::from(out_dims))).collect::<Result<Vec<_>>>()?;
        let storages = outs.iter().map(|out| out.storage().map(Arc::clone)).collect::<std::result::Result<Vec<_>, _>>()?;
        let pins = storages.iter().map(|storage| storage.pin_single()).collect::<Result<Vec<_>>>()?;
        let (n, stride, lines) = self.line_geometry(axis)?;
        let ready;
        let input = match self.pending() {
            Some(node) => Input::Pending(node),
            None => {
                let (pin, bytes) = self.ready()?;
                ready = pin;
                Input::Buffer(Bound { buffer: ready.buffer(), bytes })
            }
        };
        let targets: Vec<Bound<'_>> = pins.iter().zip(&outs).map(|(pin, out)| Bound { buffer: pin.buffer(), bytes: out.byte_len() }).collect();
        let operator = Operator { source: String::new(), scalar_tys: Vec::new(), scalars: Vec::new(), extremum: false };
        let mut run = Lines::new(ctx, &operator, T::wgsl_type());
        run.reduce(input, Cells { n, stride, lines }, ops, false, &targets)?;
        drop(run);
        for storage in &storages {
            crate::jit::check_finite(ctx, storage, ty)?;
        }
        Ok(outs)
    }

    /// Runs the arg reduction `op` over the lines `cells` of this array into an array of `out_dims`.
    /// A pending operand is computed inside the first pass unless it cannot be (see
    /// [`jit::axis_reduce`]).
    fn run_arg_reduce(&self, cells: Cells, out_dims: [usize; 4], op: ReduceOp, name: &str) -> Outcome<Array<u32>> {
        let ctx = self.context();
        let ty = ElemTy::of::<T>()?;
        if ty.layout() != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported(format!("{name} is not available for {} arrays", ty.display())).into());
        }
        if ctx.can_run_on_host()
            && self.len() <= crate::host::MAX_HOST_ELEMS
            && self.error().is_none()
            && let Some(out) = crate::host::try_arg_reduce(self, [cells.n, cells.stride, cells.lines], out_dims, op)
        {
            return Ok(out);
        }
        let out = Array::<u32>::alloc(ctx, Shape::from(out_dims))?;
        let storage = Arc::clone(out.storage()?);
        let pin = storage.pin_single()?;
        let ready;
        let input = match self.pending() {
            Some(node) => Input::Pending(node),
            None => {
                let (pin, bytes) = self.ready()?;
                ready = pin;
                Input::Buffer(Bound { buffer: ready.buffer(), bytes })
            }
        };
        let target = Bound { buffer: pin.buffer(), bytes: out.byte_len() };
        let operator = Operator { source: String::new(), scalar_tys: Vec::new(), scalars: Vec::new(), extremum: false };
        let mut run = Lines::new(ctx, &operator, T::wgsl_type());
        run.reduce(input, cells, ReduceList::new(&[op]), false, &[target])?;
        Ok(out)
    }

    /// The arg reduction `op` along `axis` (`None`: of the whole array in flat order).
    fn arg_lines(&self, axis: Option<usize>, op: ReduceOp, name: &'static str, location: &'static Location<'static>) -> Array<u32> {
        let mut dims = self.shape().dims();
        let result = match axis {
            Some(axis) if axis >= 4 => Err(LumaError::InvalidShape(format!("axis {axis} of shape {}", self.shape())).into()),
            Some(axis) => {
                dims[axis] = 1;
                self.line_geometry(axis).and_then(|(n, stride, lines)| self.run_arg_reduce(Cells { n, stride, lines }, dims, op, name))
            }
            None => {
                dims = [1; 4];
                u32::try_from(self.len())
                    .map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: self.shape().dims() }))
                    .and_then(|n| self.run_arg_reduce(Cells { n, stride: 1, lines: 1 }, dims, op, name))
            }
        };
        Array::resolve(self.context(), Shape::from(dims), result, name, location)
    }

    #[track_caller]
    fn scan_axis(&self, axis: usize, op: &ScanOp<'_, T>, sum: bool, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_scan(axis, op, sum);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    /// `reduce_axis` for the caller `name` at `location`.
    fn reduce_lines<const N: usize>(&self, axis: usize, ops: [Reduction; N], name: &'static str, location: &'static Location<'static>) -> [Array<T>; N] {
        let mut dims = self.shape().dims();
        if axis < 4 {
            dims[axis] = 1;
        }
        let shape = Shape::from(dims);
        // Left pending where they can be, so that the reductions of one operand (or of expressions that share work) that a program makes
        // one after another are recorded in one kernel when they are used (see `jit::multireduce`).
        if (1..=MAX_REDUCTIONS).contains(&N)
            && axis < 4
            && !self.is_empty()
            && let Ok((n, stride, lines)) = self.line_geometry(axis)
            && let Some(arrays) = self.defer_axis(&ops.map(Reduction::op), crate::reduce::AxisGeom { n, stride, lines }, shape, name, location)
        {
            let mut arrays = arrays.into_iter();
            return std::array::from_fn(|_| arrays.next().expect("one array per reduction"));
        }
        let result = if (1..=MAX_REDUCTIONS).contains(&N) {
            self.run_axis_reduce(axis, ReduceList::new(&ops.map(Reduction::op)))
        } else {
            Err(LumaError::InvalidShape(format!("{N} reductions along an axis: one to {MAX_REDUCTIONS} are computed together")).into())
        };
        let (done, failure) = match result {
            Ok(done) => (done, None),
            Err(failure) => (Vec::new(), Some(failure)),
        };
        let mut done = done.into_iter();
        std::array::from_fn(|_| {
            let result = match (done.next(), &failure) {
                (Some(array), _) => Ok(array),
                (None, Some(Failure::New(error))) => Err(Failure::New(error.clone())),
                (None, Some(Failure::Inherited(error))) => Err(Failure::Inherited(Arc::clone(error))),
                (None, None) => unreachable!("a reduction without an array or a failure"),
            };
            Array::resolve(self.context(), shape, result, name, location)
        })
    }
}

impl<T: Numeric> Array<T> {
    /// The inclusive scan of `op` along `axis`: element `s` of every line is
    /// `op(.. op(op(x[0], x[1]), x[2]) .., x[s])`. `op` must be associative (the kernels regroup
    /// the operands, keeping their order, so a non-associative operator gives unspecified
    /// results) and an elementwise expression of its two arguments: it is traced once on
    /// one-element placeholders and compiled into the scan kernels; anything else poisons the
    /// result with [`LumaError::InvalidShape`].
    ///
    /// Many contiguous lines (CUDA and other devices with 32-wide subgroups) are one kernel with a
    /// subgroup per line, reading and writing every cell once; the grouping of the operands, and
    /// so the rounding of `f32` sums, differs from the passes below but is fixed. Otherwise
    /// contiguous lines are cut in chunks that a workgroup scans in workgroup memory (chunk
    /// totals, a scan of the totals, a second pass seeded with them; on CUDA integer scans and
    /// `min` / `max` of long lines are one pass in which the chunks wait for each other's
    /// prefix, which gives the same results as they do not depend on the grouping), strided
    /// lines in blocks that one invocation walks, with neighbouring invocations on neighbouring
    /// lines, so every pass reads and writes memory in coalesced accesses; short lines are
    /// staged several to a workgroup. The arrays are evaluated first; arrays split over several
    /// buffers fail with [`LumaError::Chunked`]. Scalars captured by `op` are uniforms, so
    /// changing them does not recompile. The results of an associative `op` are the same on every run.
    #[track_caller]
    pub fn scan(&self, axis: usize, op: impl Fn(&Array<T>, &Array<T>) -> Array<T>) -> Array<T> {
        self.scan_axis(axis, &op, false, "scan", Location::caller())
    }

    /// The running sum along `axis` (a [`scan`](Array::scan) with `+`).
    #[track_caller]
    pub fn cumsum(&self, axis: usize) -> Array<T> {
        self.scan_axis(axis, &add, true, "cumsum", Location::caller())
    }

    /// The sum along `axis`; the result has extent 1 along it. A pending operand is not
    /// evaluated first: its expression is computed inside the first pass of the reduction.
    /// An `f32` result stays pending until it is used, like [`Array::sum`]: it is recorded then,
    /// in one kernel with the axis reductions over the same lines that are pending (`mean_axis`
    /// and `var_axis` of one array, the three components of a force, a table's column means and
    /// extrema), each of them with the value it has alone (the same bits on the CPU backend, within an ulp or two on GPU backends).
    #[track_caller]
    pub fn sum_axis(&self, axis: usize) -> Array<T> {
        let [sum] = self.reduce_lines(axis, [Reduction::Sum], "sum_axis", Location::caller());
        sum
    }

    /// The minimum along `axis`; the result has extent 1 along it. Fused like
    /// [`sum_axis`](Array::sum_axis).
    #[track_caller]
    pub fn min_axis(&self, axis: usize) -> Array<T> {
        let [min] = self.reduce_lines(axis, [Reduction::Min], "min_axis", Location::caller());
        min
    }

    /// The maximum along `axis`; the result has extent 1 along it. Fused like
    /// [`sum_axis`](Array::sum_axis).
    #[track_caller]
    pub fn max_axis(&self, axis: usize) -> Array<T> {
        let [max] = self.reduce_lines(axis, [Reduction::Max], "max_axis", Location::caller());
        max
    }

    /// Up to four reductions along `axis` in one pass over the data: result `j` is the
    /// reduction `ops[j]`, with extent 1 along `axis`. Every cell is read once for all of them,
    /// and a pending operand is computed inside that first pass, never stored. Like
    /// [`sum_axis`](Array::sum_axis), `f32` results are recorded when one of them is used (on a GPU
    /// backend, outside a step that is being recorded), the others pending over the same lines
    /// with them in one kernel; any other result is evaluated when the call returns. Another
    /// number of reductions poisons every result with [`LumaError::InvalidShape`].
    ///
    /// [`Reduction::SumOfSquares`] of integers wraps, as [`Array::reduce`] does. Contiguous lines
    /// (`axis` 0) are reduced by a workgroup per chunk of a line, short ones a group of lines per
    /// workgroup, strided lines by blocks that an invocation walks (see [`scan`](Array::scan)).
    /// `bool` arrays have no `reduce_axis`: their sums are counts (see
    /// [`Array<bool>::sum_axis`](Array::sum_axis)) and their extrema `min_axis`/`max_axis`.
    #[track_caller]
    pub fn reduce_axis<const N: usize>(&self, axis: usize, ops: [Reduction; N]) -> [Array<T>; N] {
        self.reduce_lines(axis, ops, "reduce_axis", Location::caller())
    }

    /// The flat index of the largest element, as a one-element `u32` array (NumPy's `argmax` of
    /// the flattened array: the first of equal elements, the first NaN if there is one). A pending
    /// operand is computed inside the first pass, never stored. The passes are recorded when the
    /// call returns and nothing is read back, so the index can feed other expressions on the GPU
    /// (`x.take(&x.argmax())`); [`Array::item`] reads it on the host (a synchronization, about
    /// 100 µs). For a position in several dimensions, divide and take the remainder by the
    /// extents (axis 0 varies fastest).
    ///
    /// Available for `u32`, `i32` and `f32` arrays; the arrays of other types (`f64`, `Df32`)
    /// poison the result with [`LumaError::Unsupported`]. Arrays split over several buffers fail
    /// with [`LumaError::Chunked`]. See [`argmax_axis`](Array::argmax_axis) for the passes.
    #[track_caller]
    pub fn argmax(&self) -> Array<u32> {
        self.arg_lines(None, ReduceOp::ArgMax, "argmax", Location::caller())
    }

    /// The flat index of the smallest element, as a one-element `u32` array: the first of equal
    /// elements, the first NaN if there is one (as NumPy). Computed like
    /// [`argmax`](Array::argmax).
    #[track_caller]
    pub fn argmin(&self) -> Array<u32> {
        self.arg_lines(None, ReduceOp::ArgMin, "argmin", Location::caller())
    }

    /// The index along `axis` of the largest element of every line (the first of equal elements,
    /// the first NaN if there is one, as NumPy); the result has extent 1 along `axis`. Like
    /// [`reduce_axis`](Array::reduce_axis), a pending operand is computed inside the first pass,
    /// and the arrays are evaluated when the call returns. The passes of [`sum_axis`](Array::sum_axis)
    /// carrying the pair (value, index), so an `argmax_axis` reads the data once and takes the time
    /// of a `max_axis`. Available for `u32`, `i32` and `f32` (see [`argmax`](Array::argmax)).
    #[track_caller]
    pub fn argmax_axis(&self, axis: usize) -> Array<u32> {
        self.arg_lines(Some(axis), ReduceOp::ArgMax, "argmax_axis", Location::caller())
    }

    /// The index along `axis` of the smallest element of every line, as
    /// [`argmax_axis`](Array::argmax_axis).
    #[track_caller]
    pub fn argmin_axis(&self, axis: usize) -> Array<u32> {
        self.arg_lines(Some(axis), ReduceOp::ArgMin, "argmin_axis", Location::caller())
    }
}

impl Array<bool> {
    /// The number of true elements along `axis`, as `u32`; the result has extent 1 along it
    /// (the sum of booleans is an integer, as in NumPy). Fused like [`Array::sum_axis`].
    #[track_caller]
    pub fn sum_axis(&self, axis: usize) -> Array<u32> {
        self.cast_at::<u32>(Location::caller()).sum_axis(axis)
    }

    /// The running count of true elements along `axis`, as `u32`.
    #[track_caller]
    pub fn cumsum(&self, axis: usize) -> Array<u32> {
        self.cast_at::<u32>(Location::caller()).cumsum(axis)
    }

    /// Whether every element along `axis` is true (the minimum); the result has extent 1 along
    /// it. The reduction runs on the elements as `u32`s, fused like [`Array::sum_axis`], and the
    /// result is a lazy comparison of it.
    #[track_caller]
    pub fn min_axis(&self, axis: usize) -> Array<bool> {
        self.cast_at::<u32>(Location::caller()).min_axis(axis).ne(0u32)
    }

    /// Whether any element along `axis` is true (the maximum); the result has extent 1 along it.
    /// Computed like [`Array<bool>::min_axis`](Array::min_axis).
    #[track_caller]
    pub fn max_axis(&self, axis: usize) -> Array<bool> {
        self.cast_at::<u32>(Location::caller()).max_axis(axis).ne(0u32)
    }
}

impl Array<f32> {
    /// The mean along `axis` (the sum divided by the extent, the division lazy); the result has
    /// extent 1 along it.
    #[track_caller]
    pub fn mean_axis(&self, axis: usize) -> Array<f32> {
        let n = self.shape().dims().get(axis).copied().unwrap_or(1);
        &self.sum_axis(axis) / (n as f32)
    }

    /// The population variance along `axis` (divisor `n`, NumPy's `ddof = 0`); the result has
    /// extent 1 along it. One pass over the data: with `K` the first element of each line
    /// (a broadcast slice of the input, read inside the same pass), the sum `S1` and the sum
    /// of squares `S2` of `self - K` come from a single [`reduce_axis`](Array::reduce_axis)
    /// (the subtraction is computed inside the reduction, never stored), and the variance is
    /// `(S2 - S1 * S1 / n) / n`, which is shift invariant.
    ///
    /// Accuracy: the cancellation in `S2 - S1 * S1 / n` loses digits in proportion to
    /// `(mean - K)^2 / var`, not to `mean^2 / var` as the unshifted formula does, so data far
    /// from zero but with a spread comparable to the distance of its elements from the first
    /// (an offset of `1e4` on unit noise, a layer's activations) keeps `f32` accuracy. A line
    /// whose first element is an outlier many standard deviations from the rest loses digits
    /// (about `(outlier / sigma)^2` of the `f32` epsilon) where the two-pass formula would
    /// not; a constant line has variance exactly 0. Rounding can leave the difference
    /// slightly negative, so the result is clamped at 0 (a NaN is kept, on every backend: any NaN or infinity
    /// in a line gives NaN, as the two-pass formula did).
    #[track_caller]
    pub fn var_axis(&self, axis: usize) -> Array<f32> {
        let n = self.shape().dims().get(axis).copied().unwrap_or(1);
        let shifted = self - &self.slice(axis, ..1);
        let [s1, s2] = shifted.reduce_axis(axis, [Reduction::Sum, Reduction::SumOfSquares]);
        let n = n as f32;
        let var = &(&s2 - &(&s1 * &s1) / n) / n;
        // Clamped at 0 as `(var + |var|) / 2`, exact, keeps NaN and infinity: a compare or
        // `max` would let a driver's NaN-dropping `fmax` turn a NaN into 0.
        &(&var * 0.5f32) + &(&var.abs() * 0.5f32)
    }

    /// The population standard deviation along `axis`: the square root of
    /// [`var_axis`](Array::var_axis).
    #[track_caller]
    pub fn std_axis(&self, axis: usize) -> Array<f32> {
        self.var_axis(axis).sqrt()
    }
}

pub(crate) fn add<T: ScalarType>(x: &Array<T>, y: &Array<T>) -> Array<T> {
    x.binary(Rhs::Array(y), BinOp::Add, false, "add")
}

#[cfg(test)]
mod tests {
    //! The look-back scan (see [`Pass::ChunkLookback`]) on the CPU backend, whose atomics order
    //! memory across workgroups (the flag is on by default only for CUDA), against the host and
    //! against the two-pass scan.

    use std::cell::Cell;
    use std::fmt::Debug;

    use luma_core::{BackendKind, DeviceKind};

    use super::*;
    use crate::context::ContextOptions;

    thread_local! {
        /// The window of [`lookback_window`] on this thread, if a test narrows it.
        pub(super) static WINDOW: Cell<Option<u32>> = const { Cell::new(None) };
    }

    /// Narrows the look-back window of this thread until dropped.
    struct Window;

    impl Window {
        fn of(window: u32) -> Window {
            WINDOW.set(Some(window));
            Window
        }
    }

    impl Drop for Window {
        fn drop(&mut self) {
            WINDOW.set(None);
        }
    }

    /// A CPU context whose scans of long contiguous lines look back or take two passes.
    fn cpu(lookback: bool) -> Context {
        let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
        Context::with_tuning(options, |tuning| tuning.lookback_scan = lookback).unwrap()
    }

    /// Cells of a chunk of the look-back kernel (the two-pass kernels take 1024).
    const CHUNK: usize = LOOKBACK_CHUNK as usize;

    /// The edges of a chunk of either kind (lines of up to a chunk are scanned by the two-pass
    /// kernels), one short of one, one, one more, several, a line of many.
    const LENGTHS: [usize; 18] =
        [1, 2, 255, 256, 257, 1023, 1024, 1025, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK - 1, 2 * CHUNK, 2 * CHUNK + 1, 3 * CHUNK, 3 * CHUNK + 1, 1 << 20, (1 << 20) + 7];

    fn words(n: usize, seed: u64) -> Vec<u32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 32) as u32
            })
            .collect()
    }

    fn floats(n: usize, seed: u64) -> Vec<f32> {
        words(n, seed).into_iter().map(|w| (w >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0).collect()
    }

    /// The inclusive scan of every line of `n` cells.
    fn host_scan<T: Copy>(data: &[T], n: usize, combine: impl Fn(T, T) -> T) -> Vec<T> {
        let mut out = data.to_vec();
        for line in out.chunks_mut(n) {
            for i in 1..line.len() {
                line[i] = combine(line[i - 1], line[i]);
            }
        }
        out
    }

    fn first_difference<T: PartialEq>(got: &[T], want: &[T]) -> Option<usize> {
        assert_eq!(got.len(), want.len(), "length");
        got.iter().zip(want).position(|(g, w)| g != w)
    }

    /// Scans `data` as lines of `n` cells along axis 0 on both contexts, twice in a row (the
    /// status words must be zero again for the next launch), and checks the results are `want`.
    fn exact<T: Numeric + Copy + PartialEq + Debug>(
        contexts: (&Context, &Context),
        data: &[T],
        n: usize,
        op: &ScanOp<'_, T>,
        host: impl Fn(T, T) -> T,
        what: &str,
    ) {
        let want = host_scan(data, n, host);
        let shape = [n, data.len() / n];
        for (name, ctx) in [("look-back", contexts.0), ("two-pass", contexts.1)] {
            let x = Array::from_slice_on(ctx, data, shape).unwrap();
            for round in 0..2 {
                let got = x.scan(0, op).to_vec().unwrap();
                if let Some(at) = first_difference(&got, &want) {
                    panic!("{what}, {name}, n = {n}, lines = {}, round {round}: element {at} is {:?}, want {:?}", shape[1], got[at], want[at]);
                }
            }
        }
    }

    /// Associative but not commutative: the left operand wins (`y - y` is zero; it makes the
    /// operator an expression of both of its arguments).
    #[allow(clippy::eq_op)]
    fn first(x: &Array<u32>, y: &Array<u32>) -> Array<u32> {
        x + &(y - y)
    }

    /// Associative but not commutative: the right operand wins.
    #[allow(clippy::eq_op)]
    fn last(x: &Array<u32>, y: &Array<u32>) -> Array<u32> {
        y + &(x - x)
    }

    /// Every operator on `u32` and `i32` data, along lines of every length.
    fn every_operator(window: Option<u32>, lengths: &[usize], lines: usize) {
        let _window = window.map(Window::of);
        let (look, two) = (cpu(true), cpu(false));
        for (k, &n) in lengths.iter().enumerate() {
            let len = n * lines;
            let seed = k as u64 + 1;
            let unsigned = words(len, seed);
            let signed: Vec<i32> = unsigned.iter().map(|&w| w as i32).collect();
            let contexts = (&look, &two);
            exact(contexts, &unsigned, n, &|x, y| x + y, u32::wrapping_add, "u32 sum");
            exact(contexts, &unsigned, n, &|x, y| x.minimum(y), u32::min, "u32 min");
            exact(contexts, &unsigned, n, &|x, y| x.maximum(y), u32::max, "u32 max");
            exact(contexts, &signed, n, &|x, y| x + y, i32::wrapping_add, "i32 sum");
            exact(contexts, &signed, n, &|x, y| x.minimum(y), i32::min, "i32 min");
            exact(contexts, &signed, n, &|x, y| x.maximum(y), i32::max, "i32 max");
            // Associative but not commutative: the first cell of the line wins, so a prefix that
            // folds the chunks out of order is wrong.
            exact(contexts, &unsigned, n, &first, |x, _| x, "u32 first");
            // The last wins.
            exact(contexts, &unsigned, n, &last, |_, y| y, "u32 last");
        }
    }

    #[test]
    fn integer_scans_of_a_line_look_back() {
        every_operator(None, &LENGTHS, 1);
    }

    /// Windows that are not the workgroup: the walk over several of them, with the carry.
    #[test]
    fn narrow_windows_walk_back_over_several() {
        let lengths = [CHUNK + 1, 3 * CHUNK + 1, 37 * CHUNK + 5, 1 << 20];
        for window in [1, 3, 32] {
            every_operator(Some(window), &lengths, 1);
        }
    }

    /// Several lines along axis 0: the tickets run over lines and no chunk looks into the line before it.
    #[test]
    fn integer_scans_of_several_lines_look_back() {
        every_operator(None, &[CHUNK + 1, 3 * CHUNK, 5000, 9 * CHUNK + 1], 5);
        every_operator(Some(2), &[CHUNK + 1, 3 * CHUNK, 5000], 7);
    }

    /// Float sums keep the fixed grouping of the two passes whatever the flag says (a look-back
    /// would group them by the timing of the chunks): bit for bit the same as without it, and
    /// the same on every run.
    #[test]
    fn float_sums_are_bit_identical_with_and_without_the_look_back() {
        let (look, two) = (cpu(true), cpu(false));
        for (k, &n) in LENGTHS.iter().enumerate() {
            let data = floats(n, k as u64 + 1);
            let want = host_scan(&data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), n, |a, b| a + b);
            let mut running = 0.0f64;
            let magnitude: Vec<f64> = data
                .iter()
                .map(|&v| {
                    running += f64::from(v).abs();
                    running
                })
                .collect();
            let bits = |ctx: &Context| -> Vec<Vec<u32>> {
                let x = Array::from_slice_on(ctx, &data, n).unwrap();
                let mut results = Vec::new();
                for _ in 0..2 {
                    results.push(x.cumsum(0).to_vec().unwrap());
                    results.push(x.scan(0, |a, b| a + b).to_vec().unwrap());
                }
                results.into_iter().map(|got| got.into_iter().map(f32::to_bits).collect()).collect()
            };
            let (with, without) = (bits(&look), bits(&two));
            for (round, (a, b)) in with.iter().zip(&without).enumerate() {
                assert!(a == b, "n = {n}, result {round}: the sums differ from the two-pass ones at element {:?}", first_difference(a, b));
            }
            for (round, got) in without.iter().enumerate() {
                assert!(got == &without[round % 2], "n = {n}, result {round}: the sums differ between runs");
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    let tolerance = 4e-6 * magnitude[i] + 1e-5;
                    assert!((f64::from(f32::from_bits(g)) - w).abs() <= tolerance, "n = {n}: element {i} is {}, want {w} (tolerance {tolerance})", f32::from_bits(g));
                }
            }
        }
    }

    /// The minima and maxima of floats look back (they pick one of their operands, so the result
    /// does not depend on the grouping): exactly the host's and the two-pass ones, every run.
    #[test]
    fn float_extrema_look_back_exactly() {
        let (look, two) = (cpu(true), cpu(false));
        for (k, &n) in LENGTHS.iter().enumerate() {
            let data = floats(n, k as u64 + 1);
            let (min, max) = (host_scan(&data, n, f32::min), host_scan(&data, n, f32::max));
            for (name, ctx) in [("look-back", &look), ("two-pass", &two)] {
                let x = Array::from_slice_on(ctx, &data, n).unwrap();
                for round in 0..2 {
                    assert!(x.scan(0, |a, b| a.minimum(b)).to_vec().unwrap() == min, "{name}, n = {n}, round {round}: min");
                    assert!(x.scan(0, |a, b| a.maximum(b)).to_vec().unwrap() == max, "{name}, n = {n}, round {round}: max");
                }
            }
        }
    }

    /// The launches `scan` of `x` records, after everything before it went out.
    fn launches<T: Numeric>(ctx: &Context, x: &Array<T>, op: &ScanOp<'_, T>) -> u32 {
        ctx.flush().unwrap();
        let before = ctx.pending_dispatches();
        let y = x.scan(0, op);
        let launches = ctx.pending_dispatches() - before;
        y.to_vec().unwrap();
        launches
    }

    /// The scan of a line of several chunks is two launches (the zeroing of the status, then the
    /// scan) instead of the three of the two-pass scan, and no host work in between: for
    /// integers and the minima and maxima of floats, not for float sums. A line of one chunk of
    /// the look-back kernel (but several of the two-pass ones) is three either way.
    #[test]
    fn only_scans_independent_of_the_grouping_are_two_launches() {
        let n = 3 * CHUNK;
        let (look, two) = (cpu(true), cpu(false));
        let (unsigned, signed, float) = (words(n, 7), words(n, 8).into_iter().map(|w| w as i32).collect::<Vec<_>>(), floats(n, 9));
        for (ctx, sums, extrema) in [(&look, 2, 2), (&two, 3, 3)] {
            let (u, i, f) = (
                Array::from_slice_on(ctx, &unsigned, n).unwrap(),
                Array::from_slice_on(ctx, &signed, n).unwrap(),
                Array::from_slice_on(ctx, &float, n).unwrap(),
            );
            let lookback = ctx.tuning().lookback_scan;
            assert_eq!(launches(ctx, &u, &|x, y| x + y), sums, "u32 sum, look-back: {lookback}");
            assert_eq!(launches(ctx, &i, &|x, y| x + y), sums, "i32 sum, look-back: {lookback}");
            assert_eq!(launches(ctx, &u, &|x, y| x.maximum(y)), extrema, "u32 max, look-back: {lookback}");
            assert_eq!(launches(ctx, &f, &|x, y| x.minimum(y)), extrema, "f32 min, look-back: {lookback}");
            assert_eq!(launches(ctx, &f, &|x, y| x.maximum(y)), extrema, "f32 max, look-back: {lookback}");
            assert_eq!(launches(ctx, &f, &|x, y| x + y), 3, "f32 sum, look-back: {lookback}");
            assert_eq!(launches(ctx, &f, &|x, y| x * y), 3, "f32 product, look-back: {lookback}");
        }
        let short = Array::<u32>::from_slice_on(&look, &words(CHUNK, 7), CHUNK).unwrap();
        assert_eq!(launches(&look, &short, &|x, y| x + y), 3, "one chunk");
    }

    /// The sum of a pending operand is computed in the scan's kernels (fold and scan of a line of several chunks, the look-back,
    /// strided lines): the same bits as the scan of the evaluated operand, for integers and for floats whose grouping is fixed.
    #[test]
    fn a_pending_operand_scans_as_its_buffer_does() {
        for lookback in [true, false] {
            let ctx = cpu(lookback);
            for (k, &len) in LENGTHS.iter().enumerate() {
                let data = words(len, k as u64 + 11);
                let x = Array::<u32>::from_slice_on(&ctx, &data, len).unwrap();
                let got = (&x * 3u32 + 1u32).cumsum(0).to_vec().unwrap();
                let mut run = 0u32;
                let want: Vec<u32> = data
                    .iter()
                    .map(|&v| {
                        run = run.wrapping_add(v.wrapping_mul(3).wrapping_add(1));
                        run
                    })
                    .collect();
                assert!(got == want, "u32, length {len}, look-back {lookback}: element {:?} differs", first_difference(&got, &want));

                let data = floats(len, k as u64 + 1);
                let (pending, other) = (Array::from_slice_on(&ctx, &data, len).unwrap(), Array::from_slice_on(&ctx, &data, len).unwrap());
                let evaluated = &other * 0.5f32;
                evaluated.eval().unwrap();
                let (got, want) = ((&pending * 0.5f32).cumsum(0).to_vec().unwrap(), evaluated.cumsum(0).to_vec().unwrap());
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "f32, length {len}, look-back {lookback}");
            }
            // Strided lines: one invocation each, and blocks with a fold.
            for (n, lines) in [(40usize, 7usize), (3000, 5)] {
                let data = words(n * lines, 5);
                let x = Array::<u32>::from_slice_on(&ctx, &data, [lines, n]).unwrap();
                let got = (&x + 7u32).cumsum(1).to_vec().unwrap();
                let mut want = vec![0u32; n * lines];
                for l in 0..lines {
                    let mut run = 0u32;
                    for s in 0..n {
                        run = run.wrapping_add(data[l + lines * s].wrapping_add(7));
                        want[l + lines * s] = run;
                    }
                }
                assert!(got == want, "strided {n} x {lines}, look-back {lookback}");
            }
        }
    }

    /// A step that scans, traced once and replayed (`Array::iterate`): the zeroing of the status
    /// is among the replayed launches, so every step starts from zero.
    #[test]
    fn a_replayed_step_zeroes_the_status_every_time() {
        let n = 3 * CHUNK + 5;
        let data = floats(n, 9);
        let steps = 9;
        let mut want = data.clone();
        for _ in 0..steps {
            let before = want.clone();
            let mut max = f32::NEG_INFINITY;
            for (v, &u) in want.iter_mut().zip(&before) {
                max = max.max(u);
                *v = max * 0.5 + u * 0.25;
            }
        }
        for lookback in [true, false] {
            let ctx = cpu(lookback);
            let x = Array::from_slice_on(&ctx, &data, n).unwrap();
            let got = x.iterate(steps, |u| &(&u.scan(0, |a, b| a.maximum(b)) * 0.5f32) + &(u * 0.25f32)).to_vec().unwrap();
            assert!(got == want, "look-back: {lookback}: element {:?} differs", first_difference(&got, &want));
        }
    }
}
