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
//!   Hillis-Steele ladder and every invocation rescans its run seeded with its prefix.
//! - *Strided lines*: invocations are numbered `block * lines + line`, so neighbours read
//!   neighbouring cells of the same row; each walks one block of the line sequentially.

use std::fmt::Write;
use std::panic::Location;
use std::sync::Arc;

use wgpu::Buffer;

use crate::array::{Array, Failure};
use crate::builtin::{BinOp, ReduceElem, ReduceList, ReduceOp, MAX_REDUCTIONS};
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::{ElemLayout, Numeric, ScalarType};
use crate::jit::{self, operator_source, ElemTy, Node, Operator};
use crate::kernel::KernelCore;
use crate::ops::{fold_workgroups, Outcome, Rhs};
use crate::reduce::Reduction;
use crate::shape::Shape;

// Measured on an RTX 3090 (`bench_pde --only scan` and `reduce/sum_axis`, 2^26 `f32`): every
// kernel below runs at about 800 GB/s, the bandwidth of the passes (a scan reads the input twice
// and writes it once), so the block shapes are not sensitive: workgroups of 64 to 256 and 4 to 16
// cells per invocation all give `scan/cumsum_*` 0.93-0.95 ms (before: 22-29 ms with one
// invocation per 1024 cells, 1.55 ms for the strided axis). `sum_axis` takes 0.33 ms (was 1.7).

/// Invocations of a workgroup of the contiguous-line kernels.
const WORKGROUP: u32 = 128;
/// Contiguous cells every invocation of a chunk scan folds (a chunk is `WORKGROUP * PER_THREAD`).
const PER_THREAD: u32 = 8;
/// Cells of a contiguous line one workgroup scans.
const SCAN_CHUNK: u32 = WORKGROUP * PER_THREAD;
/// Cells of a contiguous line one workgroup reduces (its invocations stride over them).
const REDUCE_CHUNK: u32 = 16384;
/// Contiguous lines up to this many cells are handled a group of lines per workgroup. One
/// workgroup per line (128 invocations for 33 cells) was 6 ms for 2^26 cells, a group of lines
/// 0.4 ms; from 256 cells on a workgroup per line is as fast.
const SHORT_LINE: u32 = 255;
/// Cells of workgroup memory a group of short lines is staged in (16 KiB, the smallest limit
/// WebGPU guarantees).
const ROWS_TILE: u32 = 4096;
/// Strided lines up to this many cells are walked by one invocation each.
const SHORT_STRIDED: u32 = 256;
/// A strided pass with this many lines or more has enough invocations without blocks.
const WIDE_LINES: u32 = 1 << 16;
/// Invocations a blocked strided pass aims for.
const TARGET_THREADS: u64 = 1 << 20;
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
    /// Contiguous lines, same workgroups: the fold of the chunk with the invocations striding
    /// over it, so the operator must be commutative as well as associative.
    ChunkReduce,
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
            Pass::ChunkReduce => "ReduceChunk",
        }
    }

    /// Whether the pass runs one workgroup (instead of one invocation) per unit of work.
    fn per_workgroup(self) -> bool {
        matches!(self, Pass::Rows { .. } | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkReduce)
    }

    fn seeded(self) -> bool {
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
        match self.pass {
            Pass::Fold => "ReduceFold",
            pass => pass.label(),
        }
    }

    /// How many buffers the pass writes (`@binding(0..)`).
    pub(crate) fn outputs(self) -> usize {
        if self.split { self.ops.as_slice().len() } else { 1 }
    }

    /// The declarations of the output buffers, of elements of `ty`.
    pub(crate) fn output_bindings(self, ty: &str) -> String {
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
    let cell = |name: &str, j: usize| if pass.partial { format!("{name}{j}") } else { name.to_owned() };
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
            let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {ROWS_TILE}>;");
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
        tile[row * pitch + (j - row * p.n)] = {load}({offset}base + j);
    }}
    workgroupBarrier();
    if (t < cnt) {{
        let row = t * pitch;
"
                );
                for &j in round {
                    let _ = writeln!(src, "        acc{j} = {};", mapped(j, "tile[row]".to_owned()));
                }
                src.push_str("        for (var s = 1u; s < p.n; s++) {\n");
                for &j in round {
                    let _ = writeln!(src, "            acc{j} = c{j}(acc{j}, {});", mapped(j, "tile[row + s]".to_owned()));
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
        Pass::Seq | Pass::Walk | Pass::ChunkFold | Pass::ChunkScan { .. } => unreachable!("scans are not reductions"),
    }
    src
}

fn source(pass: Pass, ty: &str, op: &Operator, workgroup: u32) -> String {
    let mut src = String::from("struct Params { n: u32, stride: u32, lines: u32, nb: u32, k: u32, ");
    for (k, scalar) in op.scalar_tys.iter().enumerate() {
        let _ = write!(src, "s{k}: {}, ", scalar.wgsl());
    }
    src.push_str("}\n");
    let _ = writeln!(src, "@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;");
    let _ = writeln!(src, "@group(1) @binding(1) var<storage, read> a: array<{ty}>;");
    let uniform = if pass.seeded() {
        let _ = writeln!(src, "@group(1) @binding(2) var<storage, read> pre: array<{ty}>;");
        3
    } else {
        2
    };
    let _ = writeln!(src, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    src.push_str(&op.source);
    src.push_str("fn line_base(l: u32) -> u32 { return (l % p.stride) + (l / p.stride) * p.stride * p.n; }\n");
    if pass.per_workgroup() {
        src.push_str(&chunk_source(pass, ty));
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
    var acc = a[base];
    out[base] = acc;
    for (var s = 1u; s < p.n; s++) {
        let q = base + s * p.stride;
        acc = op(acc, a[q]);
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
    var acc = a[base + s0 * p.stride];
    for (var s = s0 + 1u; s < s1; s++) { acc = op(acc, a[base + s * p.stride]); }
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
        let v = a[q];
        if (have) { acc = op(acc, v); } else { acc = v; have = true; }
        out[q] = acc;
    }
}
"
        }
        Pass::Rows { .. } | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkReduce => unreachable!("workgroup passes return above"),
    };
    src.push_str(&main.replace("__T__", ty));
    src
}

/// The entry point of the passes that run a workgroup per chunk of a contiguous line. `acc` and
/// `run` of invocations past the end of a short chunk are never read by invocations before them.
fn chunk_source(pass: Pass, ty: &str) -> String {
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
        tile[r * ld + (j - r * p.n)] = a[base + j];
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
    let k = PER_THREAD;
    let tile = SCAN_CHUNK + SCAN_CHUNK / 32 + 1;
    let _ = writeln!(src, "var<workgroup> tile: array<{ty}, {tile}>;");
    let _ = writeln!(src, "var<workgroup> part: array<{ty}, {w}>;");
    src.push_str("fn ix(j: u32) -> u32 { return j + (j >> 5u); }\n");
    src.push_str(head);
    src.push_str(&format!(
        "    let s0 = c * {SCAN_CHUNK}u;
    let len = min(p.n - s0, {SCAN_CHUNK}u);
    for (var j = t; j < len; j += {w}u) {{ tile[ix(j)] = a[base + s0 + j]; }}
    workgroupBarrier();
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
    var run: {ty};
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

/// The kernels of one operator and what they run on.
pub(crate) struct Lines<'a> {
    ctx: &'a Context,
    op: &'a Operator,
    scratch: Scratch<'a>,
    cores: Vec<(Pass, Arc<KernelCore>)>,
    reduce_cores: Vec<(AxisPass, Arc<KernelCore>)>,
    ty: &'static str,
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
enum Input<'b> {
    Buffer(Bound<'b>),
    /// A lazy expression: computed inside the pass when it can be (see [`jit::axis_reduce`]),
    /// else evaluated into a buffer first.
    Pending(&'b Arc<Node>),
}

/// The cells one invocation walks in a blocked pass over strided lines, or `None` when the lines
/// are short or numerous enough for one invocation each.
fn strided_block(n: u32, lines: u32) -> Option<u32> {
    if lines >= WIDE_LINES || n <= SHORT_STRIDED {
        return None;
    }
    let block = (u64::from(n) * u64::from(lines)).div_ceil(TARGET_THREADS).clamp(u64::from(MIN_BLOCK), u64::from(n));
    Some(block as u32)
}

/// The invocations of a workgroup of the short-line pass, which is the most lines it stages at
/// once: 64 lines of up to 63 cells fill the tile, longer lines leave room for fewer.
fn rows_width(n: u32) -> u32 {
    if n < 64 { 64 } else { 32 }
}

/// Lines of `n` cells a workgroup of the short-line pass stages at once (each row is padded to
/// an odd length, so the invocations read different banks).
fn rows_per_group(n: u32) -> u32 {
    (ROWS_TILE / (n | 1)).min(rows_width(n))
}

/// The operator `+` on `u32`, for [`Lines::words`].
pub(crate) fn add_operator(ctx: &Context) -> Outcome<Operator> {
    Array::<u32>::alloc(ctx, Shape::from(1usize))?.trace_operator(&add)
}

impl<'a> Lines<'a> {
    fn new(ctx: &'a Context, op: &'a Operator, ty: &'static str) -> Self {
        Lines { ctx, op, scratch: Scratch::new(ctx), cores: Vec::new(), reduce_cores: Vec::new(), ty }
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
        let core = self.ctx.jit().stencil_kernel(self.ctx, source(pass, self.ty, self.op, workgroup), pass.label())?;
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
            Pass::Rows { .. } => lines.div_ceil(nb),
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
            args.push(RawArg::read(pre.buffer, pre.bytes));
        }
        args.push(RawArg::Uniform(bytemuck::cast_slice(&words)));
        let (workgroups, size) = self.grid(pass, geometry)?;
        self.ctx.launch(&core, workgroups, size, &args)
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
        let pass = AxisPass { pass, ops, partial, split: outs.len() > 1, line_major: line_major(pass, geometry[3]) };
        let (workgroups, size) = self.grid(pass.pass, geometry)?;
        let writes: Vec<RawArg<'_>> = outs.iter().map(|out| RawArg::write(out.buffer, out.bytes)).collect();
        let evaluated;
        let input = match input {
            Input::Buffer(input) => input,
            Input::Pending(node) => {
                if jit::axis_reduce(self.ctx, node, pass, geometry, workgroups, size, &writes)? {
                    return Ok(());
                }
                let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
                evaluated = storage.pin_single()?;
                Bound { buffer: evaluated.buffer(), bytes: storage.byte_len() }
            }
        };
        let core = self.reduce_core(pass)?;
        let mut args = writes;
        args.push(RawArg::read(input.buffer, input.bytes));
        args.push(RawArg::Uniform(bytemuck::cast_slice(&geometry)));
        self.ctx.launch(&core, workgroups, size, &args)?;
        Ok(())
    }

    fn temp(&mut self, elements: u64) -> Result<(Buffer, u64)> {
        let bytes = elements * 4;
        Ok((self.scratch.take(bytes)?, bytes))
    }

    /// The inclusive scan of every line of `input` into `out`.
    pub(crate) fn scan(&mut self, input: Bound<'_>, n: u32, stride: u32, lines: u32, out: Bound<'_>) -> Result<()> {
        let seq = [n, stride, lines, 1, n];
        if stride == 1 {
            if n <= SHORT_LINE {
                return self.launch(Pass::Rows { fold: false, width: rows_width(n) }, [n, 1, lines, rows_per_group(n), n], out, input, None);
            }
            let nb = n.div_ceil(SCAN_CHUNK);
            let geometry = [n, 1, lines, nb, SCAN_CHUNK];
            if nb == 1 {
                return self.launch(Pass::ChunkScan { seeded: false }, geometry, out, input, None);
            }
            let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
            let totals = Bound { buffer: &totals, bytes };
            self.launch(Pass::ChunkFold, geometry, totals, input, None)?;
            let (prefix, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
            let prefix = Bound { buffer: &prefix, bytes };
            self.scan(totals, nb, 1, lines, prefix)?;
            return self.launch(Pass::ChunkScan { seeded: true }, geometry, out, input, Some(prefix));
        }
        let Some(k) = strided_block(n, lines) else {
            return self.launch(Pass::Seq, seq, out, input, None);
        };
        let nb = n.div_ceil(k);
        let geometry = [n, stride, lines, nb, k];
        let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
        let totals = Bound { buffer: &totals, bytes };
        self.launch(Pass::Fold, geometry, totals, input, None)?;
        let (prefix, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
        let prefix = Bound { buffer: &prefix, bytes };
        self.scan(totals, nb, lines, lines, prefix)?;
        self.launch(Pass::Walk, geometry, out, input, Some(prefix))
    }

    /// The reductions `ops` of every line of `input` into `outs` (see [`Lines::launch_reduce`]);
    /// `partial` if `input` holds the partial results of an earlier pass. The ops are commutative
    /// as well as associative. The first pass reads the expression behind a pending input
    /// directly; the passes over partial results read buffers.
    fn reduce(&mut self, input: Input<'_>, cells: Cells, ops: ReduceList, partial: bool, outs: &[Bound<'_>]) -> Outcome<()> {
        let Cells { n, stride, lines } = cells;
        if stride == 1 && n <= SHORT_LINE {
            return self.launch_reduce(Pass::Rows { fold: true, width: rows_width(n) }, ops, partial, [n, 1, lines, rows_per_group(n), n], outs, input);
        }
        let (pass, geometry, nb) = if stride == 1 {
            let nb = n.div_ceil(REDUCE_CHUNK);
            (Pass::ChunkReduce, [n, 1, lines, nb, REDUCE_CHUNK], nb)
        } else {
            let k = strided_block(n, lines).unwrap_or(n);
            let nb = n.div_ceil(k);
            (Pass::Fold, [n, stride, lines, nb, k], nb)
        };
        if nb == 1 {
            return self.launch_reduce(pass, ops, partial, geometry, outs, input);
        }
        let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb) * ops.as_slice().len() as u64)?;
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

/// The operator of a scan: an elementwise closure of two arguments.
type ScanOp<'a, T> = dyn Fn(&Array<T>, &Array<T>) -> Array<T> + 'a;

fn not_elementwise() -> Failure {
    LumaError::InvalidShape("the scan operator must be an elementwise expression of its two arguments".into()).into()
}

impl<T: ScalarType> Array<T> {
    /// `op` traced on two one-element placeholders as a WGSL function.
    fn trace_operator(&self, op: &ScanOp<'_, T>) -> Outcome<Operator> {
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

    /// Runs the scan of `op` along `axis`.
    fn run_scan(&self, axis: usize, op: &ScanOp<'_, T>) -> Outcome<Array<T>> {
        let ctx = self.context();
        let (pin, bytes) = self.ready()?;
        if ElemTy::of::<T>()?.layout() != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported(format!("scans and reductions along an axis are not available for {} arrays", ElemTy::of::<T>()?.display())).into());
        }
        let shape = self.shape();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        let operator = self.trace_operator(op)?;
        let out = Array::<T>::alloc(ctx, shape)?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let (n, stride, lines) = self.line_geometry(axis)?;
        let mut run = Lines::new(ctx, &operator, T::wgsl_type());
        let input = Bound { buffer: pin.buffer(), bytes };
        let target = Bound { buffer: out_pin.buffer(), bytes: out.byte_len() };
        run.scan(input, n, stride, lines, target)?;
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
        let operator = Operator { source: String::new(), scalar_tys: Vec::new(), scalars: Vec::new() };
        let mut run = Lines::new(ctx, &operator, T::wgsl_type());
        run.reduce(input, Cells { n, stride, lines }, ops, false, &targets)?;
        drop(run);
        for storage in &storages {
            crate::jit::check_finite(ctx, storage, ty)?;
        }
        Ok(outs)
    }

    #[track_caller]
    fn scan_axis(&self, axis: usize, op: &ScanOp<'_, T>, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let result = self.run_scan(axis, op);
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    /// `reduce_axis` for the caller `name` at `location`.
    fn reduce_lines<const N: usize>(&self, axis: usize, ops: [Reduction; N], name: &'static str, location: &'static Location<'static>) -> [Array<T>; N] {
        let mut dims = self.shape().dims();
        if axis < 4 {
            dims[axis] = 1;
        }
        let shape = Shape::from(dims);
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
    /// Contiguous lines are cut in chunks that a workgroup scans in workgroup memory (chunk
    /// totals, a scan of the totals, a second pass seeded with them), strided lines in blocks
    /// that one invocation walks, with neighbouring invocations on neighbouring lines, so every
    /// pass reads and writes memory in coalesced accesses; short lines are staged several to a
    /// workgroup. The arrays are evaluated first; arrays split over several buffers fail with
    /// [`LumaError::Chunked`]. Scalars captured by `op` are uniforms, so changing them does not
    /// recompile.
    #[track_caller]
    pub fn scan(&self, axis: usize, op: impl Fn(&Array<T>, &Array<T>) -> Array<T>) -> Array<T> {
        self.scan_axis(axis, &op, "scan", Location::caller())
    }

    /// The running sum along `axis` (a [`scan`](Array::scan) with `+`).
    #[track_caller]
    pub fn cumsum(&self, axis: usize) -> Array<T> {
        self.scan_axis(axis, &add, "cumsum", Location::caller())
    }

    /// The sum along `axis`; the result has extent 1 along it. A pending operand is not
    /// evaluated first: its expression is computed inside the first pass of the reduction.
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
    /// [`sum_axis`](Array::sum_axis), the arrays are evaluated when the call returns; another
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
    /// extent 1 along it. Two passes, so it is stable for data far from zero: the mean, then
    /// the mean of the squared deviations, in which `self - mean` is computed inside the
    /// reduction.
    #[track_caller]
    pub fn var_axis(&self, axis: usize) -> Array<f32> {
        let deviation = self - &self.mean_axis(axis);
        (&deviation * &deviation).mean_axis(axis)
    }

    /// The population standard deviation along `axis`: the square root of
    /// [`var_axis`](Array::var_axis).
    #[track_caller]
    pub fn std_axis(&self, axis: usize) -> Array<f32> {
        self.var_axis(axis).sqrt()
    }
}

fn add<T: ScalarType>(x: &Array<T>, y: &Array<T>) -> Array<T> {
    x.binary(Rhs::Array(y), BinOp::Add, false, "add")
}
