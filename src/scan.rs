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
use crate::builtin::BinOp;
use crate::context::{Context, RawArg, Scratch};
use crate::error::{LumaError, Result};
use crate::gpu_type::ScalarType;
use crate::jit::{operator_source, ElemTy, Operator};
use crate::kernel::KernelCore;
use crate::ops::{fold_workgroups, Outcome, Rhs};
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
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
    if let Pass::Rows { fold, width } = pass {
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
            {store}
        }}
    }}
",
            store = if fold { "" } else { "tile[row + s] = acc;" }
        ));
        if fold {
            src.push_str("    if (t < cnt) { out[first + t] = acc; }\n}\n");
        } else {
            src.push_str(&format!(
                "    workgroupBarrier();
    for (var j = t; j < cells; j += {r}u) {{
        let r = j / p.n;
        out[base + j] = tile[r * ld + (j - r * p.n)];
    }}
}}
"
            ));
        }
        return src;
    }
    if pass == Pass::ChunkReduce {
        let _ = writeln!(src, "var<workgroup> part: array<{ty}, {w}>;");
        src.push_str(head);
        src.push_str(&format!(
            "    let s0 = c * {REDUCE_CHUNK}u;
    let s1 = min(p.n, s0 + {REDUCE_CHUNK}u);
    var acc: {ty};
    var s = s0 + t;
    if (s < s1) {{
        acc = a[base + s];
        s += {w}u;
        for (; s < s1; s += {w}u) {{ acc = op(acc, a[base + s]); }}
    }}
    part[t] = acc;
    let valid = min({w}u, s1 - s0);
    for (var d = {w}u / 2u; d > 0u; d >>= 1u) {{
        workgroupBarrier();
        if (t < d && t + d < valid) {{ part[t] = op(part[t], part[t + d]); }}
    }}
    if (t == 0u) {{ out[wid] = part[0]; }}
}}
"
        ));
        return src.replace("__W__", &w.to_string());
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
struct Lines<'a> {
    ctx: &'a Context,
    op: &'a Operator,
    scratch: Scratch<'a>,
    cores: Vec<(Pass, Arc<KernelCore>)>,
    ty: &'static str,
}

/// A buffer and the bytes bound of it.
#[derive(Clone, Copy)]
struct Bound<'b> {
    buffer: &'b Buffer,
    bytes: u64,
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

impl<'a> Lines<'a> {
    fn core(&mut self, pass: Pass) -> Result<Arc<KernelCore>> {
        if let Some((_, core)) = self.cores.iter().find(|(p, _)| *p == pass) {
            return Ok(Arc::clone(core));
        }
        let workgroup = self.ctx.tuning().elementwise_workgroup;
        let core = self.ctx.jit().stencil_kernel(self.ctx, source(pass, self.ty, self.op, workgroup), pass.label())?;
        self.cores.push((pass, Arc::clone(&core)));
        Ok(core)
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
        let (lines, nb) = (geometry[2], geometry[3]);
        if let Pass::Rows { .. } = pass {
            let groups = lines.div_ceil(nb);
            return self.ctx.launch(&core, fold_workgroups(self.ctx, groups)?, [groups, 1, 1], &args);
        }
        if pass.per_workgroup() {
            let groups = lines
                .checked_mul(nb)
                .ok_or(LumaError::InvalidShape("too many chunks".into()))?;
            return self.ctx.launch(&core, fold_workgroups(self.ctx, groups)?, [groups, 1, 1], &args);
        }
        let count = if pass == Pass::Seq { lines } else { lines * nb };
        let workgroups = fold_workgroups(self.ctx, count.div_ceil(core.workgroup[0]))?;
        self.ctx.launch(&core, workgroups, [count, 1, 1], &args)
    }

    fn temp(&mut self, elements: u64) -> Result<(Buffer, u64)> {
        let bytes = elements * 4;
        Ok((self.scratch.take(bytes)?, bytes))
    }

    /// The inclusive scan of every line of `input` into `out`.
    fn scan(&mut self, input: Bound<'_>, n: u32, stride: u32, lines: u32, out: Bound<'_>) -> Result<()> {
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

    /// The fold of every line of `input` into `out[0..lines]`; `op` is commutative.
    fn reduce(&mut self, input: Bound<'_>, n: u32, stride: u32, lines: u32, out: Bound<'_>) -> Result<()> {
        if stride == 1 && n <= SHORT_LINE {
            return self.launch(Pass::Rows { fold: true, width: rows_width(n) }, [n, 1, lines, rows_per_group(n), n], out, input, None);
        }
        if stride == 1 {
            let nb = n.div_ceil(REDUCE_CHUNK);
            let geometry = [n, 1, lines, nb, REDUCE_CHUNK];
            if nb == 1 {
                return self.launch(Pass::ChunkReduce, geometry, out, input, None);
            }
            let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
            let totals = Bound { buffer: &totals, bytes };
            self.launch(Pass::ChunkReduce, geometry, totals, input, None)?;
            return self.reduce(totals, nb, 1, lines, out);
        }
        let k = strided_block(n, lines).unwrap_or(n);
        let nb = n.div_ceil(k);
        let geometry = [n, stride, lines, nb, k];
        if nb == 1 {
            return self.launch(Pass::Fold, geometry, out, input, None);
        }
        let (totals, bytes) = self.temp(u64::from(lines) * u64::from(nb))?;
        let totals = Bound { buffer: &totals, bytes };
        self.launch(Pass::Fold, geometry, totals, input, None)?;
        self.reduce(totals, nb, lines, lines, out)
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

    /// Runs the scan (`reduce == false`, shape kept) or the reduction of `op` along `axis`.
    fn run_lines(
        &self,
        axis: usize,
        op: &ScanOp<'_, T>,
        reduce: bool,
    ) -> Outcome<Array<T>> {
        let ctx = self.context();
        let (pin, bytes) = self.ready()?;
        if ElemTy::of::<T>()?.bytes() != 4 {
            return Err(LumaError::Unsupported(format!("scans and reductions along an axis are not available for {} arrays", ElemTy::of::<T>()?.display())).into());
        }
        let shape = self.shape();
        let dims = shape.dims();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        let operator = self.trace_operator(op)?;
        let out_shape = if reduce {
            let mut d = dims;
            d[axis] = 1;
            Shape::from(d)
        } else {
            shape
        };
        let out = Array::<T>::alloc(ctx, out_shape)?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let n = dims[axis];
        let stride: usize = dims[..axis].iter().product();
        let lines = self.len() / n;
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: dims }));
        let (n, stride, lines) = (to_u32(n)?, to_u32(stride)?, to_u32(lines)?);
        let mut run = Lines {
            ctx,
            op: &operator,
            scratch: Scratch::new(ctx),
            cores: Vec::new(),
            ty: T::wgsl_type(),
        };
        let input = Bound { buffer: pin.buffer(), bytes };
        let target = Bound { buffer: out_pin.buffer(), bytes: out.byte_len() };
        if reduce {
            run.reduce(input, n, stride, lines, target)?;
        } else {
            run.scan(input, n, stride, lines, target)?;
        }
        drop(run);
        crate::jit::check_finite(ctx, &out_storage, ElemTy::of::<T>()?)?;
        Ok(out)
    }

    #[track_caller]
    fn lines(
        &self,
        axis: usize,
        op: &ScanOp<'_, T>,
        reduce: bool,
        name: &'static str,
        location: &'static Location<'static>,
    ) -> Array<T> {
        let result = self.run_lines(axis, op, reduce);
        let mut shape = self.shape().dims();
        if reduce && axis < 4 {
            shape[axis] = 1;
        }
        Array::resolve(self.context(), Shape::from(shape), result, name, location)
    }

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
        self.lines(axis, &op, false, "scan", Location::caller())
    }

    /// The running sum along `axis` (a [`scan`](Array::scan) with `+`).
    #[track_caller]
    pub fn cumsum(&self, axis: usize) -> Array<T> {
        self.lines(axis, &add, false, "cumsum", Location::caller())
    }

    /// The sum along `axis`; the result has extent 1 along it.
    #[track_caller]
    pub fn sum_axis(&self, axis: usize) -> Array<T> {
        self.lines(axis, &add, true, "sum_axis", Location::caller())
    }

    /// The minimum along `axis`; the result has extent 1 along it.
    #[track_caller]
    pub fn min_axis(&self, axis: usize) -> Array<T> {
        self.lines(axis, &|x: &Array<T>, y: &Array<T>| x.binary(Rhs::Array(y), BinOp::Min, false, "minimum"), true, "min_axis", Location::caller())
    }

    /// The maximum along `axis`; the result has extent 1 along it.
    #[track_caller]
    pub fn max_axis(&self, axis: usize) -> Array<T> {
        self.lines(axis, &|x: &Array<T>, y: &Array<T>| x.binary(Rhs::Array(y), BinOp::Max, false, "maximum"), true, "max_axis", Location::caller())
    }
}

fn add<T: ScalarType>(x: &Array<T>, y: &Array<T>) -> Array<T> {
    x.binary(Rhs::Array(y), BinOp::Add, false, "add")
}
