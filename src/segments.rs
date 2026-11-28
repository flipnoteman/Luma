//! Segmented reductions: the sum, minimum or maximum of every contiguous segment of an array,
//! the segments given by their offsets (a CSR row pointer, the boundaries of a sorted group-by).
//!
//! The work is split by entries, not by segments, so that every invocation has the same share
//! whatever the lengths of the segments are (a power-law matrix has rows of one entry next to
//! rows of a hundred thousand) and no atomics are needed:
//!
//! 1. **Tiles.** A workgroup of [`WORKGROUP`] invocations takes a tile of [`TILE`] consecutive
//!    entries. It computes them (the pending expression the reduction was called on is fused: the
//!    entries are the values of `luma_eval`, never stored) with coalesced loads into workgroup
//!    memory, and finds the segments that start in the tile by a binary search of the offsets and
//!    a walk over the offsets that follow: the first entry of every non-empty segment is marked
//!    as a head, every empty segment is written at once. Every invocation then reduces its run of
//!    [`PER_THREAD`] consecutive entries, cutting it at the heads: a segment that starts and ends
//!    inside the run is final and written; the parts of a segment that continue into the next
//!    invocation, and the part that came from the previous one, are combined across the
//!    invocations by a segmented scan in workgroup memory, which gives the head of every segment
//!    its total. A segment that ends beyond the tile (the last one that starts in it, at most one
//!    per tile) is not final: its part is kept in the scratch buffer, together with the part of
//!    the tile's first segment that began in an earlier tile (`carry`).
//! 2. **Carries.** Only when there are several tiles: one invocation per tile whose last segment
//!    continues adds the carries of the following tiles that the segment covers, in order, to its
//!    own part and writes the segment.
//!
//! Every segment is written by exactly one invocation, and the combination order depends only
//! on the offsets: the results are the same on every run (`f32` sums are rounded, as in any
//! regrouping; sums of values that are exact in `f32` in any order are exact).

use std::fmt::Write;
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{ReduceElem, ReduceList, ReduceOp};
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::gpu_type::ElemLayout;
use crate::jit::{self, ElemTy};
use crate::ops::{fold_workgroups, len_u32, Outcome};
use crate::scan::{AxisPass, Pass};
use crate::scatter::AtomicElem;
use crate::shape::Shape;

/// Invocations of a workgroup of either pass.
pub(crate) const WORKGROUP: u32 = 128;
/// Consecutive entries one invocation of the tile pass reduces.
const PER_THREAD: u32 = 8;
/// Entries of a tile.
pub(crate) const TILE: u32 = WORKGROUP * PER_THREAD;
/// Words of scratch per tile: the carry in, the part of the last segment, the row of that segment.
const SCRATCH_WORDS: u32 = 3;
/// Words of workgroup memory per tile slot with the padding that keeps the invocations of a warp
/// on different banks (one word of padding per [`PER_THREAD`]).
const PADDED: u32 = TILE + TILE / PER_THREAD;

/// What a WGSL element type needs: its bits as a word and back, and the identity of an op.
struct Elem {
    ty: &'static str,
    to_bits: &'static str,
    from_bits: &'static str,
    min_identity: &'static str,
    max_identity: &'static str,
    zero: &'static str,
}

fn elem(ty: &str) -> Elem {
    match ty {
        "f32" => Elem {
            ty: "f32",
            to_bits: "bitcast<u32>(v)",
            from_bits: "bitcast<f32>(w)",
            min_identity: "bitcast<f32>(0x7f800000u)",
            max_identity: "bitcast<f32>(0xff800000u)",
            zero: "0.0",
        },
        "i32" => Elem {
            ty: "i32",
            to_bits: "bitcast<u32>(v)",
            from_bits: "bitcast<i32>(w)",
            min_identity: "2147483647i",
            max_identity: "(-2147483647i - 1i)",
            zero: "0i",
        },
        _ => Elem { ty: "u32", to_bits: "v", from_bits: "w", min_identity: "0xffffffffu", max_identity: "0u", zero: "0u" },
    }
}

/// The helper functions both passes share: `c` combines two values, `ident` is the value of an
/// empty segment, `tobits` and `frombits` store a value in the scratch words.
fn helpers(ty: &str, op: ReduceOp) -> String {
    let e = elem(ty);
    let identity = match op {
        ReduceOp::Min => e.min_identity,
        ReduceOp::Max => e.max_identity,
        _ => e.zero,
    };
    let ty = e.ty;
    format!(
        "fn c(u: {ty}, v: {ty}) -> {ty} {{ return {}; }}
fn ident() -> {ty} {{ return {identity}; }}
fn tobits(v: {ty}) -> u32 {{ return {}; }}
fn frombits(w: u32) -> {ty} {{ return {}; }}
",
        ReduceElem::Plain(ty).combine(op),
        e.to_bits,
        e.from_bits,
    )
}

/// The `Params` of both passes (see [`crate::scan::AXIS_PARAMS`]): `n` entries, `lines` segments,
/// `nb` tiles.
const PARAMS: &str = "struct Params { n: u32, stride: u32, lines: u32, nb: u32, k: u32, }\n";

/// The bindings of the tile pass's outputs (the one that has them declared by the caller:
/// [`AxisPass::output_bindings`]).
pub(crate) fn tile_outputs(ty: &str) -> String {
    format!(
        "@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;\n@group(1) @binding(1) var<storage, read_write> sc: array<u32>;\n"
    )
}

/// The helpers and entry point of the tile pass over the entries `{load}(q)`. Needs `Params` as
/// `p`, the buffers `out` and `sc` ([`tile_outputs`]) and the offsets `offs`.
pub(crate) fn tile_body(ty: &str, op: ReduceOp, load: &str) -> String {
    let mut src = helpers(ty, op);
    let (w, tile, per, padded) = (WORKGROUP, TILE, PER_THREAD, PADDED);
    let words = SCRATCH_WORDS;
    let _ = write!(
        src,
        "fn pad(j: u32) -> u32 {{ return j + (j >> 3u); }}
var<workgroup> prod: array<{ty}, {padded}>;
var<workgroup> hrow: array<u32, {padded}>;
var<workgroup> sv: array<{ty}, {w}>;
var<workgroup> sf: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let tile = wg.x + wg.y * nwg.x;
    if (tile >= p.nb) {{ return; }}
    let e0 = tile * {tile}u;
    let e1 = min(e0 + {tile}u, p.n);
    let last = tile + 1u == p.nb;
"
    );
    // The entries, coalesced: invocation `t` takes `t + j * W` of the tile.
    for j in 0..per {
        let _ = writeln!(src, "    var x{j} = ident();\n    if (e0 + t + {}u < e1) {{ x{j} = {load}(e0 + t + {}u); }}", j * w, j * w);
    }
    // The first segment that starts at or after the tile's first entry.
    src.push_str(
        "    var lo = 0u;
    var hi = p.lines + 1u;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (offs[mid] < e0) { lo = mid + 1u; } else { hi = mid; }
    }
",
    );
    for j in 0..per {
        let _ = writeln!(src, "    prod[pad(t + {}u)] = x{j};\n    hrow[pad(t + {}u)] = 0u;", j * w, j * w);
    }
    let _ = write!(
        src,
        "    workgroupBarrier();
    for (var r = lo + t; r < p.lines; r += {w}u) {{
        let s = offs[r];
        if (!last && s >= e1) {{ break; }}
        if (offs[r + 1u] > s) {{
            if (s < e1) {{ hrow[pad(s - e0)] = r + 1u; }}
        }} else {{
            out[r] = ident();
        }}
    }}
    workgroupBarrier();
    let c0 = t * {per}u;
    var run = ident();
    var runrow = 0u;
    var has = false;
    var lead = ident();
"
    );
    for i in 0..per {
        let _ = writeln!(
            src,
            "    {{
        let hr = hrow[pad(c0 + {i}u)];
        let x = prod[pad(c0 + {i}u)];
        if (hr != 0u) {{
            if (has) {{ out[runrow - 1u] = run; }} else {{ lead = run; }}
            has = true;
            run = x;
            runrow = hr;
        }} else {{
            run = c(run, x);
        }}
    }}"
        );
    }
    let _ = write!(
        src,
        "    if (!has) {{ lead = run; }}
    sv[t] = lead;
    sf[t] = select(0u, 1u, has);
    for (var d = 1u; d < {w}u; d = d << 1u) {{
        workgroupBarrier();
        var a = sv[t];
        var fl = sf[t];
        if (fl == 0u && t + d < {w}u) {{
            a = c(a, sv[t + d]);
            fl = sf[t + d];
        }}
        workgroupBarrier();
        sv[t] = a;
        sf[t] = fl;
    }}
    workgroupBarrier();
    if (t == 0u) {{
        sc[tile * {words}u] = tobits(sv[0]);
        if (sf[0] == 0u) {{ sc[tile * {words}u + 2u] = 0u; }}
    }}
    if (has) {{
        let nx = min(t + 1u, {w}u - 1u);
        let more = t + 1u < {w}u;
        let total = c(run, select(ident(), sv[nx], more));
        let row = runrow - 1u;
        if (!more || sf[nx] == 0u) {{
            if (offs[row + 1u] > e1) {{
                sc[tile * {words}u + 1u] = tobits(total);
                sc[tile * {words}u + 2u] = runrow;
            }} else {{
                out[row] = total;
                sc[tile * {words}u + 2u] = 0u;
            }}
        }} else {{
            out[row] = total;
        }}
    }}
}}
"
    );
    src
}

/// The tile pass over a buffer `a`.
fn tile_source(ty: &str, op: ReduceOp) -> String {
    let mut src = String::from(PARAMS);
    src.push_str(&tile_outputs(ty));
    let _ = writeln!(src, "@group(1) @binding(2) var<storage, read> a: array<{ty}>;");
    src.push_str("@group(1) @binding(3) var<storage, read> offs: array<u32>;\n@group(1) @binding(4) var<uniform> p: Params;\n");
    let _ = writeln!(src, "fn ld(q: u32) -> {ty} {{ return a[q]; }}");
    src.push_str(&tile_body(ty, op, "ld"));
    src
}

/// The carry pass: one invocation per tile whose last segment continues.
fn carry_source(ty: &str, op: ReduceOp) -> String {
    let mut src = String::from(PARAMS);
    let words = SCRATCH_WORDS;
    let (w, tile) = (WORKGROUP, TILE);
    let _ = writeln!(src, "@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;");
    src.push_str("@group(1) @binding(1) var<storage, read> sc: array<u32>;\n@group(1) @binding(2) var<storage, read> offs: array<u32>;\n@group(1) @binding(3) var<uniform> p: Params;\n");
    src.push_str(&helpers(ty, op));
    let _ = write!(
        src,
        "@compute @workgroup_size({w})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let k = gid.x + gid.y * nwg.x * {w}u;
    if (k >= p.nb) {{ return; }}
    let crossing = sc[k * {words}u + 2u];
    if (crossing == 0u) {{ return; }}
    let row = crossing - 1u;
    let end = offs[row + 1u];
    let hi = min((end - 1u) / {tile}u + 1u, p.nb);
    var a0 = ident();
    var a1 = ident();
    var a2 = ident();
    var a3 = ident();
    var j = k + 1u;
    for (; j + 3u < hi; j += 4u) {{
        a0 = c(a0, frombits(sc[j * {words}u]));
        a1 = c(a1, frombits(sc[(j + 1u) * {words}u]));
        a2 = c(a2, frombits(sc[(j + 2u) * {words}u]));
        a3 = c(a3, frombits(sc[(j + 3u) * {words}u]));
    }}
    for (; j < hi; j += 1u) {{
        a0 = c(a0, frombits(sc[j * {words}u]));
    }}
    out[row] = c(frombits(sc[k * {words}u + 1u]), c(c(a0, a1), c(a2, a3)));
}}
"
    );
    src
}

impl<T: AtomicElem> Array<T> {
    /// Runs the segmented reduction `op`: segment `i` is the elements `offsets[i]..offsets[i + 1]`
    /// of this array (in flat order). A pending operand is computed inside the first pass.
    fn run_segments(&self, offsets: &Array<u32>, op: ReduceOp) -> Outcome<Array<T>> {
        let ctx = self.context();
        let ty = ElemTy::of::<T>()?;
        if ty.layout() != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported(format!("segmented reductions are not available for {} arrays", ty.display())).into());
        }
        if offsets.len() < 2 {
            return Err(LumaError::InvalidShape(format!("{} offsets: a segmented reduction needs the offsets of at least one segment (segments + 1)", offsets.len())).into());
        }
        let (n, rows) = (len_u32(self.len())?, len_u32(offsets.len() - 1)?);
        let wgsl = T::wgsl_type();
        let out = Array::<T>::alloc(ctx, Shape::from(rows as usize))?;
        let storage = Arc::clone(out.storage()?);
        let out_pin = storage.pin_single()?;
        let (offs_pin, offs_bytes) = offsets.ready()?;
        let tiles = n.div_ceil(TILE).max(1);
        let mut scratch = Scratch::new(ctx);
        let sc_bytes = u64::from(tiles) * u64::from(SCRATCH_WORDS) * 4;
        let sc = scratch.take(sc_bytes)?;
        let geometry = [n, 1, rows, tiles, 0];
        let writes = [RawArg::write(out_pin.buffer(), out.byte_len()), RawArg::write(&sc, sc_bytes)];
        let reads = [RawArg::read(offs_pin.buffer(), offs_bytes)];
        let pass = AxisPass { pass: Pass::Segments, ops: ReduceList::new(&[op]), partial: false, split: false, line_major: false, scan: false, epi: None };
        let groups = fold_workgroups(ctx, tiles)?;
        let fused = match self.pending() {
            Some(node) => jit::axis_reduce(&mut crate::jit::structured::emit::Emit::now(ctx), node, pass, geometry, groups, [tiles, 1, 1], &writes, &reads, &[])?,
            None => false,
        };
        if !fused {
            let evaluated;
            let (buffer, bytes) = match self.pending() {
                Some(node) => {
                    let storage = node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?;
                    evaluated = storage.pin_single()?;
                    (evaluated.buffer(), storage.byte_len())
                }
                None => {
                    let (pin, bytes) = self.ready()?;
                    evaluated = pin;
                    (evaluated.buffer(), bytes)
                }
            };
            let core = ctx.jit().stencil_kernel(ctx, tile_source(wgsl, op), pass.label())?;
            let args = [writes[0], writes[1], RawArg::read(buffer, bytes), reads[0], RawArg::Uniform(bytemuck::cast_slice(&geometry))];
            ctx.launch(&core, groups, [tiles, 1, 1], &args)?;
        }
        if tiles > 1 {
            let core = ctx.jit().stencil_kernel(ctx, carry_source(wgsl, op), "SegmentsCarry")?;
            let groups = fold_workgroups(ctx, tiles.div_ceil(WORKGROUP))?;
            let args = [writes[0], RawArg::read(&sc, sc_bytes), reads[0], RawArg::Uniform(bytemuck::cast_slice(&geometry))];
            ctx.launch(&core, groups, [tiles, 1, 1], &args)?;
        }
        drop(scratch);
        jit::check_finite(ctx, &storage, ty)?;
        Ok(out)
    }

    fn segments(&self, offsets: &Array<u32>, op: ReduceOp, name: &'static str, location: &'static Location<'static>) -> Array<T> {
        let rows = offsets.len().saturating_sub(1).max(1);
        Array::resolve(self.context(), Shape::from(rows), self.run_segments(offsets, op), name, location)
    }

    /// The sum of every segment of this array: element `i` of the result adds the elements
    /// `offsets[i]..offsets[i + 1]` (in flat order), so the result has `offsets.len() - 1`
    /// elements, and a segment without elements sums to 0. The offsets are those of a CSR row
    /// pointer: non-decreasing, the first 0 and the last the length of this array (otherwise the
    /// result is unspecified, but no memory outside the arrays is read or written). Fewer than
    /// two offsets poison the result with [`LumaError::InvalidShape`].
    ///
    /// A pending operand is not evaluated first: its expression is computed inside the kernel,
    /// so a sparse matrix-vector product is one expression and two launches,
    /// `(&val * x.take(&col)).sum_segments(&ptr)`. The entries are split evenly over the
    /// workgroups whatever the segment lengths are, so a few very long segments do not leave
    /// the rest of the device idle, and no atomics are used: the result is the same on every
    /// run (`f32` sums round as any regrouping of the additions does; sums of values that are
    /// exact in `f32` in any order, such as small multiples of a power of two, are exact).
    /// Integer sums wrap.
    #[track_caller]
    pub fn sum_segments(&self, offsets: &Array<u32>) -> Array<T> {
        self.segments(offsets, ReduceOp::Sum, "sum_segments", Location::caller())
    }

    /// The minimum of every segment, as [`Array::sum_segments`]; a segment without elements gives the
    /// largest value of the type (`f32`: `+inf`), as [`Array::group_min`] does.
    #[track_caller]
    pub fn min_segments(&self, offsets: &Array<u32>) -> Array<T> {
        self.segments(offsets, ReduceOp::Min, "min_segments", Location::caller())
    }

    /// The maximum of every segment, as [`Array::sum_segments`]; a segment without elements gives the
    /// smallest value of the type (`f32`: `-inf`), as [`Array::group_max`] does.
    #[track_caller]
    pub fn max_segments(&self, offsets: &Array<u32>) -> Array<T> {
        self.segments(offsets, ReduceOp::Max, "max_segments", Location::caller())
    }
}
