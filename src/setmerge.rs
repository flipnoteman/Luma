//! The merge behind [`Array::set_union`] and [`Array::set_intersect`]: two sorted arrays of
//! `u32`, `i32` or `f32` merged by the merge path (Green, McColl and Bader 2012), each workgroup
//! taking the items `2048 * tile ..` of the merged order, in three kernels and a scan:
//!
//! 1. *Partition*: the split `(i, j)` of every tile boundary: the first `d` merged items are the
//!    first `i` of `a` and the first `j = d - i` of `b`, found by a binary search along the
//!    diagonal `d`.
//! 2. *Count*: a workgroup loads its two ranges into workgroup memory, every invocation walks 8
//!    consecutive merged items (starting from the split of its own diagonal, found in workgroup
//!    memory) and counts the items the operation keeps; the tile's count is written.
//! 3. A scan of the counts (see `scan.rs`), read back for the size of the result.
//! 4. *Write*: the same walk, with a workgroup scan of the invocations' counts, writes the kept
//!    items to their places.
//!
//! The merged order is by key, ties taking the item of `a` first, so equal items are neighbours
//! with those of `a` before those of `b`. A union keeps the first of every run of equal items; an
//! intersection keeps the item of `b` that follows an item of `a` equal to it (once per common
//! value, whatever the duplicates inside `a` and `b`). Keys compare as the sort orders them with
//! `-0.0` equal to `+0.0` (all NaNs are one value already); every zero is written as `+0.0`.

use std::fmt::Write;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::context::RawArg;
use crate::error::LumaError;
use crate::gpu_type::ElemLayout;
use crate::ops::{fold_workgroups, Outcome};
use crate::scan::{add_operator, Bound, Lines};
use crate::shape::Shape;
use crate::sort::{KeyTransform, SortKey};

const WORKGROUP: u32 = 256;
const PER_THREAD: u32 = 8;
/// Merged items of a tile.
const TILE: u32 = WORKGROUP * PER_THREAD;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Op {
    Union,
    Intersect,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Partition,
    Count,
    Write,
}

fn source(pass: Pass, op: Op, transform: KeyTransform) -> String {
    let mut src = String::from("struct Params { na: u32, nb: u32, tiles: u32, }\n");
    let float = transform == KeyTransform::Float;
    let uniform = match pass {
        Pass::Partition => {
            src.push_str("@group(1) @binding(0) var<storage, read_write> split: array<u32>;\n@group(1) @binding(1) var<storage, read> a: array<u32>;\n@group(1) @binding(2) var<storage, read> b: array<u32>;\n");
            3
        }
        Pass::Count => {
            src.push_str("@group(1) @binding(0) var<storage, read_write> counts: array<u32>;\n@group(1) @binding(1) var<storage, read> a: array<u32>;\n@group(1) @binding(2) var<storage, read> b: array<u32>;\n@group(1) @binding(3) var<storage, read> split: array<u32>;\n");
            4
        }
        Pass::Write => {
            src.push_str("@group(1) @binding(0) var<storage, read_write> out: array<u32>;\n@group(1) @binding(1) var<storage, read> a: array<u32>;\n@group(1) @binding(2) var<storage, read> b: array<u32>;\n@group(1) @binding(3) var<storage, read> split: array<u32>;\n@group(1) @binding(4) var<storage, read> offs: array<u32>;\n");
            5
        }
    };
    let _ = writeln!(src, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    let canon = if float { "select(k, 0x80000000u, k == 0x7fffffffu)" } else { "k" };
    let zero_out = if float { "select(w, 0u, w == 0x80000000u)" } else { "w" };
    let _ = write!(
        src,
        "fn key_in(b: u32) -> u32 {{\n    {}\n}}
fn canon(w: u32) -> u32 {{ let k = key_in(w); return {canon}; }}
fn zero_out(w: u32) -> u32 {{ return {zero_out}; }}
fn ka_g(i: u32) -> u32 {{ return canon(a[i]); }}
fn kb_g(i: u32) -> u32 {{ return canon(b[i]); }}
",
        transform.forward()
    );
    let (w, per, tile) = (WORKGROUP, PER_THREAD, TILE);
    if pass == Pass::Partition {
        let _ = write!(
            src,
            "@compute @workgroup_size({w})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let g = gid.x + gid.y * nwg.x * {w}u;
    if (g > p.tiles) {{ return; }}
    let d = min(g * {tile}u, p.na + p.nb);
    var lo = 0u;
    if (d > p.nb) {{ lo = d - p.nb; }}
    var hi = min(d, p.na);
    while (lo < hi) {{
        let mid = (lo + hi) / 2u;
        if (ka_g(mid) <= kb_g(d - 1u - mid)) {{ lo = mid + 1u; }} else {{ hi = mid; }}
    }}
    split[g] = lo;
}}
"
        );
        return src;
    }
    let keep = match op {
        Op::Union => "!have || key != prev",
        Op::Intersect => "have && key == prev && prev_a && !from_a",
    };
    let store_count = match pass {
        Pass::Count => "    atomicAdd(&total, mine);\n    workgroupBarrier();\n    if (t == 0u) { counts[tl] = atomicLoad(&total); }\n",
        _ => "",
    };
    let _ = write!(
        src,
        "var<workgroup> ka: array<u32, {tile}>;
var<workgroup> kb: array<u32, {tile}>;
var<workgroup> total: atomic<u32>;
var<workgroup> scan_a: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let tl = wg.x + wg.y * nwg.x;
    if (tl >= p.tiles) {{ return; }}
    let a0 = split[tl];
    let a1 = split[tl + 1u];
    let d0 = tl * {tile}u;
    let b0 = d0 - a0;
    let b1 = min(d0 + {tile}u, p.na + p.nb) - a1;
    let la = a1 - a0;
    let lb = b1 - b0;
    if (t == 0u) {{ atomicStore(&total, 0u); }}
    for (var j = t; j < la; j += {w}u) {{ ka[j] = ka_g(a0 + j); }}
    for (var j = t; j < lb; j += {w}u) {{ kb[j] = kb_g(b0 + j); }}
    workgroupBarrier();
    // This invocation's items start at the diagonal `dl` of the tile.
    let dl = min(t * {per}u, la + lb);
    var lo = 0u;
    if (dl > lb) {{ lo = dl - lb; }}
    var hi = min(dl, la);
    while (lo < hi) {{
        let mid = (lo + hi) / 2u;
        if (ka[mid] <= kb[dl - 1u - mid]) {{ lo = mid + 1u; }} else {{ hi = mid; }}
    }}
    var ia = lo;
    var ib = dl - lo;
    // The item before the first of the walk: the later of the two last items taken.
    var have = false;
    var prev = 0u;
    var prev_a = false;
    if (ia > 0u && ib > 0u) {{
        have = true;
        prev_a = ka[ia - 1u] > kb[ib - 1u];
        prev = select(kb[ib - 1u], ka[ia - 1u], prev_a);
    }} else if (ia > 0u) {{
        have = true;
        prev_a = true;
        prev = ka[ia - 1u];
    }} else if (ib > 0u) {{
        have = true;
        prev = kb[ib - 1u];
    }} else if (a0 > 0u || b0 > 0u) {{
        have = true;
        if (a0 > 0u && b0 > 0u) {{
            prev_a = ka_g(a0 - 1u) > kb_g(b0 - 1u);
            prev = select(kb_g(b0 - 1u), ka_g(a0 - 1u), prev_a);
        }} else if (a0 > 0u) {{
            prev_a = true;
            prev = ka_g(a0 - 1u);
        }} else {{
            prev = kb_g(b0 - 1u);
        }}
    }}
    // Walk the items, remembering which are kept.
    var kept_mask = 0u;
    var mine = 0u;
    var walk_a = ia;
    var walk_b = ib;
    let steps = min(dl + {per}u, la + lb) - dl;
    for (var s = 0u; s < steps; s++) {{
        let from_a = walk_a < la && (walk_b >= lb || ka[walk_a] <= kb[walk_b]);
        let key = select(kb[min(walk_b, {tile}u - 1u)], ka[min(walk_a, {tile}u - 1u)], from_a);
        if ({keep}) {{
            kept_mask |= 1u << s;
            mine++;
        }}
        have = true;
        prev = key;
        prev_a = from_a;
        if (from_a) {{ walk_a++; }} else {{ walk_b++; }}
    }}
{store_count}",
    );
    if pass == Pass::Count {
        src.push_str("}\n");
        return src;
    }
    // Write: a workgroup scan of the counts gives the first slot of every invocation.
    let _ = write!(
        src,
        "    scan_a[t] = mine;
    for (var d = 1u; d < {w}u; d <<= 1u) {{
        workgroupBarrier();
        var x = scan_a[t];
        if (t >= d) {{ x += scan_a[t - d]; }}
        workgroupBarrier();
        scan_a[t] = x;
    }}
    workgroupBarrier();
    var slot = offs[tl] - scan_a[{last}u] + scan_a[t] - mine;
    ia = lo;
    ib = dl - lo;
    for (var s = 0u; s < steps; s++) {{
        let from_a = ia < la && (ib >= lb || ka[ia] <= kb[ib]);
        if (((kept_mask >> s) & 1u) != 0u) {{
            let word = select(b[b0 + ib], a[a0 + ia], from_a);
            out[slot] = zero_out(word);
            slot++;
        }}
        if (from_a) {{ ia++; }} else {{ ib++; }}
    }}
}}
",
        last = WORKGROUP - 1,
    );
    src
}

impl<T: SortKey> Array<T> {
    /// The union or intersection of two sorted flat arrays (see the module documentation), `None`
    /// if the result is empty. Reads the size back: synchronizes once.
    pub(crate) fn merge_sorted(&self, other: &Array<T>, op: Op) -> Outcome<Option<Array<T>>> {
        let ctx = self.context();
        if T::LAYOUT != ElemLayout::plain(4) {
            return Err(LumaError::Unsupported("set operations on 4-byte elements only".into()).into());
        }
        let (na, nb) = (self.len(), other.len());
        let too_large = || LumaError::DimensionTooLarge { shape: [na + nb, 1, 1, 1] };
        let total = u32::try_from(na + nb).map_err(|_| too_large())?;
        let (na32, nb32) = (na as u32, nb as u32);
        let tiles = total.div_ceil(TILE);
        let (a_pin, a_bytes) = self.ready()?;
        let (b_pin, b_bytes) = other.ready()?;
        let split = Array::<u32>::alloc(ctx, Shape::from(tiles as usize + 1))?;
        let counts = Array::<u32>::alloc(ctx, Shape::from(tiles as usize))?;
        let scanned = Array::<u32>::alloc(ctx, Shape::from(tiles as usize))?;
        let (split_pin, split_bytes) = split.ready()?;
        let (counts_pin, counts_bytes) = counts.ready()?;
        let (scanned_pin, scanned_bytes) = scanned.ready()?;
        let transform = T::TRANSFORM;
        let params = [na32, nb32, tiles];
        let uniform = RawArg::Uniform(bytemuck::bytes_of(&params));
        let kernel = |pass: Pass, op: Op| ctx.jit().stencil_kernel(ctx, source(pass, op, transform), "SetMerge");
        let (a_arg, b_arg) = (RawArg::read(a_pin.buffer(), a_bytes), RawArg::read(b_pin.buffer(), b_bytes));
        let part = kernel(Pass::Partition, op)?;
        let groups = fold_workgroups(ctx, (tiles + 1).div_ceil(WORKGROUP))?;
        ctx.launch(&part, groups, [tiles + 1, 1, 1], &[RawArg::write(split_pin.buffer(), split_bytes), a_arg, b_arg, uniform])?;
        let count = kernel(Pass::Count, op)?;
        let tile_groups = fold_workgroups(ctx, tiles)?;
        ctx.launch(&count, tile_groups, [tiles, 1, 1], &[RawArg::write(counts_pin.buffer(), counts_bytes), a_arg, b_arg, RawArg::read(split_pin.buffer(), split_bytes), uniform])?;
        let operator = add_operator(ctx)?;
        let mut lines = Lines::words(ctx, &operator);
        lines.scan(Bound { buffer: counts_pin.buffer(), bytes: counts_bytes }, tiles, 1, 1, Bound { buffer: scanned_pin.buffer(), bytes: scanned_bytes })?;
        drop(lines);
        let kept = scanned.slice(0, tiles as usize - 1..).item().map_err(Failure::New)? as usize;
        if kept == 0 {
            return Ok(None);
        }
        let out = Array::<T>::alloc(ctx, Shape::from(kept))?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let write = kernel(Pass::Write, op)?;
        ctx.launch(
            &write,
            tile_groups,
            [tiles, 1, 1],
            &[RawArg::write(out_pin.buffer(), out.byte_len()), a_arg, b_arg, RawArg::read(split_pin.buffer(), split_bytes), RawArg::read(scanned_pin.buffer(), scanned_bytes), uniform],
        )?;
        Ok(Some(out))
    }
}
