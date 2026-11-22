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
//! On the CUDA backend (decoupled look-back needs forward progress across workgroups, see
//! `Tuning::lookback_scan`, and subgroups of 32) the three kernels and the scan become two
//! launches and one readback, `wide_partition_source` and `fused_source`: a warp finds each
//! boundary split by a 32-way search (a handful of dependent loads instead of twenty), and one
//! kernel merges, counts, finds its tile's first output slot by looking back over the tiles
//! before it (Merrill and Garland 2016, as `scan.rs` and `sort.rs`; Thrust's `set_operations.h`
//! has the same shape) and writes the kept items coalesced through workgroup memory. The size of
//! the result is the inclusive count of the last tile.
//!
//! The merged order is by key, ties taking the item of `a` first, so equal items are neighbours
//! with those of `a` before those of `b`. A union keeps the first of every run of equal items; an
//! intersection keeps the item of `b` that follows an item of `a` equal to it (once per common
//! value, whatever the duplicates inside `a` and `b`). Keys compare as the sort orders them with
//! `-0.0` equal to `+0.0` (all NaNs are one value already); every zero is written as `+0.0`.

use std::fmt::Write;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::context::RawArg;
use crate::error::LumaError;
use crate::gpu_type::ElemLayout;
use crate::ops::{fold_workgroups, Outcome};
use crate::scan::{add_operator, Bound, Lines};
use crate::shape::Shape;
use crate::sort::{KeyTransform, SortKey};

const WORKGROUP: u32 = 256;
const PER_THREAD: u32 = 8;
/// The fused kernel: invocations of a workgroup, merged items an invocation walks (a tile is
/// `WIDE_WORKGROUP * WIDE_PER_THREAD` = 6144 items; 38 registers, three workgroups of 16 warps to
/// a multiprocessor), and the lanes that search a boundary together in the partition. Measured on an
/// RTX 3090 and an RTX 3080 Laptop (`afcompare --only sets` shapes): 256 invocations with 16 items
/// take 96 us (3090), 512 with 12 take 86-89; 1024 invocations do not launch (registers); two or
/// four interleaved walks per invocation, numbering the tiles by workgroup id and a 2-, 8- or
/// 32-lane search were all within noise or slower.
const WIDE_WORKGROUP: u32 = 512;
const WIDE_PER_THREAD: u32 = 12;
const PARTITION_FAN: u32 = 4;
/// Merged items of a tile.
const TILE: u32 = WORKGROUP * PER_THREAD;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Op {
    Union,
    Intersect,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// Words of a status: the count in bits 0..30, bit 30 an aggregate, bit 31 an inclusive prefix
/// (the whole word is stored at once, so a reader never sees a flag without its count; 0: the
/// tile has not published yet).
const COUNT_MASK: &str = "0x3fffffffu";

/// What the wide partition and the fused kernel share: the declarations (`bindings`, then the
/// uniform at `uniform`) and the key functions.
fn wide_prelude(bindings: &str, uniform: u32, transform: KeyTransform) -> String {
    let float = transform == KeyTransform::Float;
    let canon = if float { "select(k, 0x80000000u, k == 0x7fffffffu)" } else { "k" };
    let mut src = String::from("struct Params { na: u32, nb: u32, tiles: u32, }\n");
    src.push_str(bindings);
    let _ = writeln!(src, "@group(1) @binding({uniform}) var<uniform> p: Params;");
    let _ = write!(
        src,
        "fn key_in(b: u32) -> u32 {{\n    {}\n}}
fn key_out(k: u32) -> u32 {{\n    {}\n}}
fn canon(w: u32) -> u32 {{ let k = key_in(w); return {canon}; }}
fn ka_g(i: u32) -> u32 {{ return canon(a[i]); }}
fn kb_g(i: u32) -> u32 {{ return canon(b[i]); }}
",
        transform.forward(),
        transform.inverse()
    );
    src
}

/// Rounds of the `fan`-way search of `wide_partition_source` for ranges of at most `range` items:
/// a round leaves at most `ceil(n / fan) - 1` candidates of `n`.
fn search_rounds(range: u32, fan: u32) -> u32 {
    let (mut rounds, mut reach) = (1, u64::from(fan));
    while reach < u64::from(range) {
        rounds += 1;
        reach *= u64::from(fan);
    }
    rounds
}

/// The partition of the wide path: groups of `fan` lanes (`32 / fan` boundaries to a warp) find
/// the split `i` of the diagonal `d` of a boundary (the first `i` for which `a[i] <= b[d - 1 - i]`
/// fails) by narrowing the range `fan` probes at a time: lane `k` of a group tests
/// `lo + k * step`, the test is true up to some lane (the predicate is monotone), and the range
/// becomes the gap after the last true probe. Every lane runs every round (the probes are clamped
/// to valid items, not skipped), so the ballot is reached by the whole warp. Also clears the
/// statuses and the ticket of the fused kernel.
fn wide_partition_source(transform: KeyTransform, rounds: u32, tile: u32, fan: u32) -> String {
    let mut src = wide_prelude(
        "@group(1) @binding(0) var<storage, read_write> split: array<u32>;\n@group(1) @binding(1) var<storage, read> a: array<u32>;\n@group(1) @binding(2) var<storage, read> b: array<u32>;\n@group(1) @binding(3) var<storage, read_write> st: array<atomic<u32>>;\n",
        4,
        transform,
    );
    let per_warp = 32 / fan;
    let boundaries = WORKGROUP / 32 * per_warp;
    let fan_mask = if fan == 32 { 0xffff_ffffu32 } else { (1 << fan) - 1 };
    let _ = write!(
        src,
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    let k = lane % {fan}u;
    let g = (wg.x + wg.y * nwg.x) * {boundaries}u + (t / 32u) * {per_warp}u + lane / {fan}u;
    let d = min(min(g, p.tiles) * {tile}u, p.na + p.nb);
    var lo = select(0u, d - p.nb, d > p.nb);
    var hi = min(d, p.na);
    for (var r = 0u; r < {rounds}u; r++) {{
        let step = (hi - lo + {fan_less}u) / {fan}u;
        let probe = lo + k * step;
        let ok = probe < hi;
        let i = select(0u, probe, ok);
        let j = min(d - 1u - i, p.nb - 1u);
        let le = ok && ka_g(i) <= kb_g(j);
        let c = countOneBits((subgroupBallot(le).x >> (lane - k)) & {fan_mask}u);
        let next = lo + c * step;
        lo = select(lo, next - step + 1u, c > 0u);
        hi = min(hi, next);
    }}
    if (k == 0u && g <= p.tiles) {{
        split[g] = lo;
        atomicStore(&st[g], 0u);
        if (g == 0u) {{ atomicStore(&st[p.tiles + 1u], 0u); }}
    }}
}}
",
        fan_less = fan - 1
    );
    src
}

/// The fused kernel: one workgroup merges, counts, finds where its items go and writes them.
///
/// 1. The workgroup takes a tile number from a ticket (the tiles are numbered in the order their
///    workgroups start, so every tile a look-back waits for is running), loads its `a` and `b`
///    ranges into workgroup memory (`a` first, `b` after it: `sm`) and every invocation finds its
///    diagonal's split by a binary search.
/// 2. Every invocation walks its `per` merged items without leaving registers (each step reads the
///    one item that replaces the one taken), keeps the key of each and a bit mask of the kept
///    ones, and counts them with one `countOneBits`. (`segs > 1` splits the walk into interleaved
///    segments; the shipped kernel has one, see `WIDE_PER_THREAD`.)
/// 3. A subgroup scan and the subgroup totals give every invocation its first slot in the
///    tile's output and the tile its count, published as an aggregate (flag 1); the first tile
///    publishes an inclusive prefix (flag 2).
/// 4. The kept keys go to workgroup memory (over the ranges, which nobody reads any more), in
///    order, while the workgroup looks back: every invocation reads the status of one of the `w`
///    tiles before this one (spinning until it is there), the nearest inclusive prefix is found
///    with a workgroup atomic minimum and the counts up to it are added with atomic adds. All
///    decisions are made on `workgroupUniformLoad`s, so the barriers are uniform. No subgroup
///    operation sits behind a spin (lanes that left a spin earlier are not waited for by an
///    `activemask`-based operation: the first version of this kernel summed partial warps).
/// 5. The inclusive prefix is published (the last tile's is the size of the result, which it
///    also stores in `size`) and the tile writes its keys, `w` consecutive words at a time,
///    through `key_out`.
fn fused_source(op: Op, transform: KeyTransform, per: u32, segs: u32, w: u32) -> String {
    let mut src = wide_prelude(
        "@group(1) @binding(0) var<storage, read_write> out: array<u32>;\n@group(1) @binding(1) var<storage, read> a: array<u32>;\n@group(1) @binding(2) var<storage, read> b: array<u32>;\n@group(1) @binding(3) var<storage, read> split: array<u32>;\n@group(1) @binding(4) var<storage, read_write> st: array<atomic<u32>>;\n@group(1) @binding(5) var<storage, read_write> size: array<u32>;\n",
        6,
        transform,
    );
    // `per` items an invocation, as `segs` segments of `seg` consecutive merged items, `w`
    // apart: the walks of an invocation do not depend on each other.
    let (seg, tile, warps) = (per / segs, w * per, w / 32);
    let words = per;
    let size = tile + seg * 2;
    let _ = write!(
        src,
        "var<workgroup> sm: array<u32, {size}>;
var<workgroup> wsum: array<u32, {wsum}>;
var<workgroup> ticket: u32;
var<workgroup> lb_near: atomic<u32>;
var<workgroup> lb_sum: atomic<u32>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    if (wg.x + wg.y * nwg.x >= p.tiles) {{ return; }}
    if (t == 0u) {{
        ticket = atomicAdd(&st[p.tiles + 1u], 1u);
        atomicStore(&lb_near, 0xffffffffu);
        atomicStore(&lb_sum, 0u);
    }}
    let tl = workgroupUniformLoad(&ticket);
    let a0 = split[tl];
    let a1 = split[tl + 1u];
    let d0 = tl * {tile}u;
    let b0 = d0 - a0;
    let b1 = min(d0 + {tile}u, p.na + p.nb) - a1;
    let la = a1 - a0;
    let lb = b1 - b0;
    let nl = la + lb;
",
        wsum = warps * segs,
    );
    // The tile's items in one run of workgroup memory (the keys of `a`, then those of `b`), the
    // loads issued before the stores.
    for m in 0..words {
        let _ = writeln!(
            src,
            "    let q{m} = {m}u * {w}u + t;\n    var v{m} = 0u;\n    if (q{m} < la) {{ v{m} = a[a0 + q{m}]; }} else if (q{m} < nl) {{ v{m} = b[b0 + q{m} - la]; }}"
        );
    }
    for m in 0..words {
        let _ = writeln!(src, "    sm[q{m}] = canon(v{m});");
    }
    src.push_str("    workgroupBarrier();\n");
    // Every segment starts at a diagonal `dl` of the tile; the splits are searched together.
    for h in 0..segs {
        let _ = write!(
            src,
            "    let dl{h} = min((t + {off}u) * {seg}u, nl);
    var lo{h} = 0u;
    if (dl{h} > lb) {{ lo{h} = dl{h} - lb; }}
    var hi{h} = min(dl{h}, la);
",
            off = h * w
        );
    }
    src.push_str("    loop {\n");
    for h in 0..segs {
        let _ = writeln!(
            src,
            "        let act{h} = lo{h} < hi{h};
        let mid{h} = (lo{h} + hi{h}) / 2u;
        let le{h} = sm[mid{h}] <= sm[min(la + dl{h} - 1u - mid{h}, {last}u)];
        lo{h} = select(lo{h}, mid{h} + 1u, act{h} && le{h});
        hi{h} = select(hi{h}, mid{h}, act{h} && !le{h});",
            last = size - 1
        );
    }
    let any = (0..segs).map(|h| format!("lo{h} < hi{h}")).collect::<Vec<_>>().join(" || ");
    let _ = writeln!(src, "        if (!({any})) {{ break; }}\n    }}");
    for h in 0..segs {
        let _ = write!(
            src,
            "    var ia{h} = lo{h};
    var ib{h} = dl{h} - lo{h};
    // The item before the first of the walk: the later of the two last items taken.
    var have{h} = false;
    var prev{h} = 0u;
    var prev_a{h} = false;
    if (ia{h} > 0u && ib{h} > 0u) {{
        have{h} = true;
        let x = sm[ia{h} - 1u];
        let y = sm[la + ib{h} - 1u];
        prev_a{h} = x > y;
        prev{h} = select(y, x, prev_a{h});
    }} else if (ia{h} > 0u) {{
        have{h} = true;
        prev_a{h} = true;
        prev{h} = sm[ia{h} - 1u];
    }} else if (ib{h} > 0u) {{
        have{h} = true;
        prev{h} = sm[la + ib{h} - 1u];
    }} else if (a0 > 0u || b0 > 0u) {{
        have{h} = true;
        if (a0 > 0u && b0 > 0u) {{
            prev_a{h} = ka_g(a0 - 1u) > kb_g(b0 - 1u);
            prev{h} = select(kb_g(b0 - 1u), ka_g(a0 - 1u), prev_a{h});
        }} else if (a0 > 0u) {{
            prev_a{h} = true;
            prev{h} = ka_g(a0 - 1u);
        }} else {{
            prev{h} = kb_g(b0 - 1u);
        }}
    }}
    let live{h} = min({seg}u, nl - dl{h});
    var ak{h} = sm[ia{h}];
    var bk{h} = sm[la + ib{h}];
    var m{h} = 0u;
"
        );
    }
    // The walks: step `s` takes the smaller head (the item of `a` on a tie), and the item that
    // replaces it becomes the head of its side. The segments' steps are interleaved.
    for s in 0..seg {
        for h in 0..segs {
            let n = format!("{h}_{s}");
            let _ = writeln!(src, "    let ta{n} = ia{h} < la && (ib{h} >= lb || ak{h} <= bk{h});\n    let k{n} = select(bk{h}, ak{h}, ta{n});");
            let cond = match (op, s) {
                (Op::Union, 0) => format!("(!have{h} || k{n} != prev{h})"),
                (Op::Union, s) => format!("k{n} != k{h}_{}", s - 1),
                (Op::Intersect, 0) => format!("(have{h} && k{n} == prev{h} && prev_a{h} && !ta{n})"),
                (Op::Intersect, s) => format!("(k{n} == k{h}_{p} && ta{h}_{p} && !ta{n})", p = s - 1),
            };
            let _ = writeln!(src, "    let keep{n} = {s}u < live{h} && {cond};\n    m{h} |= select(0u, {bit}u, keep{n});", bit = 1u32 << s);
            if s + 1 < seg {
                let _ = writeln!(
                    src,
                    "    ia{h} = select(ia{h}, ia{h} + 1u, ta{n});\n    ib{h} = select(ib{h} + 1u, ib{h}, ta{n});\n    let nv{n} = sm[select(la + ib{h}, ia{h}, ta{n})];\n    ak{h} = select(ak{h}, nv{n}, ta{n});\n    bk{h} = select(nv{n}, bk{h}, ta{n});"
                );
            }
        }
    }
    src.push_str("    let sg = t / 32u;\n");
    for h in 0..segs {
        let _ = write!(
            src,
            "    let mine{h} = countOneBits(m{h});
    let before{h} = subgroupExclusiveAdd(mine{h});
    let gtotal{h} = subgroupAdd(mine{h});
    if (lane == 0u) {{ wsum[{off}u + sg] = gtotal{h}; }}
",
            off = h * warps
        );
    }
    src.push_str("    workgroupBarrier();\n");
    for h in 0..segs {
        let total = (0..warps).map(|k| format!("wsum[{}u]", h * warps + k)).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "    let total{h} = {total};");
    }
    for h in 0..segs {
        let earlier = (0..h).map(|g| format!("total{g} + ")).collect::<String>();
        let before_group = (0..warps).map(|k| format!("select(0u, wsum[{}u], {k}u < sg)", h * warps + k)).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "    var pos{h} = {earlier}{before_group} + before{h};");
    }
    let tile_count = (0..segs).map(|h| format!("total{h}")).collect::<Vec<_>>().join(" + ");
    let _ = write!(
        src,
        "    let tile_count = {tile_count};
    if (t == 0u) {{
        if (tl == 0u) {{ atomicStore(&st[0], (2u << 30u) | tile_count); }} else {{ atomicStore(&st[tl], (1u << 30u) | tile_count); }}
    }}
"
    );
    for h in 0..segs {
        for s in 0..seg {
            let _ = writeln!(src, "    if ((m{h} & {bit}u) != 0u) {{ sm[pos{h}] = key_out(k{h}_{s}); pos{h} += 1u; }}", bit = 1u32 << s);
        }
    }
    let cm = COUNT_MASK;
    let _ = write!(
        src,
        "    var excl = 0u;
    if (tl > 0u) {{
        var back = 0u;
        loop {{
            let dist = back + t;
            var w = 0u;
            if (dist < tl) {{
                w = atomicLoad(&st[tl - 1u - dist]);
                while (w == 0u) {{ w = atomicLoad(&st[tl - 1u - dist]); }}
            }}
            if ((w >> 30u) == 2u) {{ atomicMin(&lb_near, dist); }}
            let near = workgroupUniformLoad(&lb_near);
            if (dist < tl && dist <= near) {{ atomicAdd(&lb_sum, w & {cm}); }}
            excl += workgroupUniformLoad(&lb_sum);
            if (near != 0xffffffffu) {{ break; }}
            back += {w}u;
            if (t == 0u) {{
                atomicStore(&lb_near, 0xffffffffu);
                atomicStore(&lb_sum, 0u);
            }}
            workgroupBarrier();
        }}
        if (t == 0u) {{ atomicStore(&st[tl], (2u << 30u) | (excl + tile_count)); }}
    }}
    if (t == 0u && tl + 1u == p.tiles) {{ size[0] = excl + tile_count; }}
    workgroupBarrier();
"
    );
    for m in 0..words {
        let _ = writeln!(src, "    if ({m}u * {w}u + t < tile_count) {{ out[excl + {m}u * {w}u + t] = sm[{m}u * {w}u + t]; }}");
    }
    src.push_str("}\n");
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
        // Status words keep counts in 30 bits; the fused kernel needs ordered memory across workgroups.
        if ctx.tuning().lookback_scan && ctx.tuning().subgroup32 && total < 1 << 30 {
            self.merge_wide(other, op, total)
        } else {
            self.merge_three_kernels(other, op, total)
        }
    }

    /// Partition and one fused kernel (see [`fused_source`]); the result is a window on the front
    /// of a buffer of the most items it can have.
    fn merge_wide(&self, other: &Array<T>, op: Op, total: u32) -> Outcome<Option<Array<T>>> {
        let ctx = self.context();
        let (na, nb) = (self.len(), other.len());
        let (per, wgs, fan) = (WIDE_PER_THREAD, WIDE_WORKGROUP, PARTITION_FAN);
        let tiles = total.div_ceil(wgs * per);
        let (a_pin, a_bytes) = self.ready()?;
        let (b_pin, b_bytes) = other.ready()?;
        let split = Array::<u32>::alloc(ctx, Shape::from(tiles as usize + 1))?;
        // One status word per tile and the ticket counter after them.
        let status = Array::<u32>::alloc(ctx, Shape::from(tiles as usize + 2))?;
        let size = Array::<u32>::alloc(ctx, Shape::from(1usize))?;
        let (split_pin, split_bytes) = split.ready()?;
        let (status_pin, status_bytes) = status.ready()?;
        let (size_pin, size_bytes) = size.ready()?;
        let most = match op {
            Op::Union => na + nb,
            Op::Intersect => na.min(nb),
        };
        let out = Array::<T>::alloc(ctx, Shape::from(most))?;
        let out_storage = Arc::clone(out.storage()?);
        let out_pin = out_storage.pin_single()?;
        let transform = T::TRANSFORM;
        let params = [na as u32, nb as u32, tiles];
        let uniform = RawArg::Uniform(bytemuck::bytes_of(&params));
        let (a_arg, b_arg) = (RawArg::read(a_pin.buffer(), a_bytes), RawArg::read(b_pin.buffer(), b_bytes));
        let rounds = search_rounds(na.min(nb) as u32, fan);
        let partition = ctx.jit().scan_kernel(ctx, format!("setpartition:{transform:?}:{rounds}:{}:{fan}", wgs * per), "SetPartition", || wide_partition_source(transform, rounds, wgs * per, fan))?;
        let boundary_groups = fold_workgroups(ctx, (tiles + 1).div_ceil(WORKGROUP / fan))?;
        ctx.launch(&partition, boundary_groups, [tiles + 1, 1, 1], &[RawArg::write(split_pin.buffer(), split_bytes), a_arg, b_arg, RawArg::write(status_pin.buffer(), status_bytes), uniform])?;
        let fused = ctx.jit().scan_kernel(ctx, format!("setfused:{op:?}:{transform:?}:{per}:{wgs}"), "SetFused", || fused_source(op, transform, per, 1, wgs))?;
        let tile_groups = fold_workgroups(ctx, tiles)?;
        ctx.launch(
            &fused,
            tile_groups,
            [tiles, 1, 1],
            &[RawArg::write(out_pin.buffer(), out.byte_len()), a_arg, b_arg, RawArg::read(split_pin.buffer(), split_bytes), RawArg::write(status_pin.buffer(), status_bytes), RawArg::write(size_pin.buffer(), size_bytes), uniform],
        )?;
        let kept = ctx.read::<u32>(size_pin.buffer(), 4).map_err(Failure::New)?[0] as usize;
        if kept == 0 {
            return Ok(None);
        }
        if kept == most {
            return Ok(Some(out));
        }
        let window = Array::<T>::from_storage(ctx, Shape::from(kept), Arc::new(Storage::window_of(&out_storage, 0, kept)));
        // A tiny result does not keep the big buffer: the copy is lazy and leaves it with the
        // first kernel that reads it.
        Ok(Some(if kept * 16 < most { window.copy() } else { window }))
    }

    /// The three kernels and the scan (see the module documentation), for backends without
    /// ordered memory across workgroups.
    fn merge_three_kernels(&self, other: &Array<T>, op: Op, total: u32) -> Outcome<Option<Array<T>>> {
        let ctx = self.context();
        let (na32, nb32) = (self.len() as u32, other.len() as u32);
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
        let kernel = |pass: Pass, op: Op| ctx.jit().scan_kernel(ctx, format!("setmerge:{pass:?}:{op:?}:{transform:?}"), "SetMerge", || source(pass, op, transform));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every generated kernel passes the WGSL front end and validator (subgroup operations, uniform
    /// control flow for the barriers) and lowers to Kernel IR, on any host.
    #[test]
    fn generated_kernels_validate() {
        let check = |name: &str, wgsl: String, w: u32| {
            luma_core::kir::parse_wgsl(&wgsl, "main", None, luma_core::Features::SUBGROUP, [1024; 3], 1024, 64, true).unwrap_or_else(|e| panic!("{name} ({w}): {e}\n{wgsl}"));
        };
        for transform in [KeyTransform::Identity, KeyTransform::Signed, KeyTransform::Float] {
            for op in [Op::Union, Op::Intersect] {
                for pass in [Pass::Partition, Pass::Count, Pass::Write] {
                    check(&format!("{pass:?} {op:?}"), source(pass, op, transform), 256);
                }
                for (per, segs, w) in [(8, 1, 256), (16, 1, 256), (16, 2, 256), (24, 3, 256), (32, 4, 256), (16, 1, 512), (8, 1, 1024)] {
                    check(&format!("fused {op:?} {per}x{segs} {w}"), fused_source(op, transform, per, segs, w), w);
                }
            }
            for fan in [2, 4, 8, 32] {
                check(&format!("partition fan {fan}"), wide_partition_source(transform, search_rounds(3_000_000, fan), 4096, fan), 256);
            }
        }
    }

    #[test]
    fn search_rounds_cover_the_range() {
        for fan in [2u32, 4, 32] {
            for range in [1u32, 2, 31, 32, 33, 1000, 1 << 20, (1 << 30) - 1] {
                // `n` candidates become at most `ceil(n / fan) - 1`.
                let (mut n, rounds) = (u64::from(range), search_rounds(range, fan));
                for _ in 0..rounds {
                    n = n.div_ceil(u64::from(fan)).saturating_sub(1);
                }
                assert_eq!(n, 0, "fan {fan} range {range} rounds {rounds}");
            }
        }
    }
}
