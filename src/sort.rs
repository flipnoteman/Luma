//! Sorting: `sort`, `argsort`, `sort_by_key` and `sort_axis` for `u32`, `i32` and `f32` arrays.
//!
//! An LSD radix sort on 8-bit digits, one pass per digit. Two ways of running the passes:
//!
//! *Single sweep* (CUDA: where the device orders memory across workgroups, `Tuning::lookback_scan`,
//! and ranks with ballots, see below; CUB's onesweep, Adinets and Merrill 2022): one kernel
//! reads the keys and counts the digits of all passes at once ([`RadixKind::Histogram`]); then
//! every pass is one kernel ([`RadixKind::Sweep`]): a workgroup takes a tile of keys, ranks it by
//! the digit, publishes the tile's count of every digit in a status word (aggregate flag and count
//! in one word), looks back over the statuses of the tiles before it until it finds an inclusive
//! prefix of each digit (decoupled look-back, as the scan of `scan.rs`), and writes the keys
//! (and payload) to their places. Per pass a key is read once and written once; the count
//! and scan kernels of the other way are gone. See [`radix_sweep`].
//!
//! *Count, scan, scatter* (wgpu, CPU, and anywhere else the single sweep does not apply: no
//! decoupled look-back, which needs forward-progress guarantees WGSL does not give), every pass in
//! three steps:
//!
//! 1. *Count*: a workgroup counts the digits of [`COUNT_TILES`] tiles in workgroup memory (every
//!    invocation loads all its keys of a tile before counting them, so the loads overlap) and
//!    writes them to `counts[digit * cstride + tile]`, four tiles at a time as one aligned `vec4`.
//! 2. *Scan*: the inclusive scan of `counts` (see `scan.rs`); with the digit-major layout the
//!    exclusive prefix of `(digit, tile)` is the global position of the first key of that digit
//!    in that tile.
//! 3. *Scatter*: a workgroup ranks its tile stably by the digit, then writes the keys, digit run
//!    by digit run, to their global positions.
//!
//! The ranking has two forms. On devices whose subgroups all have 32 invocations (and enough
//! workgroup memory, see `ballot_bytes`) it is a warp-level multisplit: every subgroup finds, per
//! key, the lanes with the same digit by ballots of the 8 digit bits, and a workgroup scan of the
//! per-(digit, subgroup) counters places the keys (see `ballot_scatter_body`); tiles are
//! [`BALLOT_PER_THREAD`] `* 256` keys. Elsewhere (no subgroups, other sizes: lavapipe, or
//! `LUMA_SUBGROUPS=off`) a tile of [`TILE`] keys is ranked by four rounds of a stable split by two
//! bits, each a workgroup scan of per-invocation bucket counts.
//!
//! Keys are sorted as `u32` words after an order-preserving transform ([`KeyTransform`]) applied
//! when the first pass loads them and inverted when the last pass stores them, so the passes in
//! between move plain words. A lazy key array is not evaluated first: the two kernels of the first
//! pass (count, and scatter) compute the keys from the expression, flattened into them like the
//! operand of a reduction (`jit::program::Tail::Radix`); the kernel text is the same as for a
//! buffer of keys except for how `load_key` reads. Descending order sorts the complemented key. Every
//! step keeps the input order of equal digits, so the sort is stable.
//!
//! Measured on an RTX 3090 (2^24 random `u32` keys, `bench_datasci --only sort`; the clocks of the
//! card vary by about 10% between runs), best of 7: 1.17 ms with ballots (`f32` the same, with a
//! payload 1.70 ms), 0.086 ms per count pass (4 bytes per key, at the bandwidth of the
//! card), 0.18 ms per scatter pass (8 bytes per key at the bandwidth of the card) and 0.022 ms
//! per pass for the scan; with the split ranking 2.08 ms (scatter 0.37 ms per pass).
//!
//! What mattered: the write of the scattered keys. A scatter kernel that ranks 2048 keys and
//! writes them in digit runs of 8 keys takes 0.37 ms whichever way it ranks (ballots took the
//! ranking from 0.1 ms to almost nothing and left the time unchanged), the same kernel
//! writing consecutive addresses takes 0.18 ms, and with 4096 keys per tile (runs of 16 keys, 64
//! bytes) the scattered writes are as fast as sequential ones. Loading all the keys of a tile
//! before counting them took the count kernel from 0.105 to 0.086 ms. Writing the counts as one
//! word per tile (digit-major) cost 0.06 ms per pass more than as `vec4`s of four tiles; a
//! radix-8 or subgroup scan and prefetching the offsets changed less than the run-to-run noise.

use std::fmt::Write;
use std::panic::Location;
use std::sync::Arc;

use luma_core::Buffer;

use crate::jit::structured::boundary::{dim, SortAttrs, SortParams, SortPayload, SortStage};
use crate::jit::structured::strategy::{candidates, note, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::array::{Array, Failure, Storage};
use crate::builtin::Builtin;
use crate::context::{Context, RawArg, Scratch};
use crate::error::LumaError;
use crate::gpu_type::{ElemLayout, ScalarType};
use crate::jit::{self, Node};
use crate::ops::{fold_workgroups, launch_1d, Outcome};
use crate::scan::{add_operator, Bound, Lines};
use crate::segsort::{LineJob, LineSort};
use crate::shape::Shape;
use crate::tuning::Tuning;

/// Invocations of a workgroup of the sort kernels.
pub(crate) const WORKGROUP: u32 = 256;
/// Keys an invocation owns in a tile of the split ranking.
const PER_THREAD: u32 = 8;
/// Keys a workgroup handles per pass with the split ranking.
pub(crate) const TILE: u32 = WORKGROUP * PER_THREAD;
/// Keys an invocation owns in a tile of the ballot kernels.
pub(crate) const BALLOT_PER_THREAD: u32 = 16;
/// Tiles whose counts a counting workgroup writes together.
const COUNT_TILES: u32 = 4;
/// Bits of the digit of a pass.
pub(crate) const DIGIT_BITS: u32 = 8;
/// Digits (counters per tile).
const DIGITS: u32 = 1 << DIGIT_BITS;

/// How a key is turned into a `u32` that orders like the key (and back).
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KeyTransform {
    /// `u32`: the word itself.
    Identity,
    /// `i32`: the sign bit flipped.
    Signed,
    /// `f32`: negative keys are complemented, the others get the sign bit set, and every NaN
    /// becomes the positive quiet NaN first, so NaNs sort after `+inf`.
    Float,
}

impl KeyTransform {
    /// The WGSL body of `fn key_in(b: u32) -> u32`.
    pub(crate) fn forward(self) -> &'static str {
        match self {
            KeyTransform::Identity => "return b;",
            KeyTransform::Signed => "return b ^ 0x80000000u;",
            KeyTransform::Float => {
                "let c = select(b, 0x7fc00000u, (b & 0x7fffffffu) > 0x7f800000u);
    return select(c | 0x80000000u, ~c, (c & 0x80000000u) != 0u);"
            }
        }
    }

    /// The WGSL body of `fn key_out(k: u32) -> u32`, the inverse of [`KeyTransform::forward`].
    pub(crate) fn inverse(self) -> &'static str {
        match self {
            KeyTransform::Identity => "return k;",
            KeyTransform::Signed => "return k ^ 0x80000000u;",
            KeyTransform::Float => "return select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u);",
        }
    }
}

/// What travels with the keys.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PayloadKind {
    None,
    /// The flat index of the key in the input, generated by the kernel (the first pass of an
    /// argsort).
    Index,
    /// A `u32` per key, read from a buffer.
    Buffer,
}

/// The element types [`Array::sort`] and its kin are defined for: `u32`, `i32` and `f32`.
pub trait SortKey: ScalarType {
    #[doc(hidden)]
    const TRANSFORM: KeyTransform;
}

impl SortKey for u32 {
    const TRANSFORM: KeyTransform = KeyTransform::Identity;
}

impl SortKey for i32 {
    const TRANSFORM: KeyTransform = KeyTransform::Signed;
}

impl SortKey for f32 {
    const TRANSFORM: KeyTransform = KeyTransform::Float;
}

/// What a radix kernel does.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RadixKind {
    /// Counts the digits of every tile of keys (see the module documentation).
    Count { ballot: bool },
    /// Ranks every tile by the digit and writes keys and payload to their global positions.
    /// `last` passes store keys through the inverse of the transform; `store_keys == false`
    /// stores the payload only.
    Scatter { last: bool, store_keys: bool, payload: PayloadKind, ballot: bool },
    /// One read of the keys that adds the counts of the digits of the first `digits` passes (bits
    /// `0..8 * digits`) to the global counts at the start of `st` (see [`SweepWork`]).
    Histogram { digits: u8 },
    /// One pass of the single-sweep sort (Merrill and Garland's decoupled look-back, as CUB's
    /// onesweep): ranks a tile by the digit like the ballot scatter, takes the position of every
    /// digit from the tiles before it (look-back over the statuses in `st`) and the global
    /// counts, and writes keys and payload; see [`sweep_body`].
    Sweep { last: bool, store_keys: bool, payload: PayloadKind, per_thread: u8 },
}

/// One kernel of a radix sort pass on the bits `shift..shift + 8`. The transform and the
/// direction only matter where keys are read from or written to the user's side (the first and
/// the last pass).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RadixPass {
    pub(crate) kind: RadixKind,
    pub(crate) shift: u8,
    pub(crate) transform: KeyTransform,
    pub(crate) descending: bool,
}

/// The fields every radix kernel's `Params` starts with: the keys, the tiles and the words
/// between the counts of two digits (see [`Spec`]); a fused kernel's scalars follow.
pub(crate) const PARAMS: &str = "n: u32, tiles: u32, cstride: u32, ";

/// How a radix kernel reads the key of an element.
#[derive(Clone, Copy)]
pub(crate) enum KeyLoad<'a> {
    /// From the buffer `keys`: the user's array (transformed) in the first pass, else the words
    /// the previous pass wrote.
    Buffer { first: bool },
    /// From the function `luma_key(i)` of the element type `ty` (the first pass over a lazy
    /// expression, see `jit::program::Tail::Radix`).
    Fused { ty: &'a str },
}

fn declare(src: &mut String, binding: &mut usize, access: &str, name: &str, elem: &str) {
    let _ = writeln!(src, "@group(1) @binding({binding}) var<storage, {access}> {name}: array<{elem}>;");
    *binding += 1;
}

impl RadixPass {
    /// Declares the buffers the kernel writes, from binding `*binding` on: the counts, or the
    /// keys (unless `!store_keys`) and the payload (unless there is none).
    pub(crate) fn declare_writes(self, src: &mut String, binding: &mut usize) {
        match self.kind {
            RadixKind::Count { .. } => declare(src, binding, "read_write", "counts", "vec4<u32>"),
            RadixKind::Histogram { .. } => declare(src, binding, "read_write", "st", "atomic<u32>"),
            RadixKind::Scatter { store_keys, payload, .. } | RadixKind::Sweep { store_keys, payload, .. } => {
                if store_keys {
                    declare(src, binding, "read_write", "out_keys", "u32");
                }
                if payload != PayloadKind::None {
                    declare(src, binding, "read_write", "out_pay", "u32");
                }
                if matches!(self.kind, RadixKind::Sweep { .. }) {
                    declare(src, binding, "read_write", "st", "atomic<u32>");
                }
            }
        }
    }

    /// Declares the buffers the kernel reads besides the keys: the payload (`PayloadKind::Buffer`)
    /// and the inclusive scan of the counts.
    pub(crate) fn declare_reads(self, src: &mut String, binding: &mut usize) {
        match self.kind {
            RadixKind::Scatter { payload, .. } => {
                if payload == PayloadKind::Buffer {
                    declare(src, binding, "read", "pay", "u32");
                }
                declare(src, binding, "read", "offs", "u32");
            }
            RadixKind::Sweep { payload, .. } => {
                if payload == PayloadKind::Buffer {
                    declare(src, binding, "read", "pay", "u32");
                }
            }
            RadixKind::Count { .. } | RadixKind::Histogram { .. } => {}
        }
    }
}

/// The key functions every sort kernel has. `load_key(i)` is the key of element `i` as the passes
/// order it: the first pass reads the user's array through the transform, later passes the words
/// the previous pass wrote. `key_out(w)` is the user's key of the word `w`.
fn key_functions(pass: RadixPass, load: KeyLoad<'_>) -> String {
    let complement = if pass.descending { "~" } else { "" };
    let mut src = String::new();
    let _ = writeln!(src, "fn key_in(b: u32) -> u32 {{\n    {}\n}}", pass.transform.forward());
    let _ = writeln!(src, "fn key_out(w: u32) -> u32 {{\n    let k = {complement}w;\n    {}\n}}", pass.transform.inverse());
    match load {
        KeyLoad::Buffer { first: true } => {
            let _ = writeln!(src, "fn load_key(i: u32) -> u32 {{ return {complement}key_in(keys[i]); }}");
        }
        KeyLoad::Buffer { first: false } => src.push_str("fn load_key(i: u32) -> u32 { return keys[i]; }\n"),
        KeyLoad::Fused { ty: "u32" } => {
            let _ = writeln!(src, "fn load_key(i: u32) -> u32 {{ return {complement}key_in(luma_key(i)); }}");
        }
        KeyLoad::Fused { ty } => {
            let _ = writeln!(src, "fn load_key(i: u32) -> u32 {{ return {complement}key_in(bitcast<u32>(luma_key(i))); }}");
            debug_assert!(ty == "i32" || ty == "f32");
        }
    }
    src
}

/// The WGSL of a kernel that reads the keys from the buffer `keys` (`first`: the user's array).
/// Arguments, in order: what the pass writes, keys, what it reads (see
/// [`RadixPass::declare_writes`] and [`RadixPass::declare_reads`]), uniform `Params`.
pub(crate) fn source(pass: RadixPass, first: bool) -> String {
    let mut src = format!("struct Params {{ {PARAMS}}}\n");
    let mut binding = 0;
    pass.declare_writes(&mut src, &mut binding);
    declare(&mut src, &mut binding, "read", "keys", "u32");
    pass.declare_reads(&mut src, &mut binding);
    let _ = writeln!(src, "@group(1) @binding({binding}) var<uniform> p: Params;");
    src.push_str(&key_functions(pass, KeyLoad::Buffer { first }));
    src.push_str(&body(pass));
    src
}

/// The WGSL of a first-pass kernel after its declarations, reading the keys through
/// `luma_key(i)`, which returns the element type `ty` (`"u32"`, `"i32"` or `"f32"`).
pub(crate) fn fused_body(pass: RadixPass, ty: &str) -> String {
    let mut src = key_functions(pass, KeyLoad::Fused { ty });
    src.push_str(&body(pass));
    src
}

fn body(pass: RadixPass) -> String {
    match pass.kind {
        RadixKind::Count { ballot } => count_body(pass.shift, ballot),
        RadixKind::Scatter { last, store_keys, payload, ballot: false } => scatter_body(pass.shift, last, store_keys, payload),
        RadixKind::Scatter { last, store_keys, payload, ballot: true } => ballot_scatter_body(pass.shift, last, store_keys, payload, None),
        RadixKind::Sweep { last, store_keys, payload, per_thread } => ballot_scatter_body(pass.shift, last, store_keys, payload, Some(u32::from(per_thread))),
        RadixKind::Histogram { digits } => histogram_body(digits),
    }
}

/// Counters of the digits of every pass at the start of the work buffer of a sweep sort (see
/// [`SweepWork`]), then one ticket counter per pass.
const SWEEP_TICKETS: u32 = 4 * DIGITS;
/// Words of the work buffer before the statuses of the first pass.
const SWEEP_HEADER: u32 = SWEEP_TICKETS + 8;
/// Keys an invocation owns in a tile of the sweep kernels, the most of 32 (16 with a payload), 24
/// and 16 whose workgroup memory fits (more keys per tile make longer runs of equal digits to write, and fewer
/// statuses).
pub(crate) fn sweep_per_thread(tuning: &Tuning, payload: bool) -> u32 {
    // With a payload the registers of the tile are twice as many and 16 keys per invocation was
    // measured faster (RTX 3090, 10M keys: argsort 0.83 ms against 0.85 at 24 and 0.90 at 32).
    let wanted: u32 = if payload { 16 } else { 32 };
    [32, 24, 16].into_iter().filter(|&k| k <= wanted).find(|&k| sweep_bytes(k) <= tuning.workgroup_storage).unwrap_or(16)
}

/// Bytes of workgroup memory of the sweep kernel with `per_thread` keys per invocation: the
/// counters, the digit bases, two arrays of subgroup totals, a ticket and the staged keys.
pub(crate) fn sweep_bytes(per_thread: u32) -> u32 {
    let words = DIGITS * (WORKGROUP / SUBGROUP + 1) + DIGITS + 2 * (WORKGROUP / SUBGROUP) + 1 + WORKGROUP * per_thread;
    words * 4
}

/// Keys an invocation of the histogram kernel loads before it counts them.
const HIST_PER_THREAD: u32 = 8;

/// The histogram kernel of a sweep sort: workgroups stride over tiles of
/// `WORKGROUP * HIST_PER_THREAD` keys (`p.tiles` of them are launched), count the digits of the
/// first `digits` passes in workgroup memory, and add their counts to the global ones once. Equal
/// digits of a subgroup are counted with one atomic (the common case of skewed keys, which would
/// otherwise serialise on one counter).
fn histogram_body(digits: u8) -> String {
    let tile = WORKGROUP * HIST_PER_THREAD;
    let digits_n = u32::from(digits);
    let mut src = String::new();
    let _ = writeln!(src, "var<workgroup> hist: array<atomic<u32>, {}>;", digits_n * DIGITS);
    let mut full = String::new();
    let mut tail = String::new();
    let digit = |k: u32, key: &str| format!("(({key} >> {}u) & {}u)", k * DIGIT_BITS, DIGITS - 1);
    for m in 0..HIST_PER_THREAD {
        let _ = writeln!(full, "        let k{m} = load_key(base + {}u + t);", m * WORKGROUP);
    }
    for m in 0..HIST_PER_THREAD {
        // A subgroup whose lanes hold the same key adds 32 to the counter of each digit once; the
        // others add one per lane (skewed keys would otherwise serialise on a counter).
        let _ = writeln!(full, "        {{\n            let kf = subgroupShuffle(k{m}, 0u);\n            if (subgroupBallot(k{m} == kf).x == 0xffffffffu) {{\n                if (lane == 0u) {{");
        for k in 0..digits_n {
            let _ = writeln!(full, "                    atomicAdd(&hist[{}u + {}], 32u);", k * DIGITS, digit(k, "kf"));
        }
        full.push_str("                }\n            } else {\n");
        for k in 0..digits_n {
            let _ = writeln!(full, "                atomicAdd(&hist[{}u + {}], 1u);", k * DIGITS, digit(k, &format!("k{m}")));
        }
        full.push_str("            }\n        }\n");
        let _ = writeln!(tail, "        {{\n            let i = base + {}u + t;\n            if (i < p.n) {{\n                let k = load_key(i);", m * WORKGROUP);
        for k in 0..digits_n {
            let _ = writeln!(tail, "                atomicAdd(&hist[{}u + {}], 1u);", k * DIGITS, digit(k, "k"));
        }
        tail.push_str("            }\n        }\n");
    }
    let _ = write!(
        src,
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    let group = wg.x + wg.y * nwg.x;
    if (group >= p.tiles) {{ return; }}
    for (var k = 0u; k < {digits}u; k++) {{ atomicStore(&hist[k * {DIGITS}u + t], 0u); }}
    workgroupBarrier();
    for (var base = group * {tile}u; base < p.n; base += p.tiles * {tile}u) {{
        if (base + {tile}u <= p.n) {{
{full}        }} else {{
{tail}        }}
    }}
    workgroupBarrier();
    for (var k = 0u; k < {digits}u; k++) {{
        let v = atomicLoad(&hist[k * {DIGITS}u + t]);
        if (v != 0u) {{ atomicAdd(&st[k * {DIGITS}u + t], v); }}
    }}
}}
"
    );
    src
}

/// The counting kernel of the pass that sorts by bits `shift..shift + 8`: a workgroup counts the
/// digits of [`COUNT_TILES`] consecutive tiles, so that every invocation writes the counts of its
/// digit for those tiles as one aligned `vec4` (a count per tile written on its own is a 4-byte
/// write to a sector of its own, which cost as much as reading the keys).
fn count_body(shift: u8, ballot: bool) -> String {
    let per_thread = if ballot { BALLOT_PER_THREAD } else { PER_THREAD };
    let tile = WORKGROUP * per_thread;
    let mut src = String::new();
    let mut loads = String::new();
    let mut adds = String::new();
    for m in 0..per_thread {
        let off = m * WORKGROUP;
        let _ = writeln!(loads, "        var k{m} = 0xffffffffu;\n        if (base + {off}u < p.n) {{ k{m} = load_key(base + {off}u); }}");
        if ballot {
            let _ = writeln!(
                adds,
                "        {{
            let ok = base + {off}u < p.n;
            let d = (k{m} >> {shift}u) & {mask}u;
            let act = subgroupBallot(ok).x;
            if (act != 0u) {{
                let first = firstTrailingBit(act);
                let d0 = subgroupShuffle(d, first);
                if (subgroupBallot(ok && d == d0).x == act) {{
                    if (lane == first) {{ atomicAdd(&hist[j * {DIGITS}u + d0], countOneBits(act)); }}
                }} else if (ok) {{
                    atomicAdd(&hist[j * {DIGITS}u + d], 1u);
                }}
            }}
        }}",
                mask = DIGITS - 1
            );
        } else {
            let _ = writeln!(adds, "        if (base + {off}u < p.n) {{ atomicAdd(&hist[j * {DIGITS}u + ((k{m} >> {shift}u) & {}u)], 1u); }}", DIGITS - 1);
        }
    }
    let lane_param = if ballot { ", @builtin(subgroup_invocation_id) lane: u32" } else { "" };
    let _ = writeln!(src, "var<workgroup> hist: array<atomic<u32>, {}>;", COUNT_TILES * DIGITS);
    let _ = write!(
        src,
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>{lane_param}) {{
    let group = wg.x + wg.y * nwg.x;
    let first_tile = group * {COUNT_TILES}u;
    if (first_tile >= p.tiles) {{ return; }}
    for (var j = 0u; j < {COUNT_TILES}u; j++) {{ atomicStore(&hist[j * {DIGITS}u + t], 0u); }}
    workgroupBarrier();
    for (var j = 0u; j < {COUNT_TILES}u; j++) {{
        let base = (first_tile + j) * {tile}u + t;
{loads}{adds}    }}
    workgroupBarrier();
    counts[t * (p.cstride / {COUNT_TILES}u) + group] = vec4<u32>(atomicLoad(&hist[t]), atomicLoad(&hist[{DIGITS}u + t]), atomicLoad(&hist[{d2}u + t]), atomicLoad(&hist[{d3}u + t]));
}}
",
        d2 = 2 * DIGITS,
        d3 = 3 * DIGITS,
    );
    src
}

/// The eight keys of an invocation as `v0 .. v7`: `v{j}` for `j` in `0..8`.
fn each_key(mut f: impl FnMut(u32) -> String) -> String {
    (0..PER_THREAD).map(&mut f).collect()
}

/// The scatter kernel of a pass (see [`RadixKind::Scatter`]).
fn scatter_body(shift: u8, last: bool, store_keys: bool, payload: PayloadKind) -> String {
    let mut src = String::new();
    // Without a payload the keys themselves travel through the workgroup memory, so nothing has to
    // be fetched again at the end. With one, a word holds the digit and the position of the key
    // in the tile (11 bits), which the kernel fetches key and payload by afterwards (no room in
    // workgroup memory for both).
    let carry_keys = payload == PayloadKind::None;
    let padded = TILE + TILE / 8;
    let _ = writeln!(src, "var<workgroup> sd: array<u32, {padded}>;");
    let _ = writeln!(src, "var<workgroup> scan_a: array<vec2<u32>, {WORKGROUP}>;");
    let _ = writeln!(src, "var<workgroup> scan_b: array<vec2<u32>, {WORKGROUP}>;");
    let _ = writeln!(src, "var<workgroup> start: array<u32, {DIGITS}>;");
    let _ = writeln!(src, "var<workgroup> dest: array<u32, {DIGITS}>;");
    // Padding one word per eight keeps the invocations' runs of eight keys in different banks.
    src.push_str("fn ix(i: u32) -> u32 { return i + (i >> 3u); }\n");
    let digit = if carry_keys { format!("(w >> {shift}u) & {}u", DIGITS - 1) } else { format!("w & {}u", DIGITS - 1) };
    let _ = writeln!(src, "fn digit(w: u32) -> u32 {{ return {digit}; }}");
    let _ = write!(
        src,
        "@compute @workgroup_size({WORKGROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let tile = wg.x + wg.y * nwg.x;
    if (tile >= p.tiles) {{ return; }}
    let base = tile * {TILE}u;
    // The keys past the end get the largest key, so they rank after the real ones and are never
    // written.
    for (var m = 0u; m < {PER_THREAD}u; m++) {{
        let i = m * {WORKGROUP}u + t;
        var w = 0xffffffffu;
        if (base + i < p.n) {{ w = load_key(base + i); }}
        sd[ix(i)] = {pack};
    }}
    workgroupBarrier();
",
        pack = if carry_keys { "w".to_owned() } else { format!("((w >> {shift}u) & {mask}u) | (i << 8u)", mask = DIGITS - 1) },
    );
    src.push_str(&each_key(|j| format!("    var v{j} = sd[ix(t * {PER_THREAD}u + {j}u)];\n")));
    // Two bits per round: a stable split into four buckets. Every invocation counts its keys per
    // bucket in 16-bit lanes of two words (a lane never exceeds the 2048 keys of a tile), the
    // workgroup scans those across the invocations, and every key goes to the running position of
    // its bucket, which is also kept in a lane.
    for round in 0..DIGIT_BITS / 2 {
        // The bits of the digit this round splits by, in the words that travel.
        let shift = 2 * round + if carry_keys { u32::from(shift) } else { 0 };
        let _ = writeln!(src, "    {{");
        for j in 0..PER_THREAD {
            let _ = writeln!(src, "        let b{j} = (v{j} >> {shift}u) & 3u;");
            let _ = writeln!(src, "        let h{j} = (b{j} & 1u) << 4u;");
            let _ = writeln!(src, "        let n{j} = 1u << h{j};");
        }
        let low = (0..PER_THREAD).map(|j| format!("select(0u, n{j}, b{j} < 2u)")).collect::<Vec<_>>().join(" + ");
        let high = (0..PER_THREAD).map(|j| format!("select(n{j}, 0u, b{j} < 2u)")).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "        let counts = vec2<u32>({low}, {high});");
        src.push_str("        scan_a[t] = counts;\n        workgroupBarrier();\n");
        // Hillis-Steele inclusive scan, alternating between two buffers.
        let (mut from, mut to) = ("scan_a", "scan_b");
        let mut d = 1;
        while d < WORKGROUP {
            let _ = writeln!(src, "        {{ var x = {from}[t]; if (t >= {d}u) {{ x += {from}[t - {d}u]; }} {to}[t] = x; }}");
            src.push_str("        workgroupBarrier();\n");
            std::mem::swap(&mut from, &mut to);
            d *= 2;
        }
        let _ = writeln!(src, "        let before = {from}[t] - counts;");
        let _ = writeln!(src, "        let total = {from}[{}u];", WORKGROUP - 1);
        let _ = writeln!(src, "        let t0 = total.x & 0xffffu;");
        let _ = writeln!(src, "        let t01 = t0 + (total.x >> 16u);");
        let _ = writeln!(src, "        let t012 = t01 + (total.y & 0xffffu);");
        let _ = writeln!(src, "        var w0 = before.x + (t0 << 16u);");
        let _ = writeln!(src, "        var w1 = before.y + (t01 | (t012 << 16u));");
        for j in 0..PER_THREAD {
            let _ = writeln!(src, "        let low{j} = b{j} < 2u;");
            let _ = writeln!(src, "        sd[ix((select(w1, w0, low{j}) >> h{j}) & 0xffffu)] = v{j};");
            let _ = writeln!(src, "        w0 += select(0u, n{j}, low{j});");
            let _ = writeln!(src, "        w1 += select(n{j}, 0u, low{j});");
        }
        src.push_str("        workgroupBarrier();\n");
        if round + 1 < DIGIT_BITS / 2 {
            src.push_str(&each_key(|j| format!("        v{j} = sd[ix(t * {PER_THREAD}u + {j}u)];\n")));
        }
        src.push_str("    }\n");
    }
    // The digit runs of the sorted tile and where the digits begin in the output.
    let _ = write!(
        src,
        "    for (var m = 0u; m < {PER_THREAD}u; m++) {{
        let q = m * {WORKGROUP}u + t;
        let d = digit(sd[ix(q)]);
        var run_start = q == 0u;
        if (!run_start) {{ run_start = digit(sd[ix(q - 1u)]) != d; }}
        if (run_start) {{ start[d] = q; }}
    }}
    {{
        let q = t * p.cstride + tile;
        var o = 0u;
        if (q > 0u) {{ o = offs[q - 1u]; }}
        dest[t] = o;
    }}
    workgroupBarrier();
    let valid = min({TILE}u, p.n - base);
    for (var m = 0u; m < {PER_THREAD}u; m++) {{
        let q = m * {WORKGROUP}u + t;
        if (q < valid) {{
            let w = sd[ix(q)];
            let d = digit(w);
            let pos = dest[d] + q - start[d];
",
    );
    if !carry_keys {
        src.push_str("            let idx = base + (w >> 8u);\n");
    }
    if store_keys {
        let key = match (carry_keys, last) {
            (true, true) => "key_out(w)",
            (true, false) => "w",
            (false, true) => "key_out(load_key(idx))",
            (false, false) => "load_key(idx)",
        };
        let _ = writeln!(src, "            out_keys[pos] = {key};");
    }
    match payload {
        PayloadKind::None => {}
        PayloadKind::Index => src.push_str("            out_pay[pos] = idx;\n"),
        PayloadKind::Buffer => src.push_str("            out_pay[pos] = pay[idx];\n"),
    }
    src.push_str("        }\n    }\n}\n");
    src
}

/// Invocations of a subgroup the ballot kernels assume (see `Tuning::subgroup32`).
const SUBGROUP: u32 = 32;

/// The scatter kernel of a pass on a device with 32-wide subgroups (see [`RadixKind::Scatter`]):
/// the stable ranking is a warp-level multisplit. A subgroup takes a run of `8 * 32` keys of the
/// tile, 32 at a time (keys `i` of a round are consecutive, so the reads are coalesced); for
/// each key the lanes holding the same digit are found by ANDing, over the 8 bits of the digit,
/// the ballot of the bit or of its complement (the lane's *peers*). The lowest peer adds the
/// number of peers to the subgroup's counter of the digit and hands the old value to the others,
/// so the rank of a key within (digit, subgroup) is that value plus the peers below the lane.
/// One workgroup scan of the counters, digit-major (every invocation owns a digit and its
/// subgroups' counters), gives each (digit, subgroup) its first slot in the sorted tile. Order
/// within a digit is (subgroup, round, lane) = input order, so the ranking is stable. The ranked
/// tile is staged in workgroup memory and written out in slot order, so that the invocations
/// of a subgroup write consecutive addresses of a digit's run.
fn ballot_scatter_body(shift: u8, last: bool, store_keys: bool, payload: PayloadKind, sweep_pt: Option<u32>) -> String {
    let sweep = sweep_pt.is_some();
    let bpt = sweep_pt.unwrap_or(BALLOT_PER_THREAD);
    let warps = WORKGROUP / SUBGROUP;
    // Counters of a digit are `warps + 1` words apart: no bank conflicts in the scan.
    let stride = warps + 1;
    let has_pay = payload != PayloadKind::None;
    let tile = WORKGROUP * bpt;
    let mut src = String::new();
    let _ = writeln!(src, "var<workgroup> wc: array<atomic<u32>, {}>;", DIGITS * stride);
    let _ = writeln!(src, "var<workgroup> wsum: array<u32, {warps}>;");
    let _ = writeln!(src, "var<workgroup> gbase: array<u32, {DIGITS}>;");
    let _ = writeln!(src, "var<workgroup> sk: array<u32, {tile}>;");
    if sweep {
        let _ = writeln!(src, "var<workgroup> wsum2: array<u32, {warps}>;");
        let _ = writeln!(src, "var<workgroup> ticket: u32;");
    }
    let _ = writeln!(src, "fn digit(w: u32) -> u32 {{ return (w >> {shift}u) & {}u; }}", DIGITS - 1);
    let mut copy_writes = String::new();
    if store_keys {
        let _ = writeln!(copy_writes, "                let w = load_key(idx);\n                out_keys[idx] = {};", if last { "key_out(w)" } else { "w" });
    }
    match payload {
        PayloadKind::None => {}
        PayloadKind::Index => copy_writes.push_str("                out_pay[idx] = idx;\n"),
        PayloadKind::Buffer => copy_writes.push_str("                out_pay[idx] = pay[idx];\n"),
    }
    let _ = write!(
        src,
        "@compute @workgroup_size({WORKGROUP})
fn main(
    @builtin(local_invocation_index) t: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {{
    {head}",
        head = if sweep {
            format!(
                "let group = wg.x + wg.y * nwg.x;
    if (group >= p.tiles) {{ return; }}
    let sg = t / {SUBGROUP}u;
    let sb = p.cstride;
    for (var k = 0u; k < {stride}u; k++) {{ atomicStore(&wc[k * {DIGITS}u + t], 0u); }}
    // The count of every key of digit `t` in this pass (see `SweepWork`): all keys in one digit
    // make the pass a copy.
    let gcnt = atomicLoad(&st[{gh}u + t]);
    let same_d = p.n > 0u && gcnt == p.n;
    let sg_same = subgroupBallot(same_d).x;
    if (lane == 0u) {{ wsum[sg] = sg_same; }}
    workgroupBarrier();
    if (({any_wsum}) != 0u) {{
        let base = group * {tile}u;
        let valid = min({tile}u, p.n - base);
        for (var m = 0u; m < {bpt}u; m++) {{
            let q = m * {WORKGROUP}u + t;
            if (q < valid) {{
                let idx = base + q;
{copy_writes}            }}
        }}
        return;
    }}
    // Tiles are numbered in the order their workgroups start, so every tile a look-back waits
    // for has started.
    if (t == 0u) {{ ticket = atomicAdd(&st[{tk}u], 1u); }}
    let tile = workgroupUniformLoad(&ticket);
    let base = tile * {tile}u;
    let first_slot = sg * {run}u + lane;
",
                gh = (shift as u32 / DIGIT_BITS) * DIGITS,
                tk = SWEEP_TICKETS + shift as u32 / DIGIT_BITS,
                run = bpt * SUBGROUP,
                any_wsum = (0..warps).map(|w| format!("wsum[{w}]")).collect::<Vec<_>>().join(" | "),
            )
        } else {
            format!(
                "let tile = wg.x + wg.y * nwg.x;
    if (tile >= p.tiles) {{ return; }}
    let base = tile * {tile}u;
    let sg = t / {SUBGROUP}u;
    for (var k = 0u; k < {stride}u; k++) {{ atomicStore(&wc[k * {DIGITS}u + t], 0u); }}
    // `select` evaluates both operands: the index of the previous digit's offset is clamped so that
    // digit 0 does not read `offs[-1]` (an out-of-bounds load on backends without robust access).
    let same_d = p.n > 0u && (offs[t * p.cstride + p.tiles - 1u] - select(0u, offs[max(t * p.cstride, 1u) - 1u], t > 0u) == p.n);
    let sg_same = subgroupBallot(same_d).x;
    if (lane == 0u) {{ wsum[sg] = sg_same; }}
    workgroupBarrier();
    if (({any_wsum}) != 0u) {{
        let valid = min({tile}u, p.n - base);
        for (var m = 0u; m < {bpt}u; m++) {{
            let q = m * {WORKGROUP}u + t;
            if (q < valid) {{
                let idx = base + q;
{copy_writes}            }}
        }}
        return;
    }}
    let first_slot = sg * {run}u + lane;
",
                run = bpt * SUBGROUP,
                any_wsum = (0..warps).map(|w| format!("wsum[{w}]")).collect::<Vec<_>>().join(" | "),
            )
        },
    );
    // The keys past the end get the largest key, so they rank after the real ones and are never
    // written.
    for j in 0..bpt {
        let _ = writeln!(src, "    let i{j} = base + first_slot + {}u;", j * SUBGROUP);
        let _ = writeln!(src, "    var w{j} = 0xffffffffu;\n    if (i{j} < p.n) {{ w{j} = load_key(i{j}); }}");
        match payload {
            PayloadKind::None => {}
            PayloadKind::Index => {
                let _ = writeln!(src, "    let y{j} = i{j};");
            }
            PayloadKind::Buffer => {
                let _ = writeln!(src, "    var y{j} = 0u;\n    if (i{j} < p.n) {{ y{j} = pay[i{j}]; }}");
            }
        }
    }
    src.push_str("    let below = (1u << lane) - 1u;\n");
    for j in 0..bpt {
        let _ = writeln!(src, "    var r{j} = 0u;\n    {{\n        let d = digit(w{j});\n        var peers = 0xffffffffu;");
        for bit in 0..DIGIT_BITS {
            let _ = writeln!(src, "        {{ let on = ((d >> {bit}u) & 1u) == 1u; let m = subgroupBallot(on).x; peers &= select(~m, m, on); }}");
        }
        let _ = writeln!(src, "        let leader = firstTrailingBit(peers);\n        var before = 0u;");
        let _ = writeln!(src, "        if (lane == leader) {{ before = atomicAdd(&wc[d * {stride}u + sg], countOneBits(peers)); }}");
        let _ = writeln!(src, "        r{j} = subgroupShuffle(before, leader) + countOneBits(peers & below);\n    }}");
    }
    // The invocation of digit `t`: the counters of its subgroups to prefix sums, the digits'
    // first slots by a workgroup scan of their totals.
    src.push_str("    workgroupBarrier();\n");
    for k in 0..warps {
        let _ = writeln!(src, "    let c{k} = atomicLoad(&wc[t * {stride}u + {k}u]);");
    }
    for k in 1..warps {
        let sum = (0..k).map(|i| format!("c{i}")).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "    let e{k} = {sum};");
    }
    let total = (0..warps).map(|i| format!("c{i}")).collect::<Vec<_>>().join(" + ");
    let _ = write!(
        src,
        "    let total = {total};
    let before_in_group = subgroupExclusiveAdd(total);
    let group_total = subgroupAdd(total);
    if (lane == 0u) {{ wsum[sg] = group_total; }}
{sweep_scan}    workgroupBarrier();
    var groups_before = 0u;
    for (var s = 0u; s < {warps}u; s++) {{ groups_before += select(0u, wsum[s], s < sg); }}
    let start = groups_before + before_in_group;
{gbase_init}    atomicStore(&wc[t * {stride}u], start);
",
        sweep_scan = if sweep {
            // The aggregate of the tile is published as soon as it is known; the first tile's is
            // its inclusive prefix. The other half: the digits' global first positions.
            "    if (tile == 0u) { atomicStore(&st[sb + t], (2u << 30u) | total); } else { atomicStore(&st[sb + tile * 256u + t], (1u << 30u) | total); }
    let gb_in = subgroupExclusiveAdd(gcnt);
    let gb_tot = subgroupAdd(gcnt);
    if (lane == 0u) { wsum2[sg] = gb_tot; }
"
        } else {
            ""
        },
        gbase_init = if sweep {
            ""
        } else {
            "    {
        let q = t * p.cstride + tile;
        var o = 0u;
        if (q > 0u) { o = offs[q - 1u]; }
        gbase[t] = o - start;
    }
"
        },
    );
    for k in 1..warps {
        let _ = writeln!(src, "    atomicStore(&wc[t * {stride}u + {k}u], start + e{k});");
    }
    src.push_str("    workgroupBarrier();\n");
    for j in 0..bpt {
        let _ = writeln!(src, "    let p{j} = atomicLoad(&wc[digit(w{j}) * {stride}u + sg]) + r{j};\n    sk[p{j}] = w{j};");
    }
    if sweep {
        // The first position of digit `t` in the output: the counts of the smaller digits, plus
        // the counts of `t` in the tiles before this one, found by looking back over their
        // statuses (aggregate: flag 1, inclusive prefix: flag 2, in bits 30 and 31 of a word with
        // the count; one word, so a reader never sees a flag without its count).
        src.push_str(
            "    {
        var excl = 0u;
        if (tile > 0u) {
            var j = tile;
            loop {
                j -= 1u;
                var w = atomicLoad(&st[sb + j * 256u + t]);
                while (w == 0u) { w = atomicLoad(&st[sb + j * 256u + t]); }
                excl += w & 0x3fffffffu;
                if ((w >> 30u) == 2u) { break; }
            }
            atomicStore(&st[sb + tile * 256u + t], (2u << 30u) | (excl + total));
        }
        var gb = gb_in;
        for (var s = 0u; s < 8u; s++) { gb += select(0u, wsum2[s], s < sg); }
        gbase[t] = gb + excl - start;
    }
",
        );
    }
    let _ = writeln!(src, "    workgroupBarrier();\n    let valid = min({tile}u, p.n - base);");
    if !has_pay {
        let _ = write!(
            src,
            "    for (var m = 0u; m < {bpt}u; m++) {{
        let q = m * {WORKGROUP}u + t;
        if (q < valid) {{
            let w = sk[q];
            let pos = gbase[digit(w)] + q;
"
        );
        if store_keys {
            let _ = writeln!(src, "            out_keys[pos] = {};", if last { "key_out(w)" } else { "w" });
        }
        src.push_str("        }\n    }\n}\n");
        return src;
    }
    // With a payload the keys are written first, then `sk` is reused for the payload.
    let at = |m: u32| if m == 0 { "t".to_string() } else { format!("({}u + t)", m * WORKGROUP) };
    for m in 0..bpt {
        let q = at(m);
        let _ = writeln!(src, "    var d{m} = 0u;\n    if ({q} < valid) {{\n        let w = sk[{q}];\n        d{m} = gbase[digit(w)] + {q};");
        if store_keys {
            let _ = writeln!(src, "        out_keys[d{m}] = {};", if last { "key_out(w)" } else { "w" });
        }
        src.push_str("    }\n");
    }
    src.push_str("    workgroupBarrier();\n");
    for j in 0..bpt {
        let _ = writeln!(src, "    sk[p{j}] = y{j};");
    }
    src.push_str("    workgroupBarrier();\n");
    for m in 0..bpt {
        let q = at(m);
        let _ = writeln!(src, "    if ({q} < valid) {{ out_pay[d{m}] = sk[{q}]; }}");
    }
    src.push_str("}\n");
    src
}

/// Bytes of workgroup memory the ballot scatter kernel declares: the counters (a word per digit and
/// subgroup, padded), the digit bases, the subgroup totals, and the staged keys (reused for the payload).
pub(crate) fn ballot_bytes() -> u32 {
    let words = DIGITS * (WORKGROUP / SUBGROUP + 1) + DIGITS + WORKGROUP / SUBGROUP + WORKGROUP * BALLOT_PER_THREAD;
    words * 4
}

/// Whether a sort ranks with ballots: subgroups of 32 invocations and room for the ballot
/// kernel's workgroup memory.
pub(crate) fn use_ballots(tuning: &Tuning) -> bool {
    tuning.subgroup32 && ballot_bytes() <= tuning.workgroup_storage
}

/// What a sort computes.
#[derive(Clone, Copy)]
pub(crate) struct Spec {
    pub(crate) transform: KeyTransform,
    pub(crate) descending: bool,
    /// Low bits of the transformed key that order the keys (1 to 32); higher bits are ignored.
    pub(crate) key_bits: u32,
    /// Whether the sorted keys are wanted (an argsort wants the payload only).
    pub(crate) want_keys: bool,
}

/// The payload that travels with the keys.
pub(crate) enum Payload<'a> {
    None,
    /// The flat index of every key.
    Index,
    /// The words of this storage (as many as keys).
    Buffer(&'a Storage),
}

/// The sorted keys and payload; each is `None` if not asked for.
pub(crate) struct Sorted {
    pub(crate) keys: Option<Arc<Storage>>,
    pub(crate) payload: Option<Arc<Storage>>,
}

/// The keys of a sort.
#[derive(Clone, Copy)]
pub(crate) enum Keys<'a> {
    /// The words of this storage.
    Ready(&'a Storage),
    /// The lazy expression behind this node, `u32`, `i32` or `f32`: the first pass computes the
    /// keys itself instead of reading them from a buffer (twice, by the counting and the
    /// scattering kernel; evaluating them into a buffer first costs a write and a read, more than
    /// recomputing an elementwise expression). Evaluated first if it cannot be fused: see
    /// `jit::radix_pass`.
    Pending(&'a Arc<Node>),
}

impl Keys<'_> {
    fn len(&self) -> usize {
        match self {
            Keys::Ready(storage) => storage.len(),
            Keys::Pending(node) => node.len(),
        }
    }
}

/// The boundary of the selection at the entry of a sort of `len` elements, and the host path's parameters if the registry has it
/// apply (it may still decline: `try_*` answers `None` and the sort goes on to the radix kernels).
fn host_candidate(ctx: &Context, op: &'static str, location: &'static Location<'static>, len: usize, errored: bool) -> Option<(Boundary<'static>, Params)> {
    let boundary = Boundary::new(
        FamilyTag::Sort,
        op,
        location,
        [dim(len), 1, 1, 0],
        BoundaryAttrs::Sort(SortAttrs { stage: SortStage::Entry, payload: SortPayload::None, errored, value4: false, key_bits: 32, len: len as u64 }),
    );
    let params = candidates(ctx, &boundary).find_map(|(id, params)| (id == StrategyId::SORT_HOST).then_some(params))?;
    Some((boundary, params))
}

/// Records the stable LSD radix sort of the `u32` words of `keys` (see [`Spec`]), carrying
/// `payload`. `keys` is read as is by the first pass, through `spec.transform`.
pub(crate) fn radix_sort(ctx: &Context, keys: Keys<'_>, payload: Payload<'_>, spec: Spec) -> Outcome<Sorted> {
    let n = keys.len();
    let too_large = || LumaError::DimensionTooLarge { shape: [n, 1, 1, 1] };
    let n32 = u32::try_from(n).map_err(|_| too_large())?;
    let boundary = Boundary::new(
        FamilyTag::Sort,
        "sort",
        Location::caller(),
        [n32, 1, 1, 0],
        BoundaryAttrs::Sort(SortAttrs {
            stage: SortStage::Radix,
            payload: match payload {
                Payload::None => SortPayload::None,
                Payload::Index => SortPayload::Index,
                Payload::Buffer(_) => SortPayload::Buffer,
            },
            errored: false,
            value4: false,
            key_bits: spec.key_bits.clamp(1, 32) as u8,
            len: n as u64,
        }),
    );
    let (chosen, params) = candidates(ctx, &boundary).next().expect("the count/scan/scatter radix sort always fits");
    note(ctx, &boundary, chosen, &params, &[]);
    let Params::Sort(params) = params else { unreachable!("a sort strategy runs with sort parameters") };
    if chosen == StrategyId::SORT_RADIX_SWEEP {
        return radix_sweep(ctx, keys, payload, spec, n32, params);
    }
    let ballot = chosen == StrategyId::SORT_RADIX_BALLOTS;
    let tiles = n32.div_ceil(params.tile);
    let passes = params.passes;
    let bytes = n as u64 * 4;
    // The counts of a digit are `cstride` words, a whole number of the `vec4`s a counting
    // workgroup writes; the words past `tiles` stay zero.
    let count_groups = tiles.div_ceil(COUNT_TILES);
    let cstride = count_groups * COUNT_TILES;
    let count_words = u64::from(DIGITS) * u64::from(cstride);
    let count_bytes = count_words * 4;

    // The buffer the first pass reads: unless the keys are fused into it, as soon as the count
    // kernel finds they cannot be.
    let mut key_pin = match keys {
        Keys::Ready(storage) => Some(storage.pin_single()?),
        Keys::Pending(_) => {
            if bytes > ctx.segment_bytes() {
                return Err(LumaError::Chunked("a sort of an array split over several buffers").into());
            }
            None
        }
    };
    let input_pay = match payload {
        Payload::Buffer(storage) => Some(storage.pin_single()?),
        _ => None,
    };
    let first_kind = match payload {
        Payload::None => PayloadKind::None,
        Payload::Index => PayloadKind::Index,
        Payload::Buffer(_) => PayloadKind::Buffer,
    };
    let out_keys = spec.want_keys.then(|| Arc::new(Storage::new(ctx, n, ElemLayout::plain(4))));
    let out_pay = (first_kind != PayloadKind::None).then(|| Arc::new(Storage::new(ctx, n, ElemLayout::plain(4))));
    let out_keys_pin = out_keys.as_ref().map(|storage| storage.pin_single()).transpose()?;
    let out_pay_pin = out_pay.as_ref().map(|storage| storage.pin_single()).transpose()?;

    let operator = add_operator(ctx)?;
    let mut lines = Lines::words(ctx, &operator);
    let mut scratch = Scratch::new(ctx);
    let counts = scratch.take(count_bytes)?;
    let scanned = scratch.take(count_bytes)?;
    // The results of the passes before the last alternate between two buffers.
    let mids = passes.saturating_sub(1).min(2) as usize;
    let mid_keys = (0..mids).map(|_| scratch.take(bytes)).collect::<std::result::Result<Vec<Buffer>, LumaError>>()?;
    let mid_pay = if first_kind == PayloadKind::None {
        Vec::new()
    } else {
        (0..mids).map(|_| scratch.take(bytes)).collect::<std::result::Result<Vec<Buffer>, LumaError>>()?
    };
    let params = [n32, tiles, cstride];
    let groups = fold_workgroups(ctx, tiles)?;
    let count_grid = fold_workgroups(ctx, count_groups)?;
    // Whether the first pass computes the keys (set by its count kernel).
    let mut fused = false;

    for pass in 0..passes {
        let (first, last) = (pass == 0, pass + 1 == passes);
        let shift = (pass * DIGIT_BITS) as u8;
        let src_pay = if first { input_pay.as_ref().map(|pin| pin.buffer()) } else { mid_pay.get((pass as usize - 1) % 2) };
        let dst_keys = if last { out_keys_pin.as_ref().map(|pin| pin.buffer()) } else { Some(&mid_keys[pass as usize % 2]) };
        let dst_pay = if last { out_pay_pin.as_ref().map(|pin| pin.buffer()) } else { mid_pay.get(pass as usize % 2) };
        let kind = if first { first_kind } else if first_kind == PayloadKind::None { PayloadKind::None } else { PayloadKind::Buffer };
        // The transform and the direction only matter where keys are read from or written to
        // the user's side.
        let (transform, descending) = if first || last { (spec.transform, spec.descending) } else { (KeyTransform::Identity, false) };
        let count_pass = RadixPass { kind: RadixKind::Count { ballot }, shift, transform, descending };

        let store_keys = dst_keys.is_some();
        let scatter_pass = RadixPass { kind: RadixKind::Scatter { last, store_keys, payload: kind, ballot }, shift, transform, descending };
        let mut writes = Vec::with_capacity(2);
        if let Some(buffer) = dst_keys {
            writes.push(RawArg::write(buffer, bytes));
        }
        if kind != PayloadKind::None {
            writes.push(RawArg::write(dst_pay.expect("a payload sort has payload buffers"), bytes));
        }
        let mut reads = Vec::with_capacity(2);
        if kind == PayloadKind::Buffer {
            reads.push(RawArg::read(src_pay.expect("a payload sort has payload buffers"), bytes));
        }
        reads.push(RawArg::read(&scanned, count_bytes));
        let count_args = [RawArg::write(&counts, count_bytes)];
        if let (true, Keys::Pending(node)) = (first, keys) {
            // One decision for both kernels of the pass: the scatter binds more buffers, so
            // fusing into the count kernel alone could strand the scatter.
            let bindings = writes.len() + reads.len();
            fused = jit::radix_pass(ctx, node, count_pass, params, count_grid, &count_args, &[], bindings)?;
            if !fused {
                key_pin = Some(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?.pin_single()?);
            }
        }
        if !(first && fused) {
            let src_keys = if first { key_pin.as_ref().expect("the first pass has keys").buffer() } else { &mid_keys[(pass as usize - 1) % 2] };
            let count = ctx.builtin(Builtin::Radix { pass: count_pass, first }, "u32")?;
            ctx.launch(&count, count_grid, [n32, 1, 1], &[count_args[0], RawArg::read(src_keys, bytes), RawArg::Uniform(bytemuck::bytes_of(&params))])?;
        }
        lines
            .scan(Bound { buffer: &counts, bytes: count_bytes }, count_words as u32, 1, 1, Bound { buffer: &scanned, bytes: count_bytes })?;

        if let (true, Keys::Pending(node), true) = (first, keys, fused) {
            if jit::radix_pass(ctx, node, scatter_pass, params, groups, &writes, &reads, writes.len() + reads.len())? {
                continue;
            }
            // Evaluated by someone else since the count kernel fused it: the counts stay valid,
            // so the buffer kernel scatters.
            key_pin = Some(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?.pin_single()?);
        }
        let src_keys = if first { key_pin.as_ref().expect("the first pass has keys").buffer() } else { &mid_keys[(pass as usize - 1) % 2] };
        let mut args = writes;
        args.push(RawArg::read(src_keys, bytes));
        args.extend(reads);
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        let scatter = ctx.builtin(Builtin::Radix { pass: scatter_pass, first }, "u32")?;
        ctx.launch(&scatter, groups, [n32, 1, 1], &args)?;
    }
    drop((lines, scratch));
    Ok(Sorted { keys: out_keys, payload: out_pay })
}

/// Whether a sort runs as the single sweep (see [`radix_sweep`]): the device orders memory
/// across workgroups (the look-back of the scan needs the same), ranks with ballots, and every
/// count fits the 30 bits of a status word.
pub(crate) fn use_sweep(ctx: &Context, n: u32) -> bool {
    let tuning = ctx.tuning();
    tuning.lookback_scan && use_ballots(tuning) && sweep_bytes(16) <= tuning.workgroup_storage && n < 1 << 30
}

/// The sort as CUB's onesweep does it (Adinets and Merrill 2022): one pass over the keys counts
/// the digits of every radix pass ([`RadixKind::Histogram`]), then each pass is one kernel
/// ([`RadixKind::Sweep`]) that ranks its tile of keys, finds where the tile's keys of every
/// digit go from the statuses of the tiles before it (decoupled look-back: a tile publishes its
/// counts, then walks back over the earlier tiles' until one has the inclusive prefix) and
/// scatters. That is `2 + digits` reads or writes of the keys per element less than the three
/// kernels per pass of [`radix_sort`]'s other path (count, scan of the counts, scatter).
///
/// The work buffer holds, in words: the counters of the digits of every pass (written by the
/// histogram, read by the sweeps), the ticket counters of the passes, and the statuses, `tiles *
/// 256` words per pass. It is zeroed by one launch ahead of the kernels.
fn radix_sweep(ctx: &Context, keys: Keys<'_>, payload: Payload<'_>, spec: Spec, n32: u32, params: SortParams) -> Outcome<Sorted> {
    let n = n32 as usize;
    let per_thread = params.per_thread;
    let tiles = n32.div_ceil(params.tile);
    let passes = params.passes;
    let bytes = n as u64 * 4;
    let work_words = u64::from(SWEEP_HEADER) + u64::from(passes) * u64::from(tiles) * u64::from(DIGITS);
    let work_bytes = work_words * 4;
    let hist_groups = n32.div_ceil(WORKGROUP * HIST_PER_THREAD).min(ctx.tuning().compute_units.max(1) * 4).max(1);

    let mut key_pin = match keys {
        Keys::Ready(storage) => Some(storage.pin_single()?),
        Keys::Pending(_) => {
            if bytes > ctx.segment_bytes() {
                return Err(LumaError::Chunked("a sort of an array split over several buffers").into());
            }
            None
        }
    };
    let input_pay = match payload {
        Payload::Buffer(storage) => Some(storage.pin_single()?),
        _ => None,
    };
    let first_kind = match payload {
        Payload::None => PayloadKind::None,
        Payload::Index => PayloadKind::Index,
        Payload::Buffer(_) => PayloadKind::Buffer,
    };
    let out_keys = spec.want_keys.then(|| Arc::new(Storage::new(ctx, n, ElemLayout::plain(4))));
    let out_pay = (first_kind != PayloadKind::None).then(|| Arc::new(Storage::new(ctx, n, ElemLayout::plain(4))));
    let out_keys_pin = out_keys.as_ref().map(|storage| storage.pin_single()).transpose()?;
    let out_pay_pin = out_pay.as_ref().map(|storage| storage.pin_single()).transpose()?;

    let mut scratch = Scratch::new(ctx);
    let work = scratch.take(work_bytes)?;
    let mids = passes.saturating_sub(1).min(2) as usize;
    let mid_keys = (0..mids).map(|_| scratch.take(bytes)).collect::<std::result::Result<Vec<Buffer>, LumaError>>()?;
    let mid_pay = if first_kind == PayloadKind::None {
        Vec::new()
    } else {
        (0..mids).map(|_| scratch.take(bytes)).collect::<std::result::Result<Vec<Buffer>, LumaError>>()?
    };
    let zero = ctx.builtin(Builtin::Fill, "u32")?;
    let zeroed = u32::try_from(work_words).map_err(|_| LumaError::DimensionTooLarge { shape: [n, 1, 1, 1] })?;
    launch_1d(ctx, &zero, zeroed, &[RawArg::write(&work, work_bytes), RawArg::Uniform(bytemuck::bytes_of(&0u32))])?;

    let groups = fold_workgroups(ctx, tiles)?;
    let hist_grid = fold_workgroups(ctx, hist_groups)?;
    let hist_params = [n32, hist_groups, 0];
    let hist_pass = RadixPass { kind: RadixKind::Histogram { digits: passes as u8 }, shift: 0, transform: spec.transform, descending: spec.descending };
    let hist_writes = [RawArg::write(&work, work_bytes)];
    // Whether the first pass computes the keys (decided by the histogram kernel).
    let mut fused = false;
    let worst_bindings = 4;
    if let Keys::Pending(node) = keys {
        fused = jit::radix_pass(ctx, node, hist_pass, hist_params, hist_grid, &hist_writes, &[], worst_bindings)?;
        if !fused {
            key_pin = Some(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?.pin_single()?);
        }
    }
    if !fused {
        let src_keys = key_pin.as_ref().expect("the first pass has keys").buffer();
        let hist = ctx.builtin(Builtin::Radix { pass: hist_pass, first: true }, "u32")?;
        ctx.launch(&hist, hist_grid, [n32, 1, 1], &[hist_writes[0], RawArg::read(src_keys, bytes), RawArg::Uniform(bytemuck::bytes_of(&hist_params))])?;
    }

    for pass in 0..passes {
        let (first, last) = (pass == 0, pass + 1 == passes);
        let shift = (pass * DIGIT_BITS) as u8;
        let src_pay = if first { input_pay.as_ref().map(|pin| pin.buffer()) } else { mid_pay.get((pass as usize - 1) % 2) };
        let dst_keys = if last { out_keys_pin.as_ref().map(|pin| pin.buffer()) } else { Some(&mid_keys[pass as usize % 2]) };
        let dst_pay = if last { out_pay_pin.as_ref().map(|pin| pin.buffer()) } else { mid_pay.get(pass as usize % 2) };
        let kind = if first { first_kind } else if first_kind == PayloadKind::None { PayloadKind::None } else { PayloadKind::Buffer };
        let (transform, descending) = if first || last { (spec.transform, spec.descending) } else { (KeyTransform::Identity, false) };
        let store_keys = dst_keys.is_some();
        let sweep_pass = RadixPass { kind: RadixKind::Sweep { last, store_keys, payload: kind, per_thread: per_thread as u8 }, shift, transform, descending };
        let params = [n32, tiles, SWEEP_HEADER + pass * tiles * DIGITS];
        let mut writes = Vec::with_capacity(3);
        if let Some(buffer) = dst_keys {
            writes.push(RawArg::write(buffer, bytes));
        }
        if kind != PayloadKind::None {
            writes.push(RawArg::write(dst_pay.expect("a payload sort has payload buffers"), bytes));
        }
        writes.push(RawArg::write(&work, work_bytes));
        let mut reads = Vec::with_capacity(1);
        if kind == PayloadKind::Buffer {
            reads.push(RawArg::read(src_pay.expect("a payload sort has payload buffers"), bytes));
        }
        if let (true, Keys::Pending(node), true) = (first, keys, fused) {
            if jit::radix_pass(ctx, node, sweep_pass, params, groups, &writes, &reads, worst_bindings)? {
                continue;
            }
            // Evaluated by someone else since the histogram kernel fused it: the counts stay
            // valid, so the buffer kernel sweeps.
            key_pin = Some(node.force().as_ref().map_err(|error| Failure::Inherited(Arc::clone(error)))?.pin_single()?);
        }
        let src_keys = if first { key_pin.as_ref().expect("the first pass has keys").buffer() } else { &mid_keys[(pass as usize - 1) % 2] };
        let mut args = writes;
        args.push(RawArg::read(src_keys, bytes));
        args.extend(reads);
        args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
        let sweep = ctx.builtin(Builtin::Radix { pass: sweep_pass, first }, "u32")?;
        ctx.launch(&sweep, groups, [n32, 1, 1], &args)?;
    }
    drop(scratch);
    Ok(Sorted { keys: out_keys, payload: out_pay })
}

impl<T: SortKey> Array<T> {
    /// The elements in ascending order, with the shape of this array (flat order; for one
    /// dimension per axis see [`Array::sort_axis`]). The sort is stable and runs as a radix sort
    /// over the bits of the key, a few kernels per 8 bits. A pending expression is not evaluated
    /// first: the kernels of the first pass compute the keys themselves (unless it reads
    /// circular shifts of other expressions). The result is recorded, not waited for.
    ///
    /// Floats order as their bits do after the usual total-order transform: `-inf < .. < -0.0 <
    /// +0.0 < .. < +inf < NaN`, so NaNs come last, and the sign of a NaN is not kept. Arrays
    /// split over several buffers fail with [`LumaError::Chunked`].
    #[track_caller]
    pub fn sort(&self) -> Array<T> {
        self.sorted(false, Location::caller())
    }

    /// The elements in descending order, the exact reverse of [`Array::sort`] by key: equal keys
    /// keep their input order (so `+0.0` comes before `-0.0`), NaNs come first.
    #[track_caller]
    pub fn sort_descending(&self) -> Array<T> {
        self.sorted(true, Location::caller())
    }

    /// The flat indices of the elements in ascending order, with the shape of this array: a
    /// stable sort, so equal elements keep their input order. Order of floats as for
    /// [`Array::sort`].
    #[track_caller]
    pub fn argsort(&self) -> Array<u32> {
        self.argsorted(false, "argsort", Location::caller())
    }

    /// The flat indices of the elements in descending order; ties in input order, as in
    /// [`Array::sort_descending`].
    #[track_caller]
    pub fn argsort_descending(&self) -> Array<u32> {
        self.argsorted(true, "argsort_descending", Location::caller())
    }

    /// Sorts every line along `axis` independently (axis 0 varies fastest), keeping the shape.
    /// Order of floats as for [`Array::sort`].
    ///
    /// Sorts everything by value carrying each element's index, then sorts that stably by the line
    /// of the index (see `sort_lines`), and for an axis that is not the fastest moves the lines
    /// back to their place by a tiled transpose. NaNs come out as the positive quiet NaN.
    #[track_caller]
    pub fn sort_axis(&self, axis: usize) -> Array<T> {
        let location = Location::caller();
        let result = self.run_sort_axis(axis);
        Array::resolve(self.context(), self.shape(), result, "sort_axis", location)
    }

    fn sorted(&self, descending: bool, location: &'static Location<'static>) -> Array<T> {
        let name = if descending { "sort_descending" } else { "sort" };
        Array::resolve(self.context(), self.shape(), self.run_sort(descending), name, location)
    }

    fn argsorted(&self, descending: bool, name: &'static str, location: &'static Location<'static>) -> Array<u32> {
        let result = self.run_argsort(descending).map(|storage| Array::from_storage(self.context(), self.shape(), storage));
        Array::resolve(self.context(), self.shape(), result, name, location)
    }

    fn run_sort(&self, descending: bool) -> Outcome<Array<T>> {
        let ctx = self.context();
        if self.len() <= 1 {
            self.storage()?;
            return Ok(self.clone());
        }
        if let Some((boundary, params)) = host_candidate(ctx, "sort", Location::caller(), self.len(), self.error().is_some())
            && let Some(sorted) = crate::host::try_sort(self, descending)
        {
            note(ctx, &boundary, StrategyId::SORT_HOST, &params, &[]);
            return Ok(sorted);
        }
        let spec = Spec { transform: T::TRANSFORM, descending, key_bits: 32, want_keys: true };
        let sorted = self.radix(Payload::None, spec)?;
        Ok(Array::from_storage(ctx, self.shape(), sorted.keys.expect("keys were asked for")))
    }

    /// Sorts the elements as `u32` words by `spec`, carrying `payload`: a lazy array is not
    /// evaluated first, its first pass computes the keys (see [`Keys::Pending`]).
    fn radix(&self, payload: Payload<'_>, spec: Spec) -> Outcome<Sorted> {
        let ctx = self.context();
        let sorted = match self.pending() {
            Some(node) => radix_sort(ctx, Keys::Pending(node), payload, spec)?,
            None => radix_sort(ctx, Keys::Ready(self.storage()?), payload, spec)?,
        };
        Ok(sorted)
    }

    /// The storage of the flat indices of the sorted elements.
    fn run_argsort(&self, descending: bool) -> Outcome<Arc<Storage>> {
        let ctx = self.context();
        if self.len() <= 1 {
            self.storage()?;
            return Ok(Array::<u32>::zeros_on(ctx, self.len())?.storage()?.clone());
        }
        if let Some((boundary, params)) = host_candidate(ctx, "argsort", Location::caller(), self.len(), self.error().is_some())
            && let Some(storage) = crate::host::try_argsort(self, descending)
        {
            note(ctx, &boundary, StrategyId::SORT_HOST, &params, &[]);
            return Ok(storage);
        }
        let spec = Spec { transform: T::TRANSFORM, descending, key_bits: 32, want_keys: false };
        let sorted = self.radix(Payload::Index, spec)?;
        Ok(sorted.payload.expect("the payload was asked for"))
    }

    /// The sorted elements and the flat indices that order them (one stable sort).
    pub(crate) fn sort_with_indices(&self) -> crate::error::Result<(Array<T>, Array<u32>)> {
        self.run_sort_with_indices().map_err(Failure::into_error)
    }

    fn run_sort_with_indices(&self) -> Outcome<(Array<T>, Array<u32>)> {
        let ctx = self.context();
        if self.len() <= 1 {
            self.storage()?;
            return Ok((self.clone(), Array::<u32>::zeros_on(ctx, self.shape())?));
        }
        if let Some((boundary, params)) = host_candidate(ctx, "sort_with_indices", Location::caller(), self.len(), self.error().is_some())
            && let Some(pair) = crate::host::try_sort_with_indices(self)
        {
            note(ctx, &boundary, StrategyId::SORT_HOST, &params, &[]);
            return Ok(pair);
        }
        let spec = Spec { transform: T::TRANSFORM, descending: false, key_bits: 32, want_keys: true };
        let sorted = self.radix(Payload::Index, spec)?;
        let keys = Array::from_storage(ctx, self.shape(), sorted.keys.expect("keys were asked for"));
        let perm = Array::from_storage(ctx, self.shape(), sorted.payload.expect("the payload was asked for"));
        Ok((keys, perm))
    }

    /// Every line along `axis` sorted, the lines one after the other: a flat array of `len`
    /// elements in which line `l` (`l = s + o * stride`, `s` below and `o` above the axis) is
    /// elements `l * n ..` `(l + 1) * n`, ascending. `axis` must be below 4 and the extent
    /// along it above 1.
    ///
    /// One stable sort of everything by value, carrying the flat input index, then a stable sort of
    /// that by the line of the index (a few bits, so one to three passes), carrying the values:
    /// the second sort groups the lines and keeps each in order, and nothing is gathered or
    /// scattered with a stride.
    pub(crate) fn sort_lines(&self, axis: usize) -> Outcome<Array<T>> {
        let ctx = self.context();
        let dims = self.shape().dims();
        let (len, n) = (self.len(), dims[axis]);
        let lines = len / n;
        let flat = self.reshape(len);
        if lines == 1 {
            return flat.run_sort(false);
        }
        let stride: usize = dims[..axis].iter().product();
        let (keys, perm) = flat.run_sort_with_indices()?;
        let (stride, n, lines32) = (stride as u32, n as u32, lines as u32);
        // The line of every element in the order of the sort by value.
        let line = (&perm % stride) + (&perm / (stride * n)) * stride;
        let bits = u32::BITS - (lines32 - 1).leading_zeros();
        let spec = Spec { transform: KeyTransform::Identity, descending: false, key_bits: bits, want_keys: false };
        let by_line = line.radix(Payload::Buffer(keys.storage()?), spec)?;
        Ok(Array::from_storage(ctx, Shape::from(len), by_line.payload.expect("the payload was asked for")))
    }

    /// Every line of `n` (at most [`crate::segsort::max_line`]) elements `stride` apart sorted in
    /// place of its own, by one kernel (see [`crate::segsort`]).
    fn sort_short_lines(&self, stride: usize, n: usize) -> Outcome<Array<T>> {
        let ctx = self.context();
        let len = self.len();
        let too_large = || LumaError::DimensionTooLarge { shape: self.shape().dims() };
        let job = LineJob { n: n as u32, stride: u32::try_from(stride).map_err(|_| too_large())?, lines: u32::try_from(len / n).map_err(|_| too_large())? };
        u32::try_from(len).map_err(|_| too_large())?;
        let bytes = len as u64 * 4;
        let input = self.storage()?.pin_single()?;
        let out = Arc::new(Storage::new(ctx, len, ElemLayout::plain(4)));
        let out_pin = out.pin_single()?;
        LineSort::new(T::TRANSFORM, job.n, job.stride).launch(ctx, job, RawArg::write(out_pin.buffer(), bytes), RawArg::read(input.buffer(), bytes))?;
        drop((input, out_pin));
        Ok(Array::from_storage(ctx, self.shape(), out))
    }

    /// Sorts the elements of this array (the keys) in ascending order and moves the elements of
    /// `values`, which has the same shape, along with them: `(keys, values)` sorted, flat order
    /// (for one dimension per axis see [`Array::sort_by_key_axis`]). Stable: equal keys keep
    /// their input order, so their values do too. Order of floats as for [`Array::sort`]
    /// (`-inf < .. < -0.0 < +0.0 < .. < +inf < NaN`; the sign and payload of a NaN key are not
    /// kept).
    ///
    /// `values` may have any element type. A `u32`, `i32` or `f32` array travels through the
    /// radix passes with its keys (one pass moves both); other types are gathered by the
    /// permutation of an [`Array::argsort`]. A pending `self` is not evaluated first, as for
    /// [`Array::sort`]; `values` is evaluated. Errors (different shapes or contexts, an array
    /// split over several buffers) poison both results.
    #[track_caller]
    pub fn sort_by_key<V: ScalarType>(&self, values: &Array<V>) -> (Array<T>, Array<V>) {
        self.sorted_by_key(values, false, "sort_by_key", Location::caller())
    }

    /// [`Array::sort_by_key`] with the keys in descending order, the exact reverse by key: equal
    /// keys keep their input order, NaNs come first.
    #[track_caller]
    pub fn sort_by_key_descending<V: ScalarType>(&self, values: &Array<V>) -> (Array<T>, Array<V>) {
        self.sorted_by_key(values, true, "sort_by_key_descending", Location::caller())
    }

    /// [`Array::sort_by_key`] for every line along `axis` independently (axis 0 varies fastest):
    /// the keys of each line in ascending order and the values of `values` (the same shape)
    /// carried along, keeping the shape.
    #[track_caller]
    pub fn sort_by_key_axis<V: ScalarType>(&self, axis: usize, values: &Array<V>) -> (Array<T>, Array<V>) {
        let location = Location::caller();
        let result = self.run_sort_by_key_axis(axis, values, false);
        self.resolve_pair(values, result, "sort_by_key_axis", location)
    }

    /// [`Array::sort_by_key_axis`] with every line in descending key order (as
    /// [`Array::sort_by_key_descending`]).
    #[track_caller]
    pub fn sort_by_key_axis_descending<V: ScalarType>(&self, axis: usize, values: &Array<V>) -> (Array<T>, Array<V>) {
        let location = Location::caller();
        let result = self.run_sort_by_key_axis(axis, values, true);
        self.resolve_pair(values, result, "sort_by_key_axis_descending", location)
    }

    fn sorted_by_key<V: ScalarType>(&self, values: &Array<V>, descending: bool, name: &'static str, location: &'static Location<'static>) -> (Array<T>, Array<V>) {
        let result = self.run_sort_by_key(values, descending);
        self.resolve_pair(values, result, name, location)
    }

    /// The two arrays of a pair result, or both poisoned.
    fn resolve_pair<V: ScalarType>(&self, values: &Array<V>, result: Outcome<(Array<T>, Array<V>)>, op: &'static str, location: &'static Location<'static>) -> (Array<T>, Array<V>) {
        match result {
            Ok(pair) => pair,
            Err(failure) => {
                let error = match failure {
                    Failure::Inherited(error) => error,
                    Failure::New(error) => Arc::new(LumaError::Op { op, location, source: Arc::new(error) }),
                };
                let ctx = self.context();
                (Array::poisoned(ctx, self.shape(), Arc::clone(&error)), Array::poisoned(ctx, values.shape(), error))
            }
        }
    }

    fn check_pair<V: ScalarType>(&self, values: &Array<V>) -> Outcome<()> {
        if !self.context().same_as(values.context()) {
            return Err(LumaError::ContextMismatch.into());
        }
        if self.shape() != values.shape() {
            return Err(LumaError::ShapeMismatch { lhs: self.shape().dims(), rhs: values.shape().dims() }.into());
        }
        Ok(())
    }

    fn run_sort_by_key<V: ScalarType>(&self, values: &Array<V>, descending: bool) -> Outcome<(Array<T>, Array<V>)> {
        self.run_sort_by_key_bits(values, descending, 32)
    }

    /// [`Array::sort_by_key`] of keys that differ only in their low `key_bits` bits (as transformed
    /// for the sort: for `u32` the key itself): one radix pass fewer per 8 bits that are known to
    /// be zero in every key. The sparse conversions know their indices are below the dimension.
    pub(crate) fn sort_by_key_bits<V: ScalarType>(&self, values: &Array<V>, key_bits: u32) -> (Array<T>, Array<V>) {
        let result = self.run_sort_by_key_bits(values, false, key_bits);
        self.resolve_pair(values, result, "sort_by_key", Location::caller())
    }

    fn run_sort_by_key_bits<V: ScalarType>(&self, values: &Array<V>, descending: bool, key_bits: u32) -> Outcome<(Array<T>, Array<V>)> {
        let ctx = self.context();
        self.check_pair(values)?;
        if self.len() <= 1 {
            self.storage()?;
            values.storage()?;
            return Ok((self.clone(), values.clone()));
        }
        let boundary = Boundary::new(
            FamilyTag::Sort,
            "sort_by_key",
            Location::caller(),
            [dim(self.len()), 1, 1, 0],
            BoundaryAttrs::Sort(SortAttrs { stage: SortStage::ByKey, payload: SortPayload::Buffer, errored: false, value4: V::LAYOUT == ElemLayout::plain(4), key_bits: key_bits.clamp(1, 32) as u8, len: self.len() as u64 }),
        );
        let (chosen, params) = candidates(ctx, &boundary).next().expect("the permutation sort always fits");
        note(ctx, &boundary, chosen, &params, &[]);
        if chosen == StrategyId::SORT_BY_KEY_PAYLOAD {
            let spec = Spec { transform: T::TRANSFORM, descending, key_bits, want_keys: true };
            let sorted = self.radix(Payload::Buffer(values.storage()?), spec)?;
            let keys = Array::from_storage(ctx, self.shape(), sorted.keys.expect("keys were asked for"));
            let moved = Array::from_storage(ctx, self.shape(), sorted.payload.expect("the payload was asked for"));
            return Ok((keys, moved));
        }
        let (keys, perm) = if descending {
            let perm = Array::<u32>::from_storage(ctx, self.shape(), self.run_argsort(true)?);
            (self.take(&perm), perm)
        } else {
            self.run_sort_with_indices()?
        };
        Ok((keys, values.take(&perm)))
    }

    fn run_sort_by_key_axis<V: ScalarType>(&self, axis: usize, values: &Array<V>, descending: bool) -> Outcome<(Array<T>, Array<V>)> {
        let ctx = self.context();
        let shape = self.shape();
        let dims = shape.dims();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        self.check_pair(values)?;
        let n = dims[axis];
        if n == 1 {
            self.storage()?;
            values.storage()?;
            return Ok((self.clone(), values.clone()));
        }
        let len = self.len();
        let flat = self.reshape(len);
        let flat_values = values.reshape(len);
        let boundary = Boundary::new(
            FamilyTag::Sort,
            "sort_by_key_axis",
            Location::caller(),
            [dim(n), dim(len / n), dim(len / n), 0],
            BoundaryAttrs::Sort(SortAttrs { stage: SortStage::ByKeyAxis, payload: SortPayload::None, errored: false, value4: false, key_bits: 32, len: len as u64 }),
        );
        // One line is the flat by-key sort, which selects for itself.
        let Some((chosen, params)) = candidates(ctx, &boundary).next() else {
            let (keys, moved) = flat.run_sort_by_key(&flat_values, descending)?;
            return Ok((keys.reshape(shape), moved.reshape(shape)));
        };
        note(ctx, &boundary, chosen, &params, &[]);
        // Sort everything by key carrying the flat index, then sort that stably by the line of
        // the index carrying the index again (see `sort_lines`): the result is, line by line, the
        // input positions in key order; keys and values are gathered from them.
        let stride: usize = dims[..axis].iter().product();
        let lines = len / n;
        let perm = Array::<u32>::from_storage(ctx, Shape::from(len), flat.run_argsort(descending)?);
        let (stride32, n32, lines32) = (stride as u32, n as u32, lines as u32);
        let line = (&perm % stride32) + (&perm / (stride32 * n32)) * stride32;
        let bits = u32::BITS - (lines32 - 1).leading_zeros();
        let spec = Spec { transform: KeyTransform::Identity, descending: false, key_bits: bits, want_keys: false };
        let by_line = line.radix(Payload::Buffer(perm.storage()?), spec)?;
        let order = Array::<u32>::from_storage(ctx, Shape::from(len), by_line.payload.expect("the payload was asked for"));
        let (keys, moved) = (flat.take(&order), flat_values.take(&order));
        if stride == 1 {
            return Ok((keys.reshape(shape), moved.reshape(shape)));
        }
        let outer = len / (n * stride);
        let untranspose = |a: Array<T>| a.reshape([n, stride, outer]).swap_axes(0, 1).reshape(shape);
        let keys = untranspose(keys);
        keys.request().map_err(Failure::New)?;
        let moved = moved.reshape([n, stride, outer]).swap_axes(0, 1).reshape(shape);
        moved.request().map_err(Failure::New)?;
        Ok((keys, moved))
    }

    fn run_sort_axis(&self, axis: usize) -> Outcome<Array<T>> {
        let shape = self.shape();
        let dims = shape.dims();
        if axis >= 4 {
            return Err(LumaError::InvalidShape(format!("axis {axis} of shape {shape}")).into());
        }
        self.storage()?;
        let n = dims[axis];
        if n == 1 {
            return Ok(self.clone());
        }
        let stride: usize = dims[..axis].iter().product();
        let boundary = Boundary::new(
            FamilyTag::Sort,
            "sort_axis",
            Location::caller(),
            [dim(n), dim(stride), dim(self.len() / n), 0],
            BoundaryAttrs::Sort(SortAttrs { stage: SortStage::Axis, payload: SortPayload::None, errored: false, value4: false, key_bits: 32, len: self.len() as u64 }),
        );
        let (chosen, params) = candidates(self.context(), &boundary).next().expect("a sort along an axis always fits");
        note(self.context(), &boundary, chosen, &params, &[]);
        if chosen == StrategyId::SORT_LINES {
            return self.sort_short_lines(stride, n);
        }
        let sorted = self.sort_lines(axis)?;
        if stride == 1 {
            return Ok(sorted.reshape(shape));
        }
        // Line `s + o * stride` is `[n]` at `o * stride * n + s * n`: per `o` a matrix of
        // `stride` rows of `n` that goes to `n` rows of `stride`, a tiled transpose.
        let outer = self.len() / (n * stride);
        let sorted = sorted.reshape([n, stride, outer]).swap_axes(0, 1).reshape(shape);
        sorted.request().map_err(Failure::New)?;
        Ok(sorted)
    }
}
