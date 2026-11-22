//! The summed-area table in three launches (`Array::sat`).
//!
//! A table is a running sum along axis 1 followed by one along axis 0. Axis 1 is the strided one:
//! its lines are `n1` cells long but only `n0` of them exist, too few for a thread each to fill
//! the GPU, so every line is cut in chunks of [`CHUNK`] cells:
//!
//! 1. *chunks*: one invocation scans one chunk (neighbouring invocations on neighbouring lines,
//!    so every load and store is coalesced) and writes the chunk's sum;
//! 2. *carry*: one invocation per line turns the chunk sums into exclusive prefixes;
//! 3. *lines*: one workgroup per column of axis 1 adds the carry of its chunk to the cells of
//!    the line along axis 0 (contiguous) and scans the line in workgroup memory, 1024 cells at
//!    a time, carrying the total from one segment to the next.
//!
//! Together that reads and writes the image twice, the least a two-pass table can do.

/// Cells of a line along axis 1 that one invocation scans.
pub(crate) const CHUNK: u32 = 64;
/// Invocations per workgroup of the line scan; it scans `4 *` this many cells at a time.
const LINE_GROUP: u32 = 256;
/// Invocations per workgroup of the other two passes.
pub(crate) const GROUP: u32 = 256;

/// The passes of the table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Pass {
    Chunks,
    Carry,
    Lines,
}

/// Words of the uniform block: `{n0, n1, planes, chunks, total}` and padding.
pub(crate) const PARAM_WORDS: usize = 8;

/// The WGSL of a pass on elements of type `ty`.
pub(crate) fn source(pass: Pass, ty: &str) -> String {
    let header = "struct Params { n0: u32, n1: u32, planes: u32, chunks: u32, total: u32, _a: u32, _b: u32, _c: u32 }";
    match pass {
        // Arguments: out, src, sums (`[n0, chunks, planes]`), uniform. `total` is the lines
        // (`n0 * chunks * planes`) the pass covers.
        Pass::Chunks => format!(
            "{header}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> src: array<{ty}>;
@group(1) @binding(2) var<storage, read_write> sums: array<{ty}>;
@group(1) @binding(3) var<uniform> p: Params;
@compute @workgroup_size({GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let t = gid.x + gid.y * nwg.x * {GROUP}u;
    if (t >= p.total) {{ return; }}
    let i = t % p.n0;
    let rest = t / p.n0;
    let c = rest % p.chunks;
    let plane = rest / p.chunks;
    let j0 = c * {CHUNK}u;
    let j1 = min(j0 + {CHUNK}u, p.n1);
    let base = plane * p.n0 * p.n1 + i;
    var acc = {ty}();
    for (var j = j0; j < j1; j++) {{
        let at = base + j * p.n0;
        acc = acc + src[at];
        out[at] = acc;
    }}
    sums[t] = acc;
}}
"
        ),
        // Arguments: sums, uniform. Turns the chunk sums of every line into exclusive prefixes.
        Pass::Carry => format!(
            "{header}
@group(1) @binding(0) var<storage, read_write> sums: array<{ty}>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size({GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let t = gid.x + gid.y * nwg.x * {GROUP}u;
    if (t >= p.total) {{ return; }}
    let i = t % p.n0;
    let plane = t / p.n0;
    var run = {ty}();
    for (var c = 0u; c < p.chunks; c++) {{
        let at = i + p.n0 * (c + p.chunks * plane);
        let v = sums[at];
        sums[at] = run;
        run = run + v;
    }}
}}
"
        ),
        // Arguments: out, sums (exclusive prefixes), uniform. `out` holds the chunk scans and is
        // updated in place. `total` is the lines (`n1 * planes`).
        Pass::Lines => {
            let span = 4 * LINE_GROUP;
            format!(
                "{header}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> sums: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> seg: array<{ty}, {span_pad}>;
var<workgroup> scan: array<{ty}, {LINE_GROUP}>;
var<workgroup> carry: {ty};
@compute @workgroup_size({LINE_GROUP})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let line = wid.x + wid.y * nwg.x;
    if (line >= p.total) {{ return; }}
    let j = line % p.n1;
    let plane = line / p.n1;
    let c = j / {CHUNK}u;
    let base = plane * p.n0 * p.n1 + j * p.n0;
    let sums_base = p.n0 * (c + p.chunks * plane);
    if (t == 0u) {{ carry = {ty}(); }}
    workgroupBarrier();
    for (var s0 = 0u; s0 < p.n0; s0 += {span}u) {{
        // Load coalesced, with the carry of the chunk added.
        for (var r = 0u; r < 4u; r++) {{
            let k = t + r * {LINE_GROUP}u;
            let i = s0 + k;
            var v = {ty}();
            if (i < p.n0) {{ v = out[base + i] + sums[sums_base + i]; }}
            seg[k + (k >> 5u)] = v;
        }}
        workgroupBarrier();
        // Each invocation scans four consecutive cells.
        let k0 = 4u * t;
        let a0 = seg[k0 + (k0 >> 5u)];
        let a1 = a0 + seg[k0 + 1u + ((k0 + 1u) >> 5u)];
        let a2 = a1 + seg[k0 + 2u + ((k0 + 2u) >> 5u)];
        let a3 = a2 + seg[k0 + 3u + ((k0 + 3u) >> 5u)];
        scan[t] = a3;
        workgroupBarrier();
        // Inclusive scan of the invocations' sums.
        for (var d = 1u; d < {LINE_GROUP}u; d = d << 1u) {{
            var add = {ty}();
            if (t >= d) {{ add = scan[t - d]; }}
            workgroupBarrier();
            scan[t] = scan[t] + add;
            workgroupBarrier();
        }}
        var before = carry;
        if (t > 0u) {{ before = before + scan[t - 1u]; }}
        seg[k0 + (k0 >> 5u)] = before + a0;
        seg[k0 + 1u + ((k0 + 1u) >> 5u)] = before + a1;
        seg[k0 + 2u + ((k0 + 2u) >> 5u)] = before + a2;
        seg[k0 + 3u + ((k0 + 3u) >> 5u)] = before + a3;
        workgroupBarrier();
        for (var r = 0u; r < 4u; r++) {{
            let k = t + r * {LINE_GROUP}u;
            let i = s0 + k;
            if (i < p.n0) {{ out[base + i] = seg[k + (k >> 5u)]; }}
        }}
        workgroupBarrier();
        if (t == {last}u) {{ carry = carry + scan[{last}u]; }}
        workgroupBarrier();
    }}
}}
",
                span_pad = span + span / 32,
                last = LINE_GROUP - 1,
            )
        }
    }
}
