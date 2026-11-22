//! The strongest `k` corners in descending order of score, in one workgroup.
//!
//! A radix select over the bits of the (positive) scores finds the `k`-th largest one in four
//! passes of a 256-bin histogram; the corners above it and the first of those equal to it (in
//! input order, so that ties rank like a stable sort) are listed by a scan; the list of at most
//! 1024 entries is sorted by a bitonic network in workgroup memory and the corners gathered. It
//! replaces a radix sort of every corner (about twenty launches) for the usual `k` of a few
//! hundred; larger `k` or more than [`MAX_CORNERS`] corners use [`Array::topk`].

use std::fmt::Write as _;

use super::Kernel as Corners;
use crate::array::Array;
use crate::builtin::Builtin;
use crate::context::RawArg;
use crate::ops::Outcome;
use crate::shape::Shape;
use crate::vision::{to_u32, Kernel as Any};

/// Largest `k` of the kernel.
pub(crate) const MAX_K: usize = 1024;
/// Most corners it selects from.
pub(crate) const MAX_CORNERS: usize = 1 << 20;

const THREADS: u32 = 256;

/// One kernel: `padded`, the power of two at least `k`, is a literal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kernel {
    pub(crate) padded: u16,
}

/// The WGSL. Arguments: scores, `x`, `y` (read), the selected scores, `x`, `y` (write), uniform
/// `{count, k, chunk, c}`.
pub(crate) fn source(k: Kernel) -> String {
    let padded = u32::from(k.padded);
    let mut s = String::new();
    s.push_str("struct Params { count: u32, k: u32, chunk: u32, c: u32 }\n");
    for (i, (name, rw)) in [("score", "read"), ("xs", "read"), ("ys", "read"), ("out_s", "read_write"), ("out_x", "read_write"), ("out_y", "read_write")].iter().enumerate() {
        let _ = writeln!(s, "@group(1) @binding({i}) var<storage, {rw}> {name}: array<f32>;");
    }
    s.push_str("@group(1) @binding(6) var<uniform> p: Params;\n");
    let _ = write!(
        s,
        "var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> cg: array<u32, {THREADS}>;
var<workgroup> ce: array<u32, {THREADS}>;
var<workgroup> sel_key: array<u32, {padded}>;
var<workgroup> sel_idx: array<u32, {padded}>;
var<workgroup> shared_prefix: u32;
var<workgroup> shared_rem: u32;
fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {{
    return ka > kb || (ka == kb && ia < ib);
}}
@compute @workgroup_size({THREADS})
fn main(@builtin(local_invocation_index) t: u32) {{
    var prefix = 0u;
    var rem = p.k;
"
    );
    for pass in 0..4u32 {
        let shift = 24 - 8 * pass;
        let matches = if pass == 0 { "true".to_string() } else { format!("(key >> {}u) == prefix", shift + 8) };
        let _ = write!(
            s,
            "    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    for (var i = t; i < p.count; i += {THREADS}u) {{
        let key = bitcast<u32>(score[i]);
        if ({matches}) {{ atomicAdd(&hist[(key >> {shift}u) & 255u], 1u); }}
    }}
    workgroupBarrier();
    if (t == 0u) {{
        var cum = 0u;
        var d = 255u;
        loop {{
            let c = atomicLoad(&hist[d]);
            if (cum + c >= rem || d == 0u) {{ break; }}
            cum += c;
            d -= 1u;
        }}
        shared_prefix = (prefix << 8u) | d;
        shared_rem = rem - cum;
    }}
    workgroupBarrier();
    prefix = shared_prefix;
    rem = shared_rem;
    workgroupBarrier();
"
        );
    }
    let _ = write!(
        s,
        "    // prefix is the k-th largest key, rem the corners equal to it that are taken.
    let first = t * p.chunk;
    let last = min(first + p.chunk, p.count);
    var ng = 0u;
    var ne = 0u;
    for (var i = first; i < last; i++) {{
        let key = bitcast<u32>(score[i]);
        ng += select(0u, 1u, key > prefix);
        ne += select(0u, 1u, key == prefix);
    }}
    cg[t] = ng;
    ce[t] = ne;
    workgroupBarrier();
    for (var d = 1u; d < {THREADS}u; d = d << 1u) {{
        var ag = 0u;
        var ae = 0u;
        if (t >= d) {{ ag = cg[t - d]; ae = ce[t - d]; }}
        workgroupBarrier();
        cg[t] = cg[t] + ag;
        ce[t] = ce[t] + ae;
        workgroupBarrier();
    }}
    var gslot = cg[t] - ng;
    var eslot = ce[t] - ne;
    let gt_total = p.k - rem;
    for (var i = first; i < last; i++) {{
        let key = bitcast<u32>(score[i]);
        if (key > prefix) {{
            sel_key[gslot] = key;
            sel_idx[gslot] = i;
            gslot += 1u;
        }} else if (key == prefix) {{
            if (eslot < rem) {{
                sel_key[gt_total + eslot] = key;
                sel_idx[gt_total + eslot] = i;
            }}
            eslot += 1u;
        }}
    }}
    for (var i = p.k + t; i < {padded}u; i += {THREADS}u) {{
        sel_key[i] = 0u;
        sel_idx[i] = 0xffffffffu;
    }}
    workgroupBarrier();
    for (var kk = 2u; kk <= {padded}u; kk = kk << 1u) {{
        for (var j = kk >> 1u; j > 0u; j = j >> 1u) {{
            for (var i = t; i < {padded}u; i += {THREADS}u) {{
                let o = i ^ j;
                if (o > i) {{
                    let ascending = (i & kk) == 0u;
                    let a_first = before(sel_key[i], sel_idx[i], sel_key[o], sel_idx[o]);
                    // `ascending` lists the entries that come first at the lower index.
                    if (a_first != ascending) {{
                        let tk = sel_key[i];
                        let ti = sel_idx[i];
                        sel_key[i] = sel_key[o];
                        sel_idx[i] = sel_idx[o];
                        sel_key[o] = tk;
                        sel_idx[o] = ti;
                    }}
                }}
            }}
            workgroupBarrier();
        }}
    }}
    for (var i = t; i < p.k; i += {THREADS}u) {{
        let idx = sel_idx[i];
        out_s[i] = score[idx];
        out_x[i] = xs[idx];
        out_y[i] = ys[idx];
    }}
}}
"
    );
    s
}

/// `(x, y, score)` of the selected corners.
pub(crate) type Selected = (Array<f32>, Array<f32>, Array<f32>);

/// The `k` strongest of the `count` corners `(x, y, score)` (all of one length, `count > k`),
/// strongest first; ties in input order. `None` if `k` or `count` is out of the range of the
/// kernel.
pub(crate) fn strongest(x: &Array<f32>, y: &Array<f32>, score: &Array<f32>, k: usize) -> Outcome<Option<Selected>> {
    let count = score.len();
    if k == 0 || k > MAX_K || count > MAX_CORNERS || count <= k {
        return Ok(None);
    }
    let ctx = score.context();
    let padded = k.next_power_of_two().max(2);
    let (out_s, out_x, out_y) = (Array::<f32>::alloc(ctx, Shape::from(k))?, Array::<f32>::alloc(ctx, Shape::from(k))?, Array::<f32>::alloc(ctx, Shape::from(k))?);
    let kernel = ctx.builtin(Builtin::Vision(Any::Corners(Corners::Select(Kernel { padded: padded as u16 }))), "f32")?;
    let chunk = to_u32(count.div_ceil(THREADS as usize), "chunk")?;
    let params = [to_u32(count, "count")?, to_u32(k, "k")?, chunk, 0u32];
    let (s_pin, x_pin, y_pin) = (score.ready()?, x.ready()?, y.ready()?);
    let (os_pin, ox_pin, oy_pin) = (out_s.ready()?, out_x.ready()?, out_y.ready()?);
    let args = [
        RawArg::read(s_pin.0.buffer(), s_pin.1),
        RawArg::read(x_pin.0.buffer(), x_pin.1),
        RawArg::read(y_pin.0.buffer(), y_pin.1),
        RawArg::write(os_pin.0.buffer(), os_pin.1),
        RawArg::write(ox_pin.0.buffer(), ox_pin.1),
        RawArg::write(oy_pin.0.buffer(), oy_pin.1),
        RawArg::Uniform(bytemuck::bytes_of(&params)),
    ];
    ctx.launch(&kernel, [1, 1, 1], [THREADS, 1, 1], &args)?;
    Ok(Some((out_x, out_y, out_s)))
}
