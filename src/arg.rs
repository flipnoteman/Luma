//! The kernels of `argmax` and `argmin`: the reducing passes of [`crate::scan`] (the same
//! decompositions of a line into chunks, blocks and short rows) carrying a (value, index) pair
//! instead of a value. A pass that leaves several partial results per line writes the values
//! (as `u32` bits) at `out[at]` and the indices at `out[lines * nb + at]`; the pass that leaves
//! one result per line writes the index alone.
//!
//! Order: for `argmax` the larger value wins and, among equal ones, the smaller index; `argmin` the
//! other way round. An `f32` NaN beats every number in both (NumPy returns the index of the first
//! NaN from `argmax` and from `argmin` alike), and of two NaNs the smaller index wins. The order is
//! total, so partial results combine in any order.

use std::fmt::Write;

use crate::builtin::ReduceOp;
use crate::scan::{AxisPass, Pass, REDUCE_CHUNK, ROWS_TILE, WORKGROUP};

/// `isn`, `bits`, `unbits` and `beats(x, j, v, i)`: whether the cell `x` at index `j` beats the
/// accumulator `v` at index `i`.
fn helpers(op: ReduceOp, ty: &str) -> String {
    let cmp = if op == ReduceOp::ArgMax { ">" } else { "<" };
    let (bits, unbits) = if ty == "u32" { ("v".to_owned(), "w".to_owned()) } else { ("bitcast<u32>(v)".to_owned(), format!("bitcast<{ty}>(w)")) };
    let mut src = format!("fn bits(v: {ty}) -> u32 {{ return {bits}; }}\nfn unbits(w: u32) -> {ty} {{ return {unbits}; }}\n");
    if ty == "f32" {
        src.push_str(&format!(
            "fn isn(x: f32) -> bool {{ return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }}
fn beats(x: f32, j: u32, v: f32, i: u32) -> bool {{
    let xn = isn(x);
    let vn = isn(v);
    return select(x {cmp} v || (x == v && j < i), xn && (!vn || j < i), xn || vn);
}}
"
        ));
    } else {
        src.push_str(&format!("fn beats(x: {ty}, j: u32, v: {ty}, i: u32) -> bool {{ return x {cmp} v || (x == v && j < i); }}\n"));
    }
    src
}

/// The kernel of `pass` (an arg reduction) over a buffer `a`: of `ty` cells, or of the `u32` words
/// of partial results (values as bits, then indices) when the pass is `partial`.
pub(crate) fn source(pass: AxisPass, ty: &str, workgroup: u32) -> String {
    let mut src = format!("struct Params {{ {}}}\n{}", crate::scan::AXIS_PARAMS, pass.output_bindings(ty));
    if pass.partial {
        src.push_str("@group(1) @binding(1) var<storage, read> a: array<u32>;\nfn ldw(q: u32) -> u32 { return a[q]; }\n");
    } else {
        let _ = writeln!(src, "@group(1) @binding(1) var<storage, read> a: array<{ty}>;\nfn ld(q: u32) -> {ty} {{ return a[q]; }}");
    }
    src.push_str("@group(1) @binding(2) var<uniform> p: Params;\n");
    src.push_str(&body(pass, ty, "ld", workgroup));
    src
}

/// The helper functions and the entry point of the arg-reduction `pass`, with the cells of the
/// operand from `{load}(q)` (a first pass) or from `ldw(q)` (a `partial` one). Needs `Params` (see
/// [`crate::scan::AXIS_PARAMS`]) as `p` and the output buffer.
pub(crate) fn body(pass: AxisPass, ty: &str, load: &str, workgroup: u32) -> String {
    let op = pass.ops.as_slice()[0];
    let mut src = String::from("fn line_base(l: u32) -> u32 { return (l % p.stride) + (l / p.stride) * p.stride * p.n; }\n");
    src.push_str(&helpers(op, ty));
    src.push_str(
        "fn put(at: u32, v: u32, i: u32) {
    if (p.nb == 1u) { out[at] = i; } else { out[at] = v; out[p.lines * p.nb + at] = i; }
}
",
    );
    let cin = if pass.partial { "    let cin = p.n * p.lines;\n" } else { "" };
    // The statements binding `{v}` and `{i}` to the cell at `q`, whose index along the line is `s`.
    let cell = |v: &str, i: &str, q: &str, s: &str| -> String {
        if pass.partial { format!("let {v} = unbits(ldw({q})); let {i} = ldw(cin + {q});") } else { format!("let {v} = {load}({q}); let {i} = {s};") }
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
    var accv = fv;
    var acci = fi;
    for (var s = s0 + 1u; s < s1; s++) {{
        {}
        if (beats(ev, ei, accv, acci)) {{ accv = ev; acci = ei; }}
    }}
    put({}, bits(accv), acci);
}}
",
                cell("fv", "fi", "base + s0 * p.stride", "s0"),
                cell("ev", "ei", "base + s * p.stride", "s"),
                if pass.line_major { "l * p.nb + b" } else { "i" }
            );
        }
        Pass::Rows { width, .. } => {
            let value = |q: &str| if pass.partial { format!("unbits(ldw({q}))") } else { format!("{load}({q})") };
            let index = |row: &str, s: &str| if pass.partial { format!("ldw(cin + {row} + {s})") } else { s.to_owned() };
            let _ = write!(
                src,
                "var<workgroup> tile: array<{ty}, {ROWS_TILE}>;
@compute @workgroup_size({width})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let first = (wg.x + wg.y * nwg.x) * p.nb;
    if (first >= p.lines) {{ return; }}
    let cnt = min(p.nb, p.lines - first);
    let pitch = p.n | 1u;
    let base = first * p.n;
    let cells = cnt * p.n;
{cin}    for (var j = t; j < cells; j += {width}u) {{
        let row = j / p.n;
        tile[row * pitch + (j - row * p.n)] = {};
    }}
    workgroupBarrier();
    if (t < cnt) {{
        let row = t * pitch;
        let cell = base + t * p.n;
        var accv = tile[row];
        var acci = {};
        for (var s = 1u; s < p.n; s++) {{
            let ev = tile[row + s];
            let ei = {};
            if (beats(ev, ei, accv, acci)) {{ accv = ev; acci = ei; }}
        }}
        out[first + t] = acci;
    }}
}}
",
                value("base + j"),
                index("cell", "0u"),
                index("cell", "s"),
            );
        }
        Pass::ChunkReduce => {
            let _ = write!(
                src,
                "var<workgroup> partv: array<{ty}, {w}>;
var<workgroup> parti: array<u32, {w}>;
@compute @workgroup_size({w})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let wid = wg.x + wg.y * nwg.x;
    if (wid >= p.lines * p.nb) {{ return; }}
    let l = wid / p.nb;
    let c = wid - l * p.nb;
    let base = line_base(l);
{cin}    let s0 = c * {REDUCE_CHUNK}u;
    let s1 = min(p.n, s0 + {REDUCE_CHUNK}u);
    var accv: {ty};
    var acci: u32;
    var s = s0 + t;
    if (s < s1) {{
        {}
        accv = fv;
        acci = fi;
        s += {w}u;
        for (; s < s1; s += {w}u) {{
            {}
            if (beats(ev, ei, accv, acci)) {{ accv = ev; acci = ei; }}
        }}
    }}
    partv[t] = accv;
    parti[t] = acci;
    let valid = min({w}u, s1 - s0);
    for (var d = {w}u / 2u; d > 0u; d >>= 1u) {{
        workgroupBarrier();
        if (t < d && t + d < valid) {{
            let xv = partv[t + d];
            let xi = parti[t + d];
            if (beats(xv, xi, partv[t], parti[t])) {{ partv[t] = xv; parti[t] = xi; }}
        }}
    }}
    if (t == 0u) {{ put(wid, bits(partv[0]), parti[0]); }}
}}
",
                cell("fv", "fi", "base + s", "s"),
                cell("ev", "ei", "base + s", "s"),
            );
        }
        Pass::Seq | Pass::Walk | Pass::ChunkFold | Pass::ChunkScan { .. } | Pass::ChunkLookback | Pass::Segments => unreachable!("scans are not reductions"),
    }
    src
}
