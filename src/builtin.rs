//! The built-in operations, expressed as WGSL kernels that run on the same engine as custom
//! kernels.
//!
//! Elementwise operations on arrays are not kernels of their own: they are recorded lazily and
//! fused into generated kernels, see [`crate::jit`]. What is left here are the in-place
//! operators, fills, the reduction passes over partial results, and the convolution and FFT
//! passes (generated here, in [`crate::fft`] and `operations/`). Every kernel declares its arguments in `@group(1)` and
//! uses the `luma` grid uniform of the prelude.

use std::borrow::Cow;

use crate::tuning::Tuning;

/// Workgroup size of the reduction kernels.
pub(crate) const REDUCE_WORKGROUP: u32 = 256;
/// Largest stride the tiled convolution handles; larger ones run one output per invocation.
pub(crate) const CONV_MAX_STRIDE: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum UnOp {
    Neg,
    Sqrt,
    Exp,
    Ln,
    Abs,
    Sin,
    Cos,
    Tan,
}

impl UnOp {
    /// The operation as a WGSL expression of the operand `x`.
    pub(crate) fn expr(self) -> &'static str {
        match self {
            UnOp::Neg => "-x",
            UnOp::Sqrt => "sqrt(x)",
            UnOp::Exp => "exp(x)",
            UnOp::Ln => "log(x)",
            UnOp::Abs => "abs(x)",
            UnOp::Sin => "sin(x)",
            UnOp::Cos => "cos(x)",
            UnOp::Tan => "tan(x)",
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            UnOp::Neg => "neg",
            UnOp::Sqrt => "sqrt",
            UnOp::Exp => "exp",
            UnOp::Ln => "ln",
            UnOp::Abs => "abs",
            UnOp::Sin => "sin",
            UnOp::Cos => "cos",
            UnOp::Tan => "tan",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

impl BinOp {
    /// The operation as a WGSL expression of the operands `x` and `y`.
    pub(crate) fn expr(self) -> &'static str {
        match self {
            BinOp::Add => "x + y",
            BinOp::Sub => "x - y",
            BinOp::Mul => "x * y",
            BinOp::Div => "x / y",
            BinOp::Min => "min(x, y)",
            BinOp::Max => "max(x, y)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    /// The comparison as a WGSL expression of `x` and `y`, a `u32` holding 0 or 1.
    pub(crate) fn expr(self) -> &'static str {
        match self {
            CmpOp::Lt => "select(0u, 1u, x < y)",
            CmpOp::Le => "select(0u, 1u, x <= y)",
            CmpOp::Gt => "select(0u, 1u, x > y)",
            CmpOp::Ge => "select(0u, 1u, x >= y)",
            CmpOp::Eq => "select(0u, 1u, x == y)",
            CmpOp::Ne => "select(0u, 1u, x != y)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ReduceOp {
    Sum,
    Min,
    Max,
    /// Sum of squares of the input; the later passes over its partial results are sums.
    SumSquares,
}

impl ReduceOp {
    /// The op the passes after the first apply to its partial results.
    fn rest(self) -> ReduceOp {
        match self {
            ReduceOp::SumSquares => ReduceOp::Sum,
            op => op,
        }
    }
}

/// Most reductions computed in one pass over the input.
pub(crate) const MAX_REDUCTIONS: usize = 4;

/// Reductions computed together, in order (a kernel cache key, so a fixed-size value).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceList {
    ops: [ReduceOp; MAX_REDUCTIONS],
    len: u8,
}

impl ReduceList {
    /// `ops` must hold between 1 and [`MAX_REDUCTIONS`] ops.
    pub(crate) fn new(ops: &[ReduceOp]) -> Self {
        let mut list = ReduceList { ops: [ReduceOp::Sum; MAX_REDUCTIONS], len: 0 };
        for (slot, &op) in list.ops.iter_mut().zip(ops) {
            *slot = op;
            list.len += 1;
        }
        list
    }

    pub(crate) fn as_slice(&self) -> &[ReduceOp] {
        &self.ops[..usize::from(self.len)]
    }

    /// The ops of the passes after the first.
    pub(crate) fn rest(self) -> Self {
        let mut list = self;
        list.ops.iter_mut().for_each(|op| *op = op.rest());
        list
    }
}

/// One built-in kernel. Together with the element type's WGSL name this is the cache key of the
/// kernel in the context.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Builtin {
    /// `a[i] = a[i] op b[i]`; arguments: a (read_write), b.
    InPlaceArray(BinOp),
    /// `a[i] = a[i] op s`; arguments: a (read_write), uniform scalar.
    InPlaceScalar(BinOp),
    /// `out[i] = s`; arguments: out, uniform scalar.
    Fill,
    /// One reduction pass for each op of the list (segmented: over the partial results of an
    /// earlier pass); see [`reduce`]. Arguments: out, a, uniform `{n}`.
    Reduce(ReduceList, bool),
    /// Direct convolution, one output per invocation; any stride.
    Conv1d,
    /// Direct convolution with a stride of up to [`CONV_MAX_STRIDE`] (first field), tiled with
    /// the given outputs per invocation at stride 1 (second); [`conv_tile`] outputs per
    /// workgroup.
    Conv1dTiled(u32, u32),
    /// One pass of the FFT convolution, see [`crate::fft`].
    Fft(crate::fft::FftPass),
}

/// `a[i] = a[i] op b[i]` (or `a[i] op p.s` with `scalar`).
fn in_place(ty: &str, scalar: bool, expr: &str, wg: u32) -> String {
    let (decls, load) = if scalar {
        (
            format!(
                "struct Params {{ s: {ty} }}
@group(1) @binding(0) var<storage, read_write> a: array<{ty}>;
@group(1) @binding(1) var<uniform> p: Params;
"
            ),
            "let x = a[i]; let y = p.s;",
        )
    } else {
        (
            format!(
                "@group(1) @binding(0) var<storage, read_write> a: array<{ty}>;
@group(1) @binding(1) var<storage, read> b: array<{ty}>;
"
            ),
            "let x = a[i]; let y = b[i];",
        )
    };
    format!(
        "{decls}
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
    {load}
    a[i] = {expr};
}}
"
    )
}

/// One reduction pass over `a` for every op of `ops`: each workgroup reduces
/// [`Tuning::reduce_block`] elements per op into `out[j * groups + group]`. `segmented` passes
/// reduce the partial results of an earlier pass, op `j` over `a[j * p.n ..]`; otherwise every op
/// reads the same `p.n` elements.
fn reduce(ty: &str, ops: ReduceList, segmented: bool, tuning: &Tuning) -> String {
    format!(
        "struct Params {{ n: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> a: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;

fn load_at(i: u32) -> {ty} {{ return a[i]; }}
{}",
        reduce_main(ty, ops, segmented, "load_at", tuning)
    )
}

/// `c(..)` over `values`, combined pairwise so float sums keep their accuracy.
fn pairwise(c: &str, values: &[String]) -> String {
    match values {
        [one] => one.clone(),
        _ => {
            let (l, r) = values.split_at(values.len() / 2);
            format!("{c}({}, {})", pairwise(c, l), pairwise(c, r))
        }
    }
}

/// The entry point of a reduction pass computing every op of `ops` (see [`reduce`]). Element
/// `i` comes from `{load}(i)`, and a `Params` uniform `p` with the element count `n` is bound.
pub(crate) fn reduce_main(ty: &str, ops: ReduceList, segmented: bool, load: &str, tuning: &Tuning) -> String {
    let (wg, block) = (REDUCE_WORKGROUP, tuning.reduce_block());
    let per_thread = tuning.reduce_per_thread;
    let mut body = String::new();
    // Without segments every op reduces the same elements: load each once.
    if !segmented {
        for s in 0..per_thread {
            body += &format!("    let e{s} = {load}(min(base + {wg}u * {s}u, p.n - 1u));\n");
        }
    }
    let mut combines = String::new();
    for (j, &op) in ops.as_slice().iter().enumerate() {
        let combine = match op {
            ReduceOp::Sum | ReduceOp::SumSquares => "u + v",
            ReduceOp::Min => "min(u, v)",
            ReduceOp::Max => "max(u, v)",
        };
        combines += &format!("fn c{j}(u: {ty}, v: {ty}) -> {ty} {{ return {combine}; }}\n");
        body += "    {\n";
        for s in 0..per_thread {
            if segmented {
                body += &format!("        let e{s} = {load}({j}u * p.n + min(base + {wg}u * {s}u, p.n - 1u));\n");
            }
            // Out-of-range slots: sums take zero, min/max re-read the last element (harmless).
            let value = match op {
                ReduceOp::Sum => format!("select(zero, e{s}, base + {wg}u * {s}u < p.n)"),
                ReduceOp::SumSquares => format!("select(zero, e{s} * e{s}, base + {wg}u * {s}u < p.n)"),
                ReduceOp::Min | ReduceOp::Max => format!("e{s}"),
            };
            body += &format!("        let v{s} = {value};\n");
        }
        let values: Vec<String> = (0..per_thread).map(|s| format!("v{s}")).collect();
        body += &format!(
            "        sh[lid] = {};
        workgroupBarrier();
        for (var s = {half}u; s > 0u; s = s >> 1u) {{
            if (lid < s) {{ sh[lid] = c{j}(sh[lid], sh[lid + s]); }}
            workgroupBarrier();
        }}
        if (lid == 0u && group * {block}u < p.n) {{ out[{j}u * groups + group] = sh[0]; }}
        workgroupBarrier();
    }}
",
            pairwise(&format!("c{j}"), &values),
            half = wg / 2,
        );
    }
    format!(
        "var<workgroup> sh: array<{ty}, {wg}>;

{combines}
@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let group = wid.x + wid.y * nwg.x;
    let groups = (p.n + {block}u - 1u) / {block}u;
    let base = group * {block}u + lid;
    let zero = {ty}();
{body}}}
"
    )
}

/// Direct convolution with stride `s` (1 to [`CONV_MAX_STRIDE`]), tiled. A workgroup computes
/// [`conv_tile`] consecutive outputs, an invocation `R` = `per_thread / s` consecutive ones in
/// registers. The kernel is consumed in chunks of taps whose weights and input window are staged
/// in workgroup memory (zero outside the input, so the inner loop has no bounds checks). The
/// inner loop takes `R` taps at a time from the `(R - 1) s + R` window values they touch, fully
/// unrolled, with the weights read four at a time when `R` allows. Window slots skip one word
/// every 32 so that the strided reads of neighbouring invocations fall into different banks.
/// The accumulators go back through the window so that the stores are coalesced.
fn conv1d_tiled(stride: u32, per_thread: u32, wg: u32) -> String {
    let s = stride;
    let r = per_thread / s;
    let tile = conv_tile(s, per_thread, wg);
    // A multiple of `R`, so that padding a chunk to whole groups of `R` taps stays within it.
    let chunk = wg / r * r;
    let span = tile * s + chunk;
    let window = span + span / 32 + 1;
    let mut acc = String::new();
    let mut loads = String::new();
    let mut fmas = String::new();
    let mut stage = String::new();
    let mut stash = String::new();
    let mut stores = String::new();
    for m in 0..r * s {
        stage.push_str(&format!(
            "        window[slot(lid + {0}u)] = load(start + i32(c + lid + {0}u));\n",
            m * wg
        ));
    }
    for i in 0..r {
        acc.push_str(&format!("    var acc{i} = 0.0;\n"));
        if r % 4 == 0 {
            if i % 4 == 0 {
                loads.push_str(&format!("            let q{} = weights[k / 4u + {}u];\n", i / 4, i / 4));
            }
            loads.push_str(&format!("            let w{i} = q{}.{};\n", i / 4, ["x", "y", "z", "w"][(i % 4) as usize]));
        } else {
            loads.push_str(&format!("            let w{i} = weights[k + {i}u];\n"));
        }
        stash.push_str(&format!("    window[slot(lid * {r}u + {i}u)] = acc{i};\n"));
        stores.push_str(&format!(
            "    if (first + lid + {}u < params.output_len) {{ output[first + lid + {}u] = window[slot(lid + {}u)]; }}\n",
            i * wg, i * wg, i * wg
        ));
    }
    for m in 0..(r - 1) * s + r {
        loads.push_str(&format!("            let v{m} = window[slot(at + {m}u)];\n"));
    }
    for kk in 0..r {
        for i in 0..r {
            fmas.push_str(&format!("            acc{i} = fma(v{}, w{kk}, acc{i});\n", i * s + kk));
        }
    }
    format!(
        "struct Params {{ kernel_len: u32, stride: u32, padding: u32, input_len: u32, output_len: u32 }}
@group(1) @binding(0) var<storage, read_write> output: array<f32>;
@group(1) @binding(1) var<storage, read> input_data: array<f32>;
@group(1) @binding(2) var<storage, read> kernel: array<f32>;
@group(1) @binding(3) var<uniform> params: Params;

var<workgroup> window: array<f32, {window}>;
var<workgroup> weights: array<{weights_ty}, {weights_len}>;

fn slot(i: u32) -> u32 {{ return i + (i >> 5u); }}

fn weight(c: u32, taps: u32, i: u32) -> f32 {{
    return select(0.0, kernel[min(c + i, params.kernel_len - 1u)], i < taps);
}}

fn load(g: i32) -> f32 {{
    let inside = g >= 0 && u32(g) < params.input_len;
    return select(0.0, input_data[u32(max(g, 0))], inside);
}}

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let first = (wid.x + wid.y * nwg.x) * {tile}u;
    if (first >= params.output_len) {{ return; }}
    let start = i32(first * {s}u) - i32(params.padding);
{acc}    for (var c = 0u; c < params.kernel_len; c = c + {chunk}u) {{
        let taps = min({chunk}u, params.kernel_len - c);
        let padded = (taps + {r}u - 1u) / {r}u * {r}u;
{weights_stage}
{stage}        if (lid < padded) {{
            window[slot({tile_in}u + lid)] = load(start + i32(c + {tile_in}u + lid));
        }}
        workgroupBarrier();
        for (var k = 0u; k < padded; k = k + {r}u) {{
            let at = lid * {rs}u + k;
{loads}{fmas}        }}
        workgroupBarrier();
    }}
{stash}    workgroupBarrier();
{stores}}}
",
        tile_in = tile * s,
        rs = r * s,
        weights_ty = if r % 4 == 0 { "vec4<f32>" } else { "f32" },
        weights_len = if r % 4 == 0 { chunk / 4 } else { chunk },
        weights_stage = if r % 4 == 0 {
            format!(
                "        if (lid < {}u) {{
            weights[lid] = vec4<f32>(weight(c, taps, lid * 4u), weight(c, taps, lid * 4u + 1u), weight(c, taps, lid * 4u + 2u), weight(c, taps, lid * 4u + 3u));
        }}
",
                chunk / 4
            )
        } else {
            format!("        if (lid < {chunk}u) {{ weights[lid] = weight(c, taps, lid); }}\n")
        },
    )
}

/// Outputs one workgroup of the tiled convolution computes (see [`conv1d_tiled`]).
pub(crate) fn conv_tile(stride: u32, per_thread: u32, wg: u32) -> u32 {
    wg * (per_thread / stride)
}

fn fill_kernel(ty: &str, wg: u32) -> String {
    format!(
        "struct Params {{ s: {ty} }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
    out[i] = p.s;
}}
"
    )
}

impl Builtin {
    /// The WGSL source of this kernel for elements of type `ty` (`"f32"` or `"u32"`), with the
    /// parameters `tuning` gives for the GPU.
    pub(crate) fn source(self, ty: &str, tuning: &Tuning) -> Cow<'static, str> {
        let wg = tuning.elementwise_workgroup;
        match self {
            Builtin::InPlaceArray(op) => in_place(ty, false, op.expr(), wg).into(),
            Builtin::InPlaceScalar(op) => in_place(ty, true, op.expr(), wg).into(),
            Builtin::Fill => fill_kernel(ty, wg).into(),
            Builtin::Reduce(ops, segmented) => reduce(ty, ops, segmented, tuning).into(),
            Builtin::Conv1d => include_str!("../operations/conv1d.wgsl").into(),
            Builtin::Conv1dTiled(stride, per_thread) => conv1d_tiled(stride, per_thread, tuning.conv_workgroup).into(),
            Builtin::Fft(pass) => crate::fft::source(pass).into(),
        }
    }
}
