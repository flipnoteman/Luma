//! The built-in operations, expressed as WGSL kernels that run on the same engine as custom
//! kernels.
//!
//! Elementwise operations on arrays are not kernels of their own: they are recorded lazily and
//! fused into generated kernels, see [`crate::jit`]. What is left here are the in-place
//! operators, fills, the reduction passes over partial results, and the convolution and FFT
//! passes (generated here, in [`crate::fft`] and `operations/`). Every kernel declares its arguments in `@group(1)` and
//! uses the `luma` grid uniform of the prelude.

use std::borrow::Cow;

/// Workgroup size of the elementwise kernels; their index math (`gid.x + gid.y * nwg.x * 64`)
/// folds one-dimensional problems into two dimensions to stay within the dispatch limit.
pub(crate) const ELEMENTWISE_WORKGROUP: u32 = 64;

/// Workgroup size of the reduction kernels, and how many elements each thread loads.
pub(crate) const REDUCE_WORKGROUP: u32 = 256;
pub(crate) const REDUCE_PER_THREAD: u32 = 4;
/// Number of input elements one reduction workgroup consumes.
pub(crate) const REDUCE_BLOCK: u32 = REDUCE_WORKGROUP * REDUCE_PER_THREAD;
/// Invocations of the tiled convolution, and how many consecutive outputs each computes at
/// stride 1 (stride `s` computes `1 / s` as many).
const CONV_WORKGROUP: u32 = 256;
const CONV_PER_THREAD: u32 = 16;
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
    /// Sum of squares of the input; only used as the first pass of a norm.
    SumSquares,
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
    /// One reduction pass: each workgroup reduces `REDUCE_BLOCK` elements of `a` into one element
    /// of `out`; arguments: out, a, uniform `{n}`.
    Reduce(ReduceOp),
    /// Direct convolution, one output per invocation; any stride.
    Conv1d,
    /// Direct convolution with a stride of up to [`CONV_MAX_STRIDE`], tiled; [`conv_tile`]
    /// outputs per workgroup.
    Conv1dTiled(u32),
    /// One pass of the FFT convolution, see [`crate::fft`].
    Fft(crate::fft::FftPass),
}

/// `a[i] = a[i] op b[i]` (or `a[i] op p.s` with `scalar`).
fn in_place(ty: &str, scalar: bool, expr: &str) -> String {
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
@compute @workgroup_size({ELEMENTWISE_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {ELEMENTWISE_WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    {load}
    a[i] = {expr};
}}
"
    )
}

fn reduce(ty: &str, op: ReduceOp) -> String {
    // Out-of-range loads: sums read zero, min/max re-read the last element (harmless for them).
    // Loads stay in bounds: trusted kernels run without bounds checks.
    let (load, combine) = match op {
        ReduceOp::Sum => ("select(zero, a[min(idx, p.n - 1u)], idx < p.n)", "u + v"),
        ReduceOp::SumSquares => {
            ("select(zero, a[min(idx, p.n - 1u)] * a[min(idx, p.n - 1u)], idx < p.n)", "u + v")
        }
        ReduceOp::Min => ("a[min(idx, p.n - 1u)]", "min(u, v)"),
        ReduceOp::Max => ("a[min(idx, p.n - 1u)]", "max(u, v)"),
    };
    let wg = REDUCE_WORKGROUP;
    let block = REDUCE_BLOCK;
    format!(
        "struct Params {{ n: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> a: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;

var<workgroup> sh: array<{ty}, {wg}>;

fn combine(u: {ty}, v: {ty}) -> {ty} {{ return {combine}; }}

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let group = wid.x + wid.y * nwg.x;
    let base = group * {block}u + lid;
    let zero = {ty}();
    let idx0 = base;
    let idx1 = base + {wg}u;
    let idx2 = base + {wg}u * 2u;
    let idx3 = base + {wg}u * 3u;
    var v0: {ty}; var v1: {ty}; var v2: {ty}; var v3: {ty};
    {{ let idx = idx0; v0 = {load}; }}
    {{ let idx = idx1; v1 = {load}; }}
    {{ let idx = idx2; v2 = {load}; }}
    {{ let idx = idx3; v3 = {load}; }}
    sh[lid] = combine(combine(v0, v1), combine(v2, v3));
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ sh[lid] = combine(sh[lid], sh[lid + s]); }}
        workgroupBarrier();
    }}
    if (lid == 0u && group * {block}u < p.n) {{ out[group] = sh[0]; }}
}}
",
        half = wg / 2,
    )
}

/// Direct convolution with stride `s` (1 to [`CONV_MAX_STRIDE`]), tiled. A workgroup computes
/// [`conv_tile`] consecutive outputs, an invocation `R` = [`CONV_PER_THREAD`]` / s` consecutive
/// ones in registers. The kernel is consumed in chunks of taps whose weights and input window
/// are staged in workgroup memory (zero outside the input, so the inner loop has no bounds
/// checks). The inner loop takes `R` taps at a time from the `(R - 1) s + R` window values they
/// touch, fully unrolled. Window slots skip one word every 32 so that the strided reads of
/// neighbouring invocations fall into different banks.
fn conv1d_tiled(stride: u32) -> String {
    let (wg, s) = (CONV_WORKGROUP, stride);
    let r = CONV_PER_THREAD / s;
    let tile = conv_tile(s);
    // A multiple of `R`, so that padding a chunk to whole groups of `R` taps stays within it.
    let chunk = wg / r * r;
    let span = tile * s + chunk;
    let window = span + span / 32 + 1;
    let mut acc = String::new();
    let mut loads = String::new();
    let mut fmas = String::new();
    let mut stores = String::new();
    for i in 0..r {
        acc.push_str(&format!("    var acc{i} = 0.0;\n"));
        loads.push_str(&format!("            let w{i} = weights[k + {i}u];\n"));
        stores.push_str(&format!(
            "    if (o + {i}u < params.output_len) {{ output[o + {i}u] = acc{i}; }}\n"
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
var<workgroup> weights: array<f32, {chunk}>;

fn slot(i: u32) -> u32 {{ return i + (i >> 5u); }}

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
        if (lid < {chunk}u) {{
            weights[lid] = select(0.0, kernel[min(c + lid, params.kernel_len - 1u)], lid < taps);
        }}
        for (var j = lid; j < {tile_in}u + padded; j = j + {wg}u) {{
            let g = start + i32(c + j);
            var v = 0.0;
            if (g >= 0 && u32(g) < params.input_len) {{ v = input_data[u32(g)]; }}
            window[slot(j)] = v;
        }}
        workgroupBarrier();
        for (var k = 0u; k < padded; k = k + {r}u) {{
            let at = lid * {rs}u + k;
{loads}{fmas}        }}
        workgroupBarrier();
    }}
    let o = first + lid * {r}u;
{stores}}}
",
        tile_in = tile * s,
        rs = r * s,
    )
}

/// Outputs one workgroup of the tiled convolution with `stride` computes.
pub(crate) fn conv_tile(stride: u32) -> u32 {
    CONV_WORKGROUP * (CONV_PER_THREAD / stride)
}

fn fill_kernel(ty: &str) -> String {
    format!(
        "struct Params {{ s: {ty} }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size({ELEMENTWISE_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {ELEMENTWISE_WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    out[i] = p.s;
}}
"
    )
}

impl Builtin {
    /// The WGSL source of this kernel for elements of type `ty` (`"f32"` or `"u32"`).
    pub(crate) fn source(self, ty: &str) -> Cow<'static, str> {
        match self {
            Builtin::InPlaceArray(op) => in_place(ty, false, op.expr()).into(),
            Builtin::InPlaceScalar(op) => in_place(ty, true, op.expr()).into(),
            Builtin::Fill => fill_kernel(ty).into(),
            Builtin::Reduce(op) => reduce(ty, op).into(),
            Builtin::Conv1d => include_str!("../operations/conv1d.wgsl").into(),
            Builtin::Conv1dTiled(stride) => conv1d_tiled(stride).into(),
            Builtin::Fft(pass) => crate::fft::source(pass).into(),
        }
    }
}
