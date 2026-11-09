//! The built-in operations, expressed as WGSL kernels that run on the same engine as custom
//! kernels.
//!
//! Elementwise operations are generated from templates per (operation, element type) and
//! compiled on first use; the context caches them like any other kernel. Convolution and FFT
//! passes are embedded shader files. Every kernel declares its arguments in `@group(1)` and
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
    fn expr(self) -> &'static str {
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
    fn expr(self) -> &'static str {
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
    fn expr(self) -> &'static str {
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

/// Where the operands of a binary kernel come from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Operands {
    /// `out = a op b`, two arrays.
    Arrays,
    /// `out = a op s`, with the scalar `s` passed as a uniform.
    ArrayScalar,
    /// `out = s op a`.
    ScalarArray,
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
    /// `out[i] = f(a[i])`; arguments: out, a.
    Unary(UnOp),
    /// `out[i] = a[i] op b[i]`; arguments: out, a, b / uniform scalar.
    Binary(BinOp, Operands),
    /// Like `Binary`, but `out` is `u32` holding 0 or 1.
    Compare(CmpOp, Operands),
    /// `a[i] = a[i] op b[i]`; arguments: a (read_write), b.
    InPlaceArray(BinOp),
    /// `a[i] = a[i] op s`; arguments: a (read_write), uniform scalar.
    InPlaceScalar(BinOp),
    /// `out[i] = pow(a[i], s)`; arguments: out, a, uniform.
    Powf,
    /// `out[i] = clamp(a[i], lo, hi)`; arguments: out, a, uniform `{lo, hi}`.
    Clamp,
    /// `out[i] = mask[i] != 0 ? a[i] : b[i]`; arguments: out, mask, a, b.
    Select,
    /// `out[i] = s`; arguments: out, uniform scalar.
    Fill,
    /// Circular shift; arguments: out, a, uniform `{dims, shifts}`.
    Shift,
    /// One reduction pass: each workgroup reduces `REDUCE_BLOCK` elements of `a` into one element
    /// of `out`; arguments: out, a, uniform `{n}`.
    Reduce(ReduceOp),
    Conv1d,
    FftBitReverse,
    FftButterfly,
    FftComplexMul,
    IfftButterfly,
    FftExtractReal,
    FftComplexBitReverse,
}

/// How the body of an elementwise kernel gets `x` and `y`.
enum Mode {
    /// `x = a[i]`
    Unary,
    /// `x = a[i]`, `y = b[i]`
    Arrays,
    /// `x = a[i]`, `y = p.s`
    ArrayScalar,
    /// `x = p.s`, `y = a[i]`
    ScalarArray,
    /// `x = a[i]` with a uniform `p` of the given fields available to the expression.
    ArrayParams(&'static str),
    /// `x = a[i]`, `y = b[i]`, result stored back into `a`.
    InPlaceArray,
    /// `x = a[i]`, `y = p.s`, result stored back into `a`.
    InPlaceScalar,
}

fn elementwise(ty: &str, out_ty: &str, mode: Mode, expr: &str) -> String {
    let params = |fields: &str| format!("struct Params {{ {fields} }}\n");
    let (decls, load) = match &mode {
        Mode::Unary => (
            "@group(1) @binding(0) var<storage, read_write> out: array<OUT>;\n\
             @group(1) @binding(1) var<storage, read> a: array<TY>;\n"
                .to_owned(),
            "let x = a[i];",
        ),
        Mode::Arrays => (
            "@group(1) @binding(0) var<storage, read_write> out: array<OUT>;\n\
             @group(1) @binding(1) var<storage, read> a: array<TY>;\n\
             @group(1) @binding(2) var<storage, read> b: array<TY>;\n"
                .to_owned(),
            "let x = a[i]; let y = b[i];",
        ),
        Mode::ArrayScalar | Mode::ScalarArray | Mode::ArrayParams(_) => {
            let fields = match &mode {
                Mode::ArrayParams(fields) => *fields,
                _ => "s: TY",
            };
            (
                params(fields)
                    + "@group(1) @binding(0) var<storage, read_write> out: array<OUT>;\n\
                       @group(1) @binding(1) var<storage, read> a: array<TY>;\n\
                       @group(1) @binding(2) var<uniform> p: Params;\n",
                match &mode {
                    Mode::ArrayScalar => "let x = a[i]; let y = p.s;",
                    Mode::ScalarArray => "let x = p.s; let y = a[i];",
                    _ => "let x = a[i];",
                },
            )
        }
        Mode::InPlaceArray => (
            "@group(1) @binding(0) var<storage, read_write> a: array<TY>;\n\
             @group(1) @binding(1) var<storage, read> b: array<TY>;\n"
                .to_owned(),
            "let x = a[i]; let y = b[i];",
        ),
        Mode::InPlaceScalar => (
            params("s: TY")
                + "@group(1) @binding(0) var<storage, read_write> a: array<TY>;\n\
                   @group(1) @binding(1) var<uniform> p: Params;\n",
            "let x = a[i]; let y = p.s;",
        ),
    };
    let store = match mode {
        Mode::InPlaceArray | Mode::InPlaceScalar => "a[i]",
        _ => "out[i]",
    };
    let decls = decls.replace("OUT", out_ty).replace("TY", ty);
    format!(
        "{decls}
@compute @workgroup_size({ELEMENTWISE_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {ELEMENTWISE_WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    {load}
    {store} = {expr};
}}
"
    )
}

fn reduce(ty: &str, op: ReduceOp) -> String {
    // Out-of-range loads: sums read zero, min/max re-read the last element (harmless for them).
    let (load, combine) = match op {
        ReduceOp::Sum => ("select(zero, a[idx], idx < p.n)", "u + v"),
        ReduceOp::SumSquares => ("select(zero, a[idx] * a[idx], idx < p.n)", "u + v"),
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
    let zero = {ty}(0);
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

fn select_kernel(ty: &str) -> String {
    format!(
        "@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> mask: array<u32>;
@group(1) @binding(2) var<storage, read> a: array<{ty}>;
@group(1) @binding(3) var<storage, read> b: array<{ty}>;
@compute @workgroup_size({ELEMENTWISE_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {ELEMENTWISE_WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    out[i] = select(b[i], a[i], mask[i] != 0u);
}}
"
    )
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

fn shift_kernel(ty: &str) -> String {
    // out[c] = a[(c - shift) mod dims]; shifts are normalized to 0..dims on the host.
    format!(
        "struct Params {{ dims: vec4<u32>, shifts: vec4<u32> }}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> a: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size({ELEMENTWISE_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {ELEMENTWISE_WORKGROUP}u;
    if (i >= luma.size.x) {{ return; }}
    let c0 = i % p.dims.x;
    let r1 = i / p.dims.x;
    let c1 = r1 % p.dims.y;
    let r2 = r1 / p.dims.y;
    let c2 = r2 % p.dims.z;
    let c3 = r2 / p.dims.z;
    let s0 = (c0 + p.dims.x - p.shifts.x) % p.dims.x;
    let s1 = (c1 + p.dims.y - p.shifts.y) % p.dims.y;
    let s2 = (c2 + p.dims.z - p.shifts.z) % p.dims.z;
    let s3 = (c3 + p.dims.w - p.shifts.w) % p.dims.w;
    out[i] = a[s0 + p.dims.x * (s1 + p.dims.y * (s2 + p.dims.z * s3))];
}}
"
    )
}

impl Builtin {
    /// The WGSL source of this kernel for elements of type `ty` (`"f32"` or `"u32"`).
    pub(crate) fn source(self, ty: &str) -> Cow<'static, str> {
        match self {
            Builtin::Unary(op) => elementwise(ty, ty, Mode::Unary, op.expr()).into(),
            Builtin::Binary(op, operands) => {
                elementwise(ty, ty, operands.mode(), op.expr()).into()
            }
            Builtin::Compare(op, operands) => {
                elementwise(ty, "u32", operands.mode(), op.expr()).into()
            }
            Builtin::InPlaceArray(op) => {
                elementwise(ty, ty, Mode::InPlaceArray, op.expr()).into()
            }
            Builtin::InPlaceScalar(op) => {
                elementwise(ty, ty, Mode::InPlaceScalar, op.expr()).into()
            }
            Builtin::Powf => elementwise(ty, ty, Mode::ArrayScalar, "pow(x, y)").into(),
            Builtin::Clamp => {
                elementwise(ty, ty, Mode::ArrayParams("lo: TY, hi: TY"), "clamp(x, p.lo, p.hi)")
                    .into()
            }
            Builtin::Select => select_kernel(ty).into(),
            Builtin::Fill => fill_kernel(ty).into(),
            Builtin::Shift => shift_kernel(ty).into(),
            Builtin::Reduce(op) => reduce(ty, op).into(),
            Builtin::Conv1d => include_str!("../operations/conv1d.wgsl").into(),
            Builtin::FftBitReverse => include_str!("../operations/fft_bit_reverse.wgsl").into(),
            Builtin::FftButterfly => include_str!("../operations/fft_butterfly.wgsl").into(),
            Builtin::FftComplexMul => include_str!("../operations/fft_complex_mul.wgsl").into(),
            Builtin::IfftButterfly => include_str!("../operations/ifft_butterfly.wgsl").into(),
            Builtin::FftExtractReal => include_str!("../operations/fft_extract_real.wgsl").into(),
            Builtin::FftComplexBitReverse => {
                include_str!("../operations/fft_complex_bit_reverse.wgsl").into()
            }
        }
    }
}

impl Operands {
    fn mode(self) -> Mode {
        match self {
            Operands::Arrays => Mode::Arrays,
            Operands::ArrayScalar => Mode::ArrayScalar,
            Operands::ScalarArray => Mode::ScalarArray,
        }
    }
}
