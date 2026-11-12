//! The built-in operations, expressed as WGSL kernels that run on the same engine as custom
//! kernels.
//!
//! Elementwise operations on arrays are not kernels of their own: they are recorded lazily and
//! fused into generated kernels, see [`crate::jit`]. What is left here are the
//! fills, the reduction passes over partial results, and the convolution and FFT
//! passes (generated here, in [`crate::fft`] and `operations/`). Every kernel declares its arguments in `@group(1)` and
//! uses the `luma` grid uniform of the prelude.

use std::borrow::Cow;
use std::fmt::Write;

use crate::jit::ElemTy;
use crate::tuning::{Tuning, TuningProfile};

/// Workgroup size of the reduction kernels.
pub(crate) const REDUCE_WORKGROUP: u32 = 256;
/// Largest stride the tiled convolution handles; larger ones run one output per invocation.
pub(crate) const CONV_MAX_STRIDE: u32 = 4;

/// `template` with its operand identifiers `x` and `y` replaced by `a` and `b`.
pub(crate) fn substitute(template: &str, a: &str, b: &str) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        match word.as_str() {
            "x" => out.push_str(a),
            "y" => out.push_str(b),
            other => out.push_str(other),
        }
        word.clear();
    };
    for c in template.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

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
    Tanh,
    Sign,
    Floor,
    Ceil,
    /// The identity: a copy of the operand (internal).
    Id,
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
            UnOp::Tanh => "tanh(x)",
            UnOp::Sign => "sign(x)",
            UnOp::Floor => "floor(x)",
            UnOp::Ceil => "ceil(x)",
            UnOp::Id => "x",
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
            UnOp::Tanh => "tanh",
            UnOp::Sign => "sign",
            UnOp::Floor => "floor",
            UnOp::Ceil => "ceil",
            UnOp::Id => "copy",
        }
    }
}

impl UnOp {
    /// [`UnOp::expr`] for elements of type `ty`, `None` if the operation does not exist for it
    /// (`f64` has the exact built-ins, `Df32` the library ones, see [`df_library`]).
    pub(crate) fn expr_for(self, ty: ElemTy) -> Option<&'static str> {
        match ty {
            ElemTy::F32 | ElemTy::U32 => Some(self.expr()),
            ElemTy::F64 => match self {
                UnOp::Neg | UnOp::Abs | UnOp::Sqrt | UnOp::Sign | UnOp::Floor | UnOp::Ceil | UnOp::Id => Some(self.expr()),
                _ => None,
            },
            ElemTy::Df32 => match self {
                UnOp::Neg => Some("-x"),
                UnOp::Abs => Some("df_abs(x)"),
                UnOp::Sqrt => Some("df_sqrt(x)"),
                UnOp::Id => Some("x"),
                _ => None,
            },
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
    Atan2,
    Hypot,
    Copysign,
    Rem,
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
            BinOp::Atan2 => "atan2(x, y)",
            BinOp::Hypot => "length(vec2<f32>(x, y))",
            BinOp::Copysign => "bitcast<f32>((bitcast<u32>(x) & 0x7fffffffu) | (bitcast<u32>(y) & 0x80000000u))",
            BinOp::Rem => "x % y",
        }
    }
}

impl BinOp {
    /// [`BinOp::expr`] for elements of type `ty`, `None` if the operation does not exist for it.
    pub(crate) fn expr_for(self, ty: ElemTy) -> Option<&'static str> {
        match ty {
            ElemTy::F32 | ElemTy::U32 => Some(self.expr()),
            ElemTy::F64 => match self {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Min | BinOp::Max => Some(self.expr()),
                _ => None,
            },
            ElemTy::Df32 => match self {
                BinOp::Add => Some("df_add(x, y)"),
                BinOp::Sub => Some("df_add(x, -y)"),
                BinOp::Mul => Some("df_mul(x, y)"),
                BinOp::Div => Some("df_div(x, y)"),
                BinOp::Min => Some("df_min(x, y)"),
                BinOp::Max => Some("df_max(x, y)"),
                _ => None,
            },
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

impl CmpOp {
    /// [`CmpOp::expr`] for elements of type `ty`.
    pub(crate) fn expr_for(self, ty: ElemTy) -> &'static str {
        match ty {
            ElemTy::Df32 => match self {
                CmpOp::Lt => "select(0u, 1u, df_lt(x, y))",
                CmpOp::Le => "select(0u, 1u, df_le(x, y))",
                CmpOp::Gt => "select(0u, 1u, df_lt(y, x))",
                CmpOp::Ge => "select(0u, 1u, df_le(y, x))",
                CmpOp::Eq => "select(0u, 1u, df_eq(x, y))",
                CmpOp::Ne => "select(1u, 0u, df_eq(x, y))",
            },
            _ => self.expr(),
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

/// Most elementwise operations fused behind the reduction passes (see [`Epilogue`]).
pub(crate) const MAX_EPILOGUE: usize = 4;

/// One elementwise operation on the result of a reduction, with a scalar operand if binary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum EpilogueStep {
    /// `op(value)`.
    Unary(UnOp),
    /// `value op scalar`, or `scalar op value` if `scalar_left`.
    Binary { op: BinOp, scalar_left: bool },
}

/// Elementwise operations on `f32` applied to every value a reduction leaves, inside its last
/// pass: `sqrt(sum / n)` is the reduction and two steps. Only the operations are part of the
/// kernel (a kernel cache key); the values of the scalar operands are uniforms (see [`Finish`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct Epilogue {
    steps: [Option<EpilogueStep>; MAX_EPILOGUE],
    len: u8,
}

impl Epilogue {
    /// This epilogue followed by `step`, `None` if it has [`MAX_EPILOGUE`] steps already.
    pub(crate) fn then(self, step: EpilogueStep) -> Option<Self> {
        let at = usize::from(self.len);
        let mut next = self;
        *next.steps.get_mut(at)? = Some(step);
        next.len += 1;
        Some(next)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The steps in order.
    pub(crate) fn steps(&self) -> impl Iterator<Item = EpilogueStep> + '_ {
        self.steps[..usize::from(self.len)].iter().flatten().copied()
    }

    /// The fields of the `Params` uniform that hold the scalar operands: `e{k}: f32,` for every
    /// binary step `k`.
    pub(crate) fn fields(&self) -> String {
        let mut out = String::new();
        for (k, step) in self.steps().enumerate() {
            if matches!(step, EpilogueStep::Binary { .. }) {
                let _ = write!(out, "e{k}: f32, ");
            }
        }
        out
    }

    /// WGSL statements applying the steps to the `f32` expression `value` (`let {prefix}{k}` is
    /// the result of step `k`) and the name of the final result (`value` if there are none).
    pub(crate) fn statements(&self, value: &str, prefix: &str) -> (String, String) {
        let mut out = String::new();
        let mut last = value.to_owned();
        for (k, step) in self.steps().enumerate() {
            let expr = match step {
                EpilogueStep::Unary(op) => substitute(op.expr(), &last, ""),
                EpilogueStep::Binary { op, scalar_left: false } => substitute(op.expr(), &last, &format!("p.e{k}")),
                EpilogueStep::Binary { op, scalar_left: true } => substitute(op.expr(), &format!("p.e{k}"), &last),
            };
            let _ = writeln!(out, "        let {prefix}{k}: f32 = {expr};");
            last = format!("{prefix}{k}");
        }
        (out, last)
    }
}

/// An [`Epilogue`] with the values of its scalar operands (the bit patterns of `f32`s).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Finish {
    pub(crate) epilogue: Epilogue,
    /// The value of the scalar of step `k`, for a binary step `k`.
    pub(crate) words: [u32; MAX_EPILOGUE],
}

impl Finish {
    /// The words the `Params` uniform holds after the fixed fields, in the order of
    /// [`Epilogue::fields`].
    pub(crate) fn uniform_words(&self) -> impl Iterator<Item = u32> + '_ {
        self.epilogue.steps().enumerate().filter(|(_, step)| matches!(step, EpilogueStep::Binary { .. })).map(|(k, _)| self.words[k])
    }
}

/// Reductions computed together, in order (a kernel cache key, so a fixed-size value), and what
/// is applied to their results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceList {
    ops: [ReduceOp; MAX_REDUCTIONS],
    len: u8,
    /// Applied to every value of the pass's output; only a pass that is the last (one group)
    /// has one.
    epilogue: Epilogue,
}

impl ReduceList {
    /// `ops` must hold between 1 and [`MAX_REDUCTIONS`] ops.
    pub(crate) fn new(ops: &[ReduceOp]) -> Self {
        let mut list = ReduceList { ops: [ReduceOp::Sum; MAX_REDUCTIONS], len: 0, epilogue: Epilogue::default() };
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

    /// These reductions with `epilogue` applied to the results. The kernel must run as a single
    /// group (the last pass): its output is the final value.
    pub(crate) fn with_epilogue(self, epilogue: Epilogue) -> Self {
        ReduceList { epilogue, ..self }
    }

    pub(crate) fn epilogue(&self) -> Epilogue {
        self.epilogue
    }
}

/// One built-in kernel. Together with the element type's WGSL name this is the cache key of the
/// kernel in the context.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Builtin {
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
    /// The convergence gate between the steps of `iterate_until`; arguments: the residual, the
    /// state `[done, count]`, the indirect arguments of the next step, their template, uniform
    /// `{tol, step, words}`.
    UntilGate,
    /// Block Thomas solves, one invocation per line; see [`tridiagonal`]. `block` is the block
    /// size (1 = scalar), `starts` whether every line begins at its own cell, `workgroup` the
    /// invocations per workgroup (see [`tridiagonal_workgroup`]).
    Tridiagonal { block: u8, starts: bool, workgroup: u8 },
    /// `out[i] = 1` where word `i` of `a` is a NaN or an infinity (as `f32` bits), else 0;
    /// arguments: out, a. Element type `"u32"`. See `jit::check_finite`.
    NonFinite,
}

/// One reduction pass over `a` for every op of `ops`: each workgroup reduces its share of the `p.n`
/// elements into `out[j * groups + group]`, `groups` being [`Tuning::reduce_groups`]; a workgroup
/// takes [`Tuning::reduce_block`] elements at a time, strided over the array. `segmented` passes
/// reduce the partial results of an earlier pass, op `j` over `a[j * p.n ..]`; otherwise every op
/// reads the same `p.n` elements.
fn reduce(ty: &str, ops: ReduceList, segmented: bool, tuning: &Tuning) -> String {
    // The double-float kernel takes the guard of its arithmetic (see `df_library`) as a uniform.
    let fields = ops.epilogue().fields();
    let (elem, params) = match ty {
        "df32" => (ReduceElem::Df32 { guard: "p.guard" }, "struct Params { n: u32, guard: u32 }".to_owned()),
        other if fields.is_empty() => (ReduceElem::Plain(other), "struct Params { n: u32 }".to_owned()),
        other => (ReduceElem::Plain(other), format!("struct Params {{ n: u32, {fields}}}")),
    };
    let ty = elem.storage_ty();
    format!(
        "{params}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> a: array<{ty}>;
@group(1) @binding(2) var<uniform> p: Params;

fn load_at(i: u32) -> {ty} {{ return a[i]; }}
{}",
        reduce_main(elem, ops, segmented, "load_at", tuning)
    )
}

/// The element type a reduction pass works on.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReduceElem<'a> {
    /// A type WGSL adds, compares and squares itself.
    Plain(&'a str),
    /// A double-float (`vec2<f32>`) combined with the library of [`df_library`]; `guard` names
    /// the uniform that is always 0.
    Df32 { guard: &'a str },
}

impl<'a> ReduceElem<'a> {
    fn storage_ty(self) -> &'a str {
        match self {
            ReduceElem::Plain(ty) => ty,
            ReduceElem::Df32 { .. } => "vec2<f32>",
        }
    }

    /// The expression combining accumulators `u` and `v`.
    fn combine(self, op: ReduceOp) -> &'static str {
        match (self, op) {
            (ReduceElem::Plain(_), ReduceOp::Sum | ReduceOp::SumSquares) => "u + v",
            (ReduceElem::Plain(_), ReduceOp::Min) => "min(u, v)",
            (ReduceElem::Plain(_), ReduceOp::Max) => "max(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Sum | ReduceOp::SumSquares) => "df_add(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Min) => "df_min(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Max) => "df_max(u, v)",
        }
    }

    /// The square of the expression `e`.
    fn square(self, e: &str) -> String {
        match self {
            ReduceElem::Plain(_) => format!("{e} * {e}"),
            ReduceElem::Df32 { .. } => format!("df_mul({e}, {e})"),
        }
    }

    /// Whether the workgroup built-ins (`subgroupAdd`, ...) work on the type.
    fn subgroups(self) -> bool {
        matches!(self, ReduceElem::Plain("f32" | "u32" | "vec2<f32>" | "vec4<f32>"))
    }

    fn library(self) -> String {
        match self {
            ReduceElem::Plain(_) => String::new(),
            ReduceElem::Df32 { guard } => df_library(guard),
        }
    }
}

/// Double-float arithmetic on `vec2<f32>` `(hi, lo)`. Every intermediate of an error-free
/// transformation goes through `luma_o`, which ORs the bits with the uniform `guard` (always 0,
/// but unknown to the shader compiler): no reassociation or fast-math contraction can cancel
/// the transformations. (It is an OR, not an XOR, so that `luma_o(luma_o(x))`, as in a cast
/// feeding a cast, cannot fold back to `x`: the guard would cancel with itself.)
pub(crate) fn df_library(guard: &str) -> String {
    const LIBRARY: &str = "fn luma_o(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | {G}); }
fn df_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = luma_o(a + b); let bb = luma_o(s - a);
    let da = luma_o(a - luma_o(s - bb)); let db = luma_o(b - bb);
    return vec2<f32>(s, da + db);
}
fn df_quick(a: f32, b: f32) -> vec2<f32> { let s = luma_o(a + b); return vec2<f32>(s, b - luma_o(s - a)); }
fn df_split(a: f32) -> vec2<f32> { let t = luma_o(4097.0 * a); let hi = luma_o(t - luma_o(t - a)); return vec2<f32>(hi, luma_o(a - hi)); }
fn df_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = luma_o(a * b); let x = df_split(a); let y = df_split(b);
    let e1 = luma_o(x.x * y.x - p); let e2 = luma_o(e1 + x.x * y.y); let e3 = luma_o(e2 + x.y * y.x);
    return vec2<f32>(p, e3 + x.y * y.y);
}
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x); let t = df_two_sum(a.y, b.y);
    let c = df_quick(s.x, luma_o(s.y + t.x)); return df_quick(c.x, luma_o(c.y + t.y));
}
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = df_two_prod(a.x, b.x); return df_quick(p.x, luma_o(p.y + luma_o(a.x * b.y + a.y * b.x)));
}
fn df_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x; let r1 = df_add(a, -df_mul(b, vec2<f32>(q1, 0.0)));
    let q2 = r1.x / b.x; let r2 = df_add(r1, -df_mul(b, vec2<f32>(q2, 0.0)));
    let q3 = r2.x / b.x; return df_add(df_quick(q1, q2), vec2<f32>(q3, 0.0));
}
fn df_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) { return vec2<f32>(sqrt(a.x), 0.0); }
    let x = 1.0 / sqrt(a.x); let ax = luma_o(a.x * x);
    let e = df_add(a, -df_two_prod(ax, ax)); return df_two_sum(ax, e.x * (x * 0.5));
}
fn df_lt(a: vec2<f32>, b: vec2<f32>) -> bool { return a.x < b.x || (a.x == b.x && a.y < b.y); }
fn df_le(a: vec2<f32>, b: vec2<f32>) -> bool { return a.x < b.x || (a.x == b.x && a.y <= b.y); }
fn df_eq(a: vec2<f32>, b: vec2<f32>) -> bool { return a.x == b.x && a.y == b.y; }
fn df_min(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return select(b, a, df_lt(a, b)); }
fn df_max(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return select(a, b, df_lt(a, b)); }
fn df_abs(a: vec2<f32>) -> vec2<f32> { return select(a, -a, df_lt(a, vec2<f32>(0.0, 0.0))); }
";
    LIBRARY.replace("{G}", guard)
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

/// Most subgroups a reduction workgroup can have: the smallest subgroup size WebGPU allows is 4.
const MAX_SUBGROUPS: u32 = REDUCE_WORKGROUP / 4;

/// The entry point of a reduction pass computing every op of `ops` (see [`reduce`]). Element
/// `i` comes from `{load}(i)`, and a `Params` uniform `p` with the element count `n` is bound.
/// Every invocation accumulates its elements of the strided blocks, then the workgroup combines
/// the accumulators: within subgroups with `subgroupAdd` and friends and across them through
/// workgroup memory when the device has subgroups, by a shared-memory tree otherwise.
pub(crate) fn reduce_main(elem: ReduceElem<'_>, ops: ReduceList, segmented: bool, load: &str, tuning: &Tuning) -> String {
    let ty = elem.storage_ty();
    let (wg, block) = (REDUCE_WORKGROUP, tuning.reduce_block());
    let per_thread = tuning.reduce_per_thread;
    let mut combines = String::new();
    let mut inits = String::new();
    let mut loop_body = String::new();
    // Without segments every op reduces the same elements: load each once.
    if !segmented {
        for s in 0..per_thread {
            loop_body += &format!("        let e{s} = {load}(min(base + {wg}u * {s}u, p.n - 1u));\n");
        }
    }
    for (j, &op) in ops.as_slice().iter().enumerate() {
        let combine = elem.combine(op);
        combines += &format!("fn c{j}(u: {ty}, v: {ty}) -> {ty} {{ return {combine}; }}\n");
        let segment = if segmented { format!("{j}u * p.n + ") } else { String::new() };
        // Every group starts below `p.n`, so the first element is a real one for min and max.
        inits += &match op {
            ReduceOp::Sum | ReduceOp::SumSquares => format!("    var acc{j} = zero;\n"),
            ReduceOp::Min | ReduceOp::Max => format!("    var acc{j} = {load}({segment}min(group * {block}u + lid, p.n - 1u));\n"),
        };
        for s in 0..per_thread {
            if segmented {
                loop_body += &format!("        let e{j}_{s} = {load}({segment}min(base + {wg}u * {s}u, p.n - 1u));\n");
            }
            let e = if segmented { format!("e{j}_{s}") } else { format!("e{s}") };
            // Out-of-range slots: sums take zero, min/max re-read the last element (harmless).
            let value = match op {
                ReduceOp::Sum => format!("select(zero, {e}, base + {wg}u * {s}u < p.n)"),
                ReduceOp::SumSquares => format!("select(zero, {}, base + {wg}u * {s}u < p.n)", elem.square(&e)),
                ReduceOp::Min | ReduceOp::Max => e,
            };
            loop_body += &format!("        let v{j}_{s} = {value};\n");
        }
        let values: Vec<String> = (0..per_thread).map(|s| format!("v{j}_{s}")).collect();
        loop_body += &format!("        acc{j} = c{j}(acc{j}, {});\n", pairwise(&format!("c{j}"), &values));
    }

    let mut finish = String::new();
    let (shared, extra_args) = if tuning.subgroups && elem.subgroups() {
        let n = ops.as_slice().len() as u32 * MAX_SUBGROUPS;
        let mut lanes = String::new();
        let mut first = String::new();
        for (j, &op) in ops.as_slice().iter().enumerate() {
            let (group_op, ident) = match op {
                ReduceOp::Sum | ReduceOp::SumSquares => ("subgroupAdd", "zero".to_owned()),
                ReduceOp::Min => ("subgroupMin", format!("sh[{}u]", j as u32 * MAX_SUBGROUPS)),
                ReduceOp::Max => ("subgroupMax", format!("sh[{}u]", j as u32 * MAX_SUBGROUPS)),
            };
            lanes += &format!("    let r{j} = {group_op}(acc{j});\n    if (lane == 0u) {{ sh[{}u + sgid] = r{j}; }}\n", j as u32 * MAX_SUBGROUPS);
            let (lets, fin) = ops.epilogue().statements(&format!("f{j}"), &format!("g{j}_"));
            first += &format!(
                "        var t{j} = {ident};
        for (var k = lane; k < nsg; k += sg) {{ t{j} = c{j}(t{j}, sh[{}u + k]); }}
        let f{j} = {group_op}(t{j});
        if (lane == 0u) {{ {lets}out[{j}u * groups + group] = {fin}; }}
",
                j as u32 * MAX_SUBGROUPS
            );
        }
        finish += &format!(
            "    let sgid = lid / sg;
{lanes}    workgroupBarrier();
    if (sgid == 0u) {{
        let nsg = ({wg}u + sg - 1u) / sg;
{first}    }}
"
        );
        (
            format!("var<workgroup> sh: array<{ty}, {n}>;"),
            "\n    @builtin(subgroup_invocation_id) lane: u32,\n    @builtin(subgroup_size) sg: u32,",
        )
    } else {
        for j in 0..ops.as_slice().len() {
            let (lets, fin) = ops.epilogue().statements("sh[0]", &format!("g{j}_"));
            finish += &format!(
                "    sh[lid] = acc{j};
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ sh[lid] = c{j}(sh[lid], sh[lid + s]); }}
        workgroupBarrier();
    }}
    if (lid == 0u) {{ {lets}out[{j}u * groups + group] = {fin}; }}
    workgroupBarrier();
",
                half = wg / 2
            );
        }
        (format!("var<workgroup> sh: array<{ty}, {wg}>;"), "")
    };
    format!(
        "{}{shared}

{combines}
@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,{extra_args}
) {{
    let group = wid.x + wid.y * nwg.x;
    let groups = min((p.n + {block}u - 1u) / {block}u, {cap}u);
    if (group >= groups) {{ return; }}
    let zero = {ty}();
{inits}    for (var b = group * {block}u; b < p.n; b += groups * {block}u) {{
        let base = b + lid;
{loop_body}    }}
{finish}}}
",
        elem.library(),
        cap = tuning.reduce_max_groups
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
        if r.is_multiple_of(4) {
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
        weights_ty = if r.is_multiple_of(4) { "vec4<f32>" } else { "f32" },
        weights_len = if r.is_multiple_of(4) { chunk / 4 } else { chunk },
        weights_stage = if r.is_multiple_of(4) {
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

/// Invocations per workgroup of the tridiagonal solve of `lines` lines. It runs one invocation
/// per grid line, so there are often only a thousand or a few thousand of them. With the
/// elementwise workgroup (256 on NVIDIA) they fill a handful of multiprocessors; small workgroups
/// spread them over all of them (a workgroup runs on one multiprocessor) and the solve is
/// latency bound, so lanes of a warp left idle cost nothing. Aiming for 512 workgroups, from 8
/// invocations to a warp of 32 (RTX 3090; 4096 lines, `tri/solve_axis0`: 9.4 ms with 256, 3.9
/// with 32, 2.6 with 8; 1024 block-4 lines: 9.5, 2.1 ms with 32, 8 after unrolling the kernel).
/// Software rasterizers keep the elementwise size.
pub(crate) fn tridiagonal_workgroup(tuning: &Tuning, lines: u32) -> u8 {
    match tuning.profile {
        TuningProfile::Cpu => tuning.elementwise_workgroup as u8,
        _ => (lines / 512).max(1).next_power_of_two().clamp(8, 32).min(tuning.elementwise_workgroup) as u8,
    }
}

/// The part of the tridiagonal kernels before the entry point: parameters, bindings and the line
/// addressing. `out` holds the forward-swept right-hand sides and then the solution, `cp` the
/// swept upper blocks. Cell `q` of the grid holds block `lo/diag/up[q * B * B + row * B + col]`
/// and `rhs[q * B + row]`; the line `l` has `p.n` cells `line_base(l) + s * p.stride`.
fn tridiagonal_header(starts: bool) -> String {
    let (starts_binding, starts_decl) = if starts {
        ("@group(1) @binding(6) var<storage, read> starts: array<u32>;\n", 7)
    } else {
        ("", 6)
    };
    format!(
        "struct Params {{ n: u32, stride: u32, lines: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read_write> cp: array<f32>;
@group(1) @binding(2) var<storage, read> lo: array<f32>;
@group(1) @binding(3) var<storage, read> diag: array<f32>;
@group(1) @binding(4) var<storage, read> up: array<f32>;
@group(1) @binding(5) var<storage, read> rhs: array<f32>;
{starts_binding}@group(1) @binding({starts_decl}) var<uniform> p: Params;
fn line_base(l: u32) -> u32 {{ return (l % p.stride) + (l / p.stride) * p.stride * p.n; }}
"
    )
}

/// Solves one block-tridiagonal system per grid line with the block Thomas algorithm
/// (`block` x `block` blocks, Gaussian elimination with partial pivoting; see
/// [`tridiagonal_header`] for the layout). With `starts` the system of line `l` begins at cell
/// `starts[l]` and the cells before it are zero. A singular pivot gives Inf/NaN.
///
/// Blocks of up to [`UNROLLED_BLOCK`] are generated as straight-line code over scalar variables
/// (function-local arrays indexed in loops live in per-thread memory, which is slow on every
/// GPU); larger ones keep the loop form.
fn tridiagonal(block: u32, starts: bool, wg: u32) -> String {
    if block <= UNROLLED_BLOCK {
        tridiagonal_unrolled(block as usize, starts, wg)
    } else {
        tridiagonal_loops(block, starts, wg)
    }
}

/// The largest block size the solve is unrolled for.
const UNROLLED_BLOCK: u32 = 5;

/// The straight-line solve of `b`-blocks. Per cell the forward sweep holds the diagonal block
/// `m`, the right-hand side `r = [up | rhs]` (all `b x (b + 1)` entries as scalars) and the
/// previous cell's swept values `pc`/`pd` in registers; it eliminates with partial pivoting
/// (the pivot row is found with `select`s, rows are exchanged with `select`s, nothing branches)
/// and writes the swept values for the backward sweep, which carries the solution of the next
/// cell in registers. For `b == 1` this is the plain Thomas algorithm.
fn tridiagonal_unrolled(b: usize, starts: bool, wg: u32) -> String {
    let w = b + 1;
    let bb = b * b;
    let s0 = if starts { "min(starts[l], p.n)" } else { "0u" };
    let mut src = tridiagonal_header(starts);
    let _ = write!(
        src,
        "@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let l = gid.x + gid.y * nwg.x * {wg}u;
    if (l >= p.lines) {{ return; }}
    let base = line_base(l);
    let s0 = {s0};
    for (var s = 0u; s < s0; s++) {{
        let oq = (base + s * p.stride) * {b}u;
"
    );
    for c in 0..b {
        let _ = writeln!(src, "        out[oq + {c}u] = 0.0;");
    }
    src.push_str("    }\n    if (s0 >= p.n) { return; }\n");
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "    var pc{i}{j} = 0.0;");
        }
        let _ = writeln!(src, "    var pd{i} = 0.0;");
    }
    src.push_str(
        "    for (var s = s0; s < p.n; s++) {
        let q = base + s * p.stride;
        let o = q * __BB__u;
        let oq = q * __B__u;
"
        .replace("__BB__", &bb.to_string())
        .replace("__B__", &b.to_string())
        .as_str(),
    );
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        var m{i}{j} = diag[o + {}u];", i * b + j);
            let _ = writeln!(src, "        var r{i}{j} = up[o + {}u];", i * b + j);
        }
        let _ = writeln!(src, "        var r{i}{b} = rhs[oq + {i}u];");
    }
    src.push_str("        if (s > s0) {\n");
    for i in 0..b {
        for k in 0..b {
            let _ = writeln!(src, "            let l{i}{k} = lo[o + {}u];", i * b + k);
        }
    }
    for i in 0..b {
        for j in 0..b {
            let sum = (0..b).map(|k| format!("l{i}{k} * pc{k}{j}")).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "            m{i}{j} -= {sum};");
        }
        let sum = (0..b).map(|k| format!("l{i}{k} * pd{k}")).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "            r{i}{b} -= {sum};");
    }
    src.push_str("        }\n");
    // Elimination with partial pivoting (the first row of largest magnitude wins ties).
    for col in 0..b {
        if col + 1 < b {
            let _ = writeln!(src, "        var best{col} = abs(m{col}{col});\n        var pv{col} = {col}u;");
            for i in col + 1..b {
                let _ = writeln!(
                    src,
                    "        let a{col}{i} = abs(m{i}{col});\n        let gt{col}{i} = a{col}{i} > best{col};\n        best{col} = select(best{col}, a{col}{i}, gt{col}{i});\n        pv{col} = select(pv{col}, {i}u, gt{col}{i});"
                );
            }
            for i in col + 1..b {
                let _ = writeln!(src, "        let sw{col}{i} = pv{col} == {i}u;");
                let entries = (col..b).map(|j| format!("m{{}}{j}")).chain((0..w).map(|j| format!("r{{}}{j}")));
                for entry in entries {
                    let (top, other) = (entry.replace("{}", &col.to_string()), entry.replace("{}", &i.to_string()));
                    let _ = writeln!(
                        src,
                        "        {{ let t = {top}; {top} = select({top}, {other}, sw{col}{i}); {other} = select({other}, t, sw{col}{i}); }}"
                    );
                }
            }
        }
        let _ = writeln!(src, "        let v{col} = 1.0 / m{col}{col};");
        for i in col + 1..b {
            let _ = writeln!(src, "        let f{col}{i} = m{i}{col} * v{col};");
            for j in col + 1..b {
                let _ = writeln!(src, "        m{i}{j} -= f{col}{i} * m{col}{j};");
            }
            for j in 0..w {
                let _ = writeln!(src, "        r{i}{j} -= f{col}{i} * r{col}{j};");
            }
        }
    }
    // Back substitution of the eliminated system.
    for j in 0..w {
        for i in (0..b).rev() {
            let mut x = format!("r{i}{j}");
            for k in i + 1..b {
                let _ = write!(x, " - m{i}{k} * r{k}{j}");
            }
            let _ = writeln!(src, "        r{i}{j} = ({x}) * v{i};");
        }
    }
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        cp[o + {}u] = r{i}{j};\n        pc{i}{j} = r{i}{j};", i * b + j);
        }
        let _ = writeln!(src, "        out[oq + {i}u] = r{i}{b};\n        pd{i} = r{i}{b};");
    }
    src.push_str("    }\n");
    // The solution of the last cell is its swept right-hand side, still in `pd`.
    for i in 0..b {
        let _ = writeln!(src, "    var x{i} = pd{i};");
    }
    let _ = write!(
        src,
        "    for (var j = 2u; j <= p.n - s0; j++) {{
        let q = base + (p.n - j) * p.stride;
        let o = q * {bb}u;
        let oq = q * {b}u;
"
    );
    for i in 0..b {
        let sum = (0..b).map(|k| format!("cp[o + {}u] * x{k}", i * b + k)).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "        let y{i} = out[oq + {i}u] - ({sum});");
    }
    for i in 0..b {
        let _ = writeln!(src, "        out[oq + {i}u] = y{i};\n        x{i} = y{i};");
    }
    src.push_str("    }\n}\n");
    src
}

/// The loop form of the solve for any block size: `m` and `r` are function-local arrays. Same
/// algorithm and layout as [`tridiagonal_unrolled`].
fn tridiagonal_loops(block: u32, starts: bool, wg: u32) -> String {
    let (b, bb, w) = (block, block * block, block + 1);
    let (starts_binding, starts_decl, s0) = if starts {
        ("@group(1) @binding(6) var<storage, read> starts: array<u32>;\n", 7, "min(starts[l], p.n)")
    } else {
        ("", 6, "0u")
    };
    format!(
        "struct Params {{ n: u32, stride: u32, lines: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<storage, read_write> cp: array<f32>;
@group(1) @binding(2) var<storage, read> lo: array<f32>;
@group(1) @binding(3) var<storage, read> diag: array<f32>;
@group(1) @binding(4) var<storage, read> up: array<f32>;
@group(1) @binding(5) var<storage, read> rhs: array<f32>;
{starts_binding}@group(1) @binding({starts_decl}) var<uniform> p: Params;
fn line_base(l: u32) -> u32 {{ return (l % p.stride) + (l / p.stride) * p.stride * p.n; }}
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let l = gid.x + gid.y * nwg.x * {wg}u;
    if (l >= p.lines) {{ return; }}
    let base = line_base(l);
    let s0 = {s0};
    for (var s = 0u; s < s0; s++) {{
        let q = base + s * p.stride;
        for (var c = 0u; c < {b}u; c++) {{ out[q * {b}u + c] = 0.0; }}
    }}
    var m: array<f32, {bb}>;
    var r: array<f32, {rw}>;
    for (var s = s0; s < p.n; s++) {{
        let q = base + s * p.stride;
        for (var i = 0u; i < {b}u; i++) {{
            for (var j = 0u; j < {b}u; j++) {{
                m[i * {b}u + j] = diag[q * {bb}u + i * {b}u + j];
                r[i * {w}u + j] = up[q * {bb}u + i * {b}u + j];
            }}
            r[i * {w}u + {b}u] = rhs[q * {b}u + i];
        }}
        if (s > s0) {{
            let qp = base + (s - 1u) * p.stride;
            for (var i = 0u; i < {b}u; i++) {{
                for (var j = 0u; j < {b}u; j++) {{
                    var acc = 0.0;
                    for (var k = 0u; k < {b}u; k++) {{ acc += lo[q * {bb}u + i * {b}u + k] * cp[qp * {bb}u + k * {b}u + j]; }}
                    m[i * {b}u + j] -= acc;
                }}
                var accd = 0.0;
                for (var k = 0u; k < {b}u; k++) {{ accd += lo[q * {bb}u + i * {b}u + k] * out[qp * {b}u + k]; }}
                r[i * {w}u + {b}u] -= accd;
            }}
        }}
        for (var col = 0u; col < {b}u; col++) {{
            var pivot = col;
            var best = abs(m[col * {b}u + col]);
            for (var i = col + 1u; i < {b}u; i++) {{
                let v = abs(m[i * {b}u + col]);
                if (v > best) {{ best = v; pivot = i; }}
            }}
            if (pivot != col) {{
                for (var j = 0u; j < {b}u; j++) {{
                    let t = m[col * {b}u + j]; m[col * {b}u + j] = m[pivot * {b}u + j]; m[pivot * {b}u + j] = t;
                }}
                for (var j = 0u; j < {w}u; j++) {{
                    let t = r[col * {w}u + j]; r[col * {w}u + j] = r[pivot * {w}u + j]; r[pivot * {w}u + j] = t;
                }}
            }}
            let inv = 1.0 / m[col * {b}u + col];
            for (var i = col + 1u; i < {b}u; i++) {{
                let f = m[i * {b}u + col] * inv;
                for (var j = col; j < {b}u; j++) {{ m[i * {b}u + j] -= f * m[col * {b}u + j]; }}
                for (var j = 0u; j < {w}u; j++) {{ r[i * {w}u + j] -= f * r[col * {w}u + j]; }}
            }}
        }}
        for (var j = 0u; j < {w}u; j++) {{
            for (var back = 0u; back < {b}u; back++) {{
                let i = {b}u - 1u - back;
                var x = r[i * {w}u + j];
                for (var k = i + 1u; k < {b}u; k++) {{ x -= m[i * {b}u + k] * r[k * {w}u + j]; }}
                r[i * {w}u + j] = x / m[i * {b}u + i];
            }}
        }}
        for (var i = 0u; i < {b}u; i++) {{
            for (var j = 0u; j < {b}u; j++) {{ cp[q * {bb}u + i * {b}u + j] = r[i * {w}u + j]; }}
            out[q * {b}u + i] = r[i * {w}u + {b}u];
        }}
    }}
    let count = select(0u, p.n - s0, p.n > s0);
    for (var j = 2u; j <= count; j++) {{
        let q = base + (p.n - j) * p.stride;
        let qn = q + p.stride;
        for (var i = 0u; i < {b}u; i++) {{
            var acc = 0.0;
            for (var k = 0u; k < {b}u; k++) {{ acc += cp[q * {bb}u + i * {b}u + k] * out[qn * {b}u + k]; }}
            out[q * {b}u + i] -= acc;
        }}
    }}
}}
",
        rw = b * w
    )
}

/// Writes the indirect arguments of the next step of `iterate_until`: the template, or zeros
/// once the residual is within the tolerance (a NaN residual never is) so that the steps after
/// convergence dispatch no workgroups.
const UNTIL_GATE: &str = "struct Params { tol: f32, step: u32, words: u32 }
@group(1) @binding(0) var<storage, read> res: array<f32>;
@group(1) @binding(1) var<storage, read_write> st: array<u32>;
@group(1) @binding(2) var<storage, read_write> args: array<u32>;
@group(1) @binding(3) var<storage, read> tmpl: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let done = st[0] != 0u || res[0] <= p.tol;
    if (g.x == 0u && st[0] == 0u) {
        if (res[0] <= p.tol) { st[0] = 1u; }
        st[1] = p.step + 1u;
    }
    if (g.x < p.words) { args[g.x] = select(tmpl[g.x], 0u, done); }
}
";

impl Builtin {
    /// The WGSL source of this kernel for elements of type `ty` (`"f32"` or `"u32"`), with the
    /// parameters `tuning` gives for the GPU.
    pub(crate) fn source(self, ty: &str, tuning: &Tuning) -> Cow<'static, str> {
        let wg = tuning.elementwise_workgroup;
        match self {
            Builtin::Fill => fill_kernel(ty, wg).into(),
            Builtin::Reduce(ops, segmented) => reduce(ty, ops, segmented, tuning).into(),
            Builtin::Conv1d => include_str!("../operations/conv1d.wgsl").into(),
            Builtin::Conv1dTiled(stride, per_thread) => conv1d_tiled(stride, per_thread, tuning.conv_workgroup).into(),
            Builtin::Fft(pass) => crate::fft::source(pass).into(),
            Builtin::UntilGate => UNTIL_GATE.into(),
            Builtin::Tridiagonal { block, starts, workgroup } => tridiagonal(block as u32, starts, u32::from(workgroup)).into(),
            Builtin::NonFinite => format!(
                "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<storage, read> a: array<u32>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = gid.x + gid.y * nwg.x * {wg}u;
    if (i >= luma.size.x) {{ return; }}
    out[i] = select(0u, 1u, (a[i] & 0x7f800000u) == 0x7f800000u);
}}
"
            )
            .into(),
        }
    }
}
