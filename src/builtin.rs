//! The built-in operations, expressed as kernels that run on the same engine as custom kernels:
//! the simple ones (fills, the non-finite test, the spatial convolution, the gate of
//! `iterate_until`) are built directly as Kernel IR modules, the others are generated WGSL.
//!
//! Elementwise operations on arrays are not kernels of their own: they are recorded lazily and
//! fused into generated kernels, see [`crate::jit`]. What is left here are the
//! fills, the reduction passes over partial results, and the convolution and FFT
//! passes (generated here and in [`crate::fft`]). Every kernel declares its arguments as the
//! bindings of its module (`@group(1)` in WGSL text) and uses the `luma` grid of the prelude.

use std::borrow::Cow;
use std::fmt::Write;

use luma_core::kir::{BarrierScope, BuiltinVar, FnBuilder, KernelModule, KirType, ModuleBuilder, ScalarTy, Val};

use crate::gpu::lower::finish;
use crate::jit::ElemTy;
use crate::solve::Io;
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
    /// Rounds toward zero.
    Trunc,
    /// Rounds to the nearest integer, halves to the even one (as NumPy's `round`).
    Round,
    /// `ln(1 + x)`, accurate for small `x`.
    Log1p,
    /// `e^x - 1`, accurate for small `x`.
    Expm1,
    /// The identity: a copy of the operand (internal).
    Id,
    /// Bitwise complement (`u32` only).
    BitNot,
    /// Shift left by a constant amount `0..32` (`u32` only). The amount is part of the
    /// operation, so it is a literal of the kernel: the compiler turns a rotation by a constant
    /// into a funnel shift, which one by a value read from the uniform is not.
    Shl(u8),
    /// Logical shift right by a constant amount `0..32`.
    Shr(u8),
    /// Rotation left by a constant amount `0..32`.
    Rotl(u8),
}

/// The WGSL of the shifts and rotations by a constant (kind 0, 1, 2: `Shl`, `Shr`, `Rotl`).
fn by_constant(kind: usize, amount: u8) -> &'static str {
    static TABLES: std::sync::OnceLock<[Vec<String>; 3]> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        let table = |f: &dyn Fn(u32) -> String| (0..32).map(|k| if k == 0 { "x".to_owned() } else { f(k) }).collect();
        [
            table(&|k| format!("(x << {k}u)")),
            table(&|k| format!("(x >> {k}u)")),
            table(&|k| format!("((x << {k}u) | (x >> {}u))", 32 - k)),
        ]
    });
    &tables[kind][usize::from(amount & 31)]
}

/// `x / (2 + x)` and its square: the argument of the series of [`log1p_expr`].
const LOG1P_S: &str = "(x / (2.0 + x))";

/// `ln(1 + x)` of `f32`: for `-0.5 < x < 1` the series `2 s (1 + s^2/3 + s^4/5 + ..)` of
/// `2 atanh(s)` with `s = x / (2 + x)` (`|s| <= 1/3`, terms to `s^15`), which keeps the relative
/// accuracy at small `x` where `log(1 + x)` has none; `log(1 + x)` elsewhere, where `1 + x` is
/// exact below `-0.5` and loses at most a bit above 1. NaN fails both tests and stays NaN.
fn log1p_expr() -> &'static str {
    static EXPR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPR.get_or_init(|| {
        let (s, w) = (LOG1P_S, format!("({LOG1P_S} * {LOG1P_S})"));
        let mut series = "0.13333334".to_owned();
        for c in ["0.15384616", "0.18181819", "0.22222222", "0.2857143", "0.4", "0.6666667", "2.0"] {
            series = format!("({c} + {w} * {series})");
        }
        format!("select(log(1.0 + x), {s} * {series}, (x > -0.5) & (x < 1.0))")
    })
}

/// `e^x - 1` of `f32`: the Taylor series to `x^12 / 12!` for `|x| < 1`, `exp(x) - 1` elsewhere,
/// where the subtraction loses at most 1.4 bits.
fn expm1_expr() -> &'static str {
    static EXPR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPR.get_or_init(|| {
        let mut series = "2.0876757e-9".to_owned();
        for c in ["2.5052108e-8", "2.7557319e-7", "2.7557319e-6", "2.4801587e-5", "0.0001984127", "0.0013888889", "0.008333334", "0.041666668", "0.16666667", "0.5", "1.0"] {
            series = format!("({c} + x * {series})");
        }
        format!("select(exp(x) - 1.0, x * {series}, abs(x) < 1.0)")
    })
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
            // Metal's `tanh` overflows `exp(2x)` to `Inf / Inf = 0` from `|x| >= 44`; for `f32`,
            // `tanh(+-15)` is already `+-1.0` to the last bit.
            UnOp::Tanh => "select(tanh(x), select(-1.0, 1.0, x > 0.0), abs(x) > 15.0)",
            UnOp::Sign => "sign(x)",
            UnOp::Floor => "floor(x)",
            UnOp::Ceil => "ceil(x)",
            UnOp::Trunc => "trunc(x)",
            UnOp::Round => "round(x)",
            UnOp::Log1p => log1p_expr(),
            UnOp::Expm1 => expm1_expr(),
            UnOp::Id => "x",
            UnOp::BitNot => "~x",
            UnOp::Shl(k) => by_constant(0, k),
            UnOp::Shr(k) => by_constant(1, k),
            UnOp::Rotl(k) => by_constant(2, k),
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
            UnOp::Trunc => "trunc",
            UnOp::Round => "round",
            UnOp::Log1p => "log1p",
            UnOp::Expm1 => "expm1",
            UnOp::Id => "copy",
            UnOp::BitNot => "not",
            UnOp::Shl(_) => "shl",
            UnOp::Shr(_) => "shr",
            UnOp::Rotl(_) => "rotate_left",
        }
    }
}

impl UnOp {
    /// Whether the operation only exists for `u32`.
    pub(crate) fn is_bitwise(self) -> bool {
        matches!(self, UnOp::BitNot | UnOp::Shl(_) | UnOp::Shr(_) | UnOp::Rotl(_))
    }

    /// The operation that shifts or rotates by the constant `amount` (taken modulo 32) in place
    /// of the binary `op`, if there is one.
    pub(crate) fn by_constant(op: BinOp, amount: u32) -> Option<UnOp> {
        let k = (amount & 31) as u8;
        match op {
            BinOp::Shl => Some(UnOp::Shl(k)),
            BinOp::Shr => Some(UnOp::Shr(k)),
            BinOp::Rotl => Some(UnOp::Rotl(k)),
            _ => None,
        }
    }
}

impl UnOp {
    /// [`UnOp::expr`] for elements of type `ty`, `None` if the operation does not exist for it
    /// (`f64` has the exact built-ins, `Df32` the library ones, see [`df_library`]).
    pub(crate) fn expr_for(self, ty: ElemTy) -> Option<&'static str> {
        match ty {
            ElemTy::U32 => match self {
                UnOp::BitNot | UnOp::Id | UnOp::Shl(_) | UnOp::Shr(_) | UnOp::Rotl(_) => Some(self.expr()),
                _ => None,
            },
            ElemTy::Bool => match self {
                UnOp::Id => Some("x"),
                UnOp::BitNot => Some("(x ^ 1u)"),
                _ => None,
            },
            ElemTy::I32 => match self {
                UnOp::Neg | UnOp::Abs | UnOp::Sign | UnOp::Id | UnOp::BitNot | UnOp::Shl(_) | UnOp::Shr(_) => Some(self.expr()),
                _ => None,
            },
            ElemTy::F32 if self.is_bitwise() => None,
            ElemTy::F32 => Some(self.expr()),
            ElemTy::F64 => match self {
                UnOp::Neg | UnOp::Abs | UnOp::Sqrt | UnOp::Sign | UnOp::Floor | UnOp::Ceil | UnOp::Id => Some(self.expr()),
                // The drivers' `f64` `round` and `trunc` are not exact everywhere (Mesa's lavapipe
                // rounds -2.50000000025 to -2), so both are built from `floor` of `|x|` and exact
                // `f64` arithmetic on its remainder; a zero keeps its sign, a NaN stays NaN.
                UnOp::Round => Some(
                    "select(x, select((floor(abs(x)) + select(f64(0.0), f64(1.0), (abs(x) - floor(abs(x))) > f64(0.5) || ((abs(x) - floor(abs(x))) == f64(0.5) && floor(abs(x)) - f64(2.0) * floor(floor(abs(x)) * f64(0.5)) == f64(1.0)))), -(floor(abs(x)) + select(f64(0.0), f64(1.0), (abs(x) - floor(abs(x))) > f64(0.5) || ((abs(x) - floor(abs(x))) == f64(0.5) && floor(abs(x)) - f64(2.0) * floor(floor(abs(x)) * f64(0.5)) == f64(1.0)))), x < f64(0.0)), x != f64(0.0))",
                ),
                UnOp::Trunc => Some(
                    "select(x, select(floor(abs(x)), -floor(abs(x)), x < f64(0.0)), x != f64(0.0))",
                ),
                _ => None,
            },
            ElemTy::Df32 => match self {
                UnOp::Neg => Some("-x"),
                UnOp::Abs => Some("df_abs(x)"),
                UnOp::Sqrt => Some("df_sqrt(x)"),
                UnOp::Exp => Some("df_exp(x)"),
                UnOp::Ln => Some("df_ln(x)"),
                UnOp::Tanh => Some("df_tanh(x)"),
                UnOp::Log1p => Some("df_log1p(x)"),
                UnOp::Expm1 => Some("df_expm1(x)"),
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
    /// Bitwise operations on `u32` (shift and rotate amounts are taken modulo 32).
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Rotl,
}

impl BinOp {
    /// Whether the operation only exists for `u32`.
    pub(crate) fn is_bitwise(self) -> bool {
        matches!(self, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr | BinOp::Rotl)
    }
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
            // Scaled by the larger magnitude so that squaring neither overflows nor underflows.
            // Where that does not apply (zero, infinity, or a NaN that `max` skipped) the result is
            // `|x| + |y|`, which is NaN with a NaN operand; an infinite operand gives infinity even
            // with a NaN (IEEE). The tests are comparisons, false for NaN, as in `tanh`.
            BinOp::Hypot => "select(select(abs(x) + abs(y), max(abs(x), abs(y)) * length(vec2<f32>(x / max(abs(x), abs(y)), y / max(abs(x), abs(y)))), max(abs(x), abs(y)) > 0.0 && max(abs(x), abs(y)) < 3.4028235e38), bitcast<f32>(0x7f800000u), abs(x) > 3.4028235e38 || abs(y) > 3.4028235e38)",
            BinOp::Copysign => "bitcast<f32>((bitcast<u32>(x) & 0x7fffffffu) | (bitcast<u32>(y) & 0x80000000u))",
            BinOp::Rem => "x % y",
            BinOp::BitAnd => "x & y",
            BinOp::BitOr => "x | y",
            BinOp::BitXor => "x ^ y",
            BinOp::Shl => "x << (y & 31u)",
            BinOp::Shr => "x >> (y & 31u)",
            BinOp::Rotl => "((x << (y & 31u)) | (x >> ((32u - y) & 31u)))",
        }
    }
}

impl BinOp {
    /// [`BinOp::expr`] for elements of type `ty`, `None` if the operation does not exist for it.
    pub(crate) fn expr_for(self, ty: ElemTy) -> Option<&'static str> {
        match ty {
            ElemTy::U32 => Some(self.expr()),
            ElemTy::Bool => match self {
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Min | BinOp::Max => Some(self.expr()),
                _ => None,
            },
            ElemTy::I32 => match self {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Min | BinOp::Max | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => Some(self.expr()),
                BinOp::Shl => Some("x << (bitcast<u32>(y) & 31u)"),
                BinOp::Shr => Some("x >> (bitcast<u32>(y) & 31u)"),
                _ => None,
            },
            ElemTy::F32 => (!self.is_bitwise()).then(|| self.expr()),
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
    /// The comparison as a WGSL expression of `x` and `y`, a `u32` holding 0 or 1 (a `bool`).
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

/// A test of what kind of floating-point value an element is, read from the bits of its
/// exponent: comparisons with `NaN` or infinity are what a compiler that assumes neither exists
/// may fold away.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Class {
    /// Neither infinite nor `NaN`.
    Finite,
    Nan,
    /// Positive or negative infinity.
    Inf,
}

impl Class {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Class::Finite => "is_finite",
            Class::Nan => "is_nan",
            Class::Inf => "is_inf",
        }
    }

    /// The test as a WGSL expression of the `f32` `x`, a `u32` holding 0 or 1 (a `bool`): the bits of `x`
    /// below the sign are compared with those of infinity. They come through `luma_bits` of the
    /// double-float library, which ORs them with the guard uniform (always 0): a compiler that
    /// assumes no `NaN` or infinity arises in the arithmetic of the kernel (a device's fast
    /// math) would otherwise fold a test of the bits of a computed value to its finite answer.
    pub(crate) fn expr(self) -> &'static str {
        match self {
            Class::Finite => "select(0u, 1u, (luma_bits(x) & 0x7fffffffu) < 0x7f800000u)",
            Class::Nan => "select(0u, 1u, (luma_bits(x) & 0x7fffffffu) > 0x7f800000u)",
            Class::Inf => "select(0u, 1u, (luma_bits(x) & 0x7fffffffu) == 0x7f800000u)",
        }
    }
}

/// An `f32` with the class of the value `a` of type `ty` (see [`Class`]): `f32` itself; the high
/// word of a `Df32` (the low word cannot make a finite value of it infinite); an `f64` scaled
/// into the range of `f32` by `2^-1000` (there is no bit cast of `f64`), which keeps `NaN` and
/// infinity and turns every finite value finite.
pub(crate) fn class_operand(ty: ElemTy, a: &str) -> String {
    match ty {
        ElemTy::Df32 => format!("{a}.x"),
        ElemTy::F64 => format!("f32({a} * 9.332636185032189e-302lf)"),
        _ => a.to_owned(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ReduceOp {
    Sum,
    Min,
    Max,
    /// Sum of squares of the input; the later passes over its partial results are sums.
    SumSquares,
    /// The index of the first largest element of a line (a NaN is the largest); the partial results
    /// of a pass are (value, index) pairs. Only the axis passes of [`crate::arg`] run it, alone in
    /// its [`ReduceList`].
    ArgMax,
    /// The index of the first smallest element of a line (a NaN is the smallest), as `ArgMax`.
    ArgMin,
}

impl ReduceOp {
    /// The op the passes after the first apply to its partial results.
    fn rest(self) -> ReduceOp {
        match self {
            ReduceOp::SumSquares => ReduceOp::Sum,
            op => op,
        }
    }

    /// Whether the op carries an index with its value (see [`ReduceOp::ArgMax`]).
    pub(crate) fn is_arg(self) -> bool {
        matches!(self, ReduceOp::ArgMax | ReduceOp::ArgMin)
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
        self.fields_tagged("")
    }

    /// [`Epilogue::fields`] with `tag` between the `e` and the step's number.
    fn fields_tagged(&self, tag: &str) -> String {
        let mut out = String::new();
        for (k, step) in self.steps().enumerate() {
            if matches!(step, EpilogueStep::Binary { .. }) {
                let _ = write!(out, "e{tag}{k}: f32, ");
            }
        }
        out
    }

    /// WGSL statements applying the steps to the `f32` expression `value` (`let {prefix}{k}` is
    /// the result of step `k`) and the name of the final result (`value` if there are none).
    pub(crate) fn statements(&self, value: &str, prefix: &str) -> (String, String) {
        self.statements_with(value, prefix, |k| format!("p.e{k}"))
    }

    /// [`Epilogue::statements`] with the scalar operand of binary step `k` named `field(k)`.
    fn statements_with(&self, value: &str, prefix: &str, field: impl Fn(usize) -> String) -> (String, String) {
        let mut out = String::new();
        let mut last = value.to_owned();
        for (k, step) in self.steps().enumerate() {
            let expr = match step {
                EpilogueStep::Unary(op) => substitute(op.expr(), &last, ""),
                EpilogueStep::Binary { op, scalar_left: false } => substitute(op.expr(), &last, &field(k)),
                EpilogueStep::Binary { op, scalar_left: true } => substitute(op.expr(), &field(k), &last),
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

/// A gate folded into a reduction kernel: WGSL text, registered once and named in a kernel's cache
/// key by this handle (see [`ReduceList::with_gate`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GateId(u32);

/// The registered gates: their sources by handle, and the handles by source.
#[derive(Default)]
struct Gates {
    sources: Vec<std::sync::Arc<str>>,
    ids: crate::gpu::hash::FxMap<std::sync::Arc<str>, u32>,
}

static GATES: std::sync::LazyLock<std::sync::Mutex<Gates>> = std::sync::LazyLock::new(Default::default);

/// Storage bindings a folded gate takes, after the reduction's own: `st`, `args`, `tmpl`, `hist`,
/// `agg`, `stepv` (see [`reduce_main`]).
pub(crate) const GATE_BINDINGS: usize = 6;

/// What a gate does once the loop it belongs to has stopped, which is the same for every gate: it
/// copies the template `tmpl` of the dispatch arguments to `args`, but for the launch `st[4]`
/// (the one the gate runs in) it zeroes them, `st[3]` words of them. [`reduce_main`] defines it
/// for a gate folded into a reduction, which calls it as `luma_stopped()`.
pub(crate) const GATE_STOPPED: &str = "fn luma_stopped() {
    for (var w = 0u; w < st[3u]; w++) { args[w] = select(0u, tmpl[w], w / 3u == st[4u]); }
}
";

impl GateId {
    /// The handle of `source`, the WGSL of a gate (see [`reduce_main`] for what it defines).
    pub(crate) fn register(source: &str) -> GateId {
        let mut gates = crate::gpu::lock(&GATES);
        if let Some(&id) = gates.ids.get(source) {
            return GateId(id);
        }
        let id = gates.sources.len() as u32;
        let source: std::sync::Arc<str> = source.into();
        gates.sources.push(std::sync::Arc::clone(&source));
        gates.ids.insert(source, id);
        GateId(id)
    }

    fn source(self) -> std::sync::Arc<str> {
        std::sync::Arc::clone(&crate::gpu::lock(&GATES).sources[self.0 as usize])
    }
}

/// Most accumulators one reduction kernel computes when it reduces several operands (see
/// [`ReduceList::merged`]).
pub(crate) const MAX_ACCS: usize = 32;

/// What a merged reduction adds to its [`ReduceList`]: which operand every op reduces, and the
/// epilogue of every op. Interned (see [`Merged`]), so that the list stays a small `Copy` value.
#[derive(PartialEq, Eq, Hash, Debug)]
struct MergedSpec {
    /// The root (index into the kernel's outputs) every op reduces; empty for the passes after the
    /// first, which reduce partial results.
    roots: Vec<u8>,
    /// The epilogue of every op.
    epilogues: Vec<Epilogue>,
}

/// The registered merged specs: by handle, and the handles by spec.
#[derive(Default)]
struct MergedSpecs {
    specs: Vec<std::sync::Arc<MergedSpec>>,
    ids: crate::gpu::hash::FxMap<(Vec<u8>, Vec<Epilogue>), u32>,
}

static MERGED_SPECS: std::sync::LazyLock<std::sync::Mutex<MergedSpecs>> = std::sync::LazyLock::new(Default::default);

/// A merged reduction's roots and epilogues, by handle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct MergedId(u32);

impl MergedId {
    fn intern(roots: &[u8], epilogues: &[Epilogue]) -> MergedId {
        let mut specs = crate::gpu::lock(&MERGED_SPECS);
        let key = (roots.to_vec(), epilogues.to_vec());
        if let Some(&id) = specs.ids.get(&key) {
            return MergedId(id);
        }
        let id = specs.specs.len() as u32;
        specs.specs.push(std::sync::Arc::new(MergedSpec { roots: key.0.clone(), epilogues: key.1.clone() }));
        specs.ids.insert(key, id);
        MergedId(id)
    }

    fn spec(self) -> std::sync::Arc<MergedSpec> {
        std::sync::Arc::clone(&crate::gpu::lock(&MERGED_SPECS).specs[self.0 as usize])
    }
}

/// A [`ReduceList`] of ops over several operands, each with an epilogue of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Merged {
    id: MergedId,
    /// Whether the epilogues are applied by this pass (the last one).
    applied: bool,
}

/// Reductions computed together, in order (a kernel cache key, so a fixed-size value), and what
/// is applied to their results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceList {
    ops: [ReduceOp; MAX_ACCS],
    len: u8,
    /// Applied to every value of the pass's output; only a pass that is the last (one group,
    /// or the last workgroup of a [single-pass](ReduceList::single_pass) kernel) has one.
    epilogue: Epilogue,
    /// Whether the kernel finishes the reduction itself (see [`ReduceList::single_pass`]).
    single_pass: bool,
    /// The gate the workgroup that has the final values runs on them (see [`ReduceList::with_gate`]).
    gate: Option<GateId>,
    /// Set for a list of several operands (see [`ReduceList::merged`]): `epilogue` is not used then.
    merged: Option<Merged>,
}

impl ReduceList {
    /// `ops` must hold between 1 and [`MAX_REDUCTIONS`] ops.
    pub(crate) fn new(ops: &[ReduceOp]) -> Self {
        debug_assert!(ops.len() <= MAX_REDUCTIONS);
        Self::with_ops(ops)
    }

    fn with_ops(ops: &[ReduceOp]) -> Self {
        let mut list = ReduceList { ops: [ReduceOp::Sum; MAX_ACCS], len: 0, epilogue: Epilogue::default(), single_pass: false, gate: None, merged: None };
        for (slot, &op) in list.ops.iter_mut().zip(ops) {
            *slot = op;
            list.len += 1;
        }
        list
    }

    /// The ops `ops` (at most [`MAX_ACCS`]), op `j` reducing the root `roots[j]` of the kernel's
    /// program (all of one element type), with the epilogue `epilogues[j]` applied once the
    /// pass is the last (see [`ReduceList::finished`]).
    pub(crate) fn merged(ops: &[ReduceOp], roots: &[u8], epilogues: &[Epilogue]) -> Self {
        debug_assert!(ops.len() == roots.len() && ops.len() == epilogues.len() && ops.len() <= MAX_ACCS);
        let mut list = Self::with_ops(ops);
        list.merged = Some(Merged { id: MergedId::intern(roots, epilogues), applied: false });
        list
    }

    pub(crate) fn as_slice(&self) -> &[ReduceOp] {
        &self.ops[..usize::from(self.len)]
    }

    /// The words of every partial result cell: one per op, two (value and index) for an arg
    /// reduction.
    pub(crate) fn cell_words(&self) -> usize {
        if self.as_slice().first().is_some_and(|op| op.is_arg()) { 2 } else { self.as_slice().len() }
    }

    /// The ops of the passes after the first.
    pub(crate) fn rest(self) -> Self {
        let mut list = self;
        list.ops.iter_mut().for_each(|op| *op = op.rest());
        if let Some(merged) = list.merged {
            list.merged = Some(Merged { id: MergedId::intern(&[], &merged.id.spec().epilogues), ..merged });
        }
        list
    }

    /// These reductions with `epilogue` applied to the results. The kernel must run as a single
    /// group (the last pass), or be [single-pass](ReduceList::single_pass): its output is the
    /// final value. Not for a [merged](ReduceList::merged) list (see [`ReduceList::finished`]).
    pub(crate) fn with_epilogue(self, epilogue: Epilogue) -> Self {
        debug_assert!(self.merged.is_none());
        ReduceList { epilogue, ..self }
    }

    /// This list as the last pass applying `epilogue` to every op's value: a merged list applies
    /// the epilogues it was made with instead.
    pub(crate) fn finished(self, epilogue: Epilogue) -> Self {
        match self.merged {
            Some(merged) => ReduceList { merged: Some(Merged { applied: true, ..merged }), ..self },
            None => self.with_epilogue(epilogue),
        }
    }

    /// The kernel of a first pass over the elements that also finishes the reduction: every
    /// workgroup leaves its partial result, and the last one to do so (counted by a one-word
    /// atomic counter that it resets, see [`reduce_main`]) combines them, applies the epilogue
    /// and writes one value per op to a `result` buffer. For `groups > 1`; part of the kernel's
    /// cache key.
    pub(crate) fn single_pass(self) -> Self {
        ReduceList { single_pass: true, ..self }
    }

    pub(crate) fn is_single_pass(&self) -> bool {
        self.single_pass
    }

    /// These reductions of one op with `gate` run by the workgroup that ends up with the final
    /// value (the last one of a [single-pass](ReduceList::single_pass) kernel, the only one of
    /// a kernel of a single group): [`reduce_main`] says what the gate gets and may use. Part of
    /// the kernel's cache key.
    pub(crate) fn with_gate(self, gate: GateId) -> Self {
        debug_assert_eq!(self.as_slice().len(), 1, "a gate takes one value");
        debug_assert!(self.merged.is_none());
        ReduceList { gate: Some(gate), ..self }
    }

    pub(crate) fn gate(&self) -> Option<GateId> {
        self.gate
    }

    /// The root (output of the kernel's program) every op of a first pass of a
    /// [merged](ReduceList::merged) list reduces; `None` for any other list, whose ops all reduce
    /// the program's last value.
    pub(crate) fn roots(&self) -> Option<Vec<u8>> {
        let roots = &self.merged?.id.spec().roots;
        (!roots.is_empty()).then(|| roots.clone())
    }

    /// The epilogue op `j` applies to its value, if this is the pass that applies it.
    fn epilogue_of(&self, j: usize) -> Epilogue {
        match self.merged {
            Some(Merged { id, applied: true }) => id.spec().epilogues[j],
            Some(_) => Epilogue::default(),
            None => self.epilogue,
        }
    }

    /// The fields of the `Params` uniform that hold the scalar operands of the epilogues, in the
    /// order of the words the pass is given (see [`Epilogue::fields`]; op `j`'s step `k` is `e{j}_{k}` of a merged list).
    pub(crate) fn epilogue_fields(&self) -> String {
        match self.merged {
            None => self.epilogue.fields(),
            Some(_) => (0..self.as_slice().len()).map(|j| self.epilogue_of(j).fields_tagged(&format!("{j}_"))).collect(),
        }
    }

    /// WGSL statements applying the epilogue of op `j` to the `f32` expression `value`, and the name
    /// of the result (see [`Epilogue::statements`]).
    fn epilogue_statements(&self, j: usize, value: &str, prefix: &str) -> (String, String) {
        match self.merged {
            None => self.epilogue.statements(value, prefix),
            Some(_) => self.epilogue_of(j).statements_with(value, prefix, |k| format!("p.e{j}_{k}")),
        }
    }
}

/// One built-in kernel. Together with the element type's WGSL name this is the cache key of the
/// kernel in the context.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Builtin {
    /// `out[i] = s`; arguments: out, uniform scalar.
    Fill,
    /// Fills the words of a bit-packed `bool` array (see `ElemLayout::PACKED`): every word is
    /// all ones if `s` is 1, else 0, but for the unused bits of the last word, which stay 0;
    /// arguments: out, uniform `{s, len}` (`len` elements, `luma.size.x` the words).
    FillBits,
    /// The first pass of a reduction of a bit-packed `bool` array over its words (see
    /// [`reduce_bits`]); arguments: out, the words, uniform `{n, len}` (`n` words holding `len`
    /// elements). Element type `"u32"`; the one op of the list is the count of true elements
    /// (`Sum`), whether any (`Max`) or all (`Min`) are.
    ReduceBits(ReduceList),
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
    /// Block Thomas solves; see [`tridiagonal`]. `block` is the block size (1 = scalar), `starts`
    /// whether every line begins at its own cell, `workgroup` the invocations per workgroup,
    /// `twisted` whether two invocations solve a line (see [`thomas_plan`], [`thomas_workgroups`];
    /// only up to [`UNROLLED_BLOCK`]).
    Tridiagonal { block: u8, starts: bool, workgroup: u8, twisted: bool },
    /// Cyclic-reduction solves, one workgroup per line; see [`crate::cyclic`]. `block` and
    /// `starts` as for [`Builtin::Tridiagonal`], `workgroup` the invocations per workgroup, rows
    /// that are a multiple of `1 << shift` live in workgroup memory (`shared_rows` of them), the
    /// others in a scratch buffer.
    CyclicTridiagonal { block: u8, starts: bool, workgroup: u16, shift: u8, shared_rows: u32 },
    /// Line solves on parts that are all buffers (the fused ones are kernels of
    /// `jit::program::Tail::Solve`); see [`crate::solve`].
    SolveParts(crate::solve::SolveSpec),
    /// `out[i] = 1` where word `i` of `a` is a NaN or an infinity (as `f32` bits), else 0;
    /// arguments: out, a. Element type `"u32"`. See `jit::check_finite`.
    NonFinite,
    /// A kernel of a radix sort pass reading its keys from a buffer; see [`crate::sort`].
    /// Element type `"u32"`. `first` passes read the user's keys.
    Radix { pass: crate::sort::RadixPass, first: bool },
    /// Tiled matrix product of the operands (buffers) `a` and `b`; see [`crate::matmul`].
    /// Arguments: out, a, b, uniform `{m, n, k, kc}`. `split` slices K over the third grid
    /// dimension and writes every slice's partial product after the previous one.
    Matmul { tile: crate::matmul::MatmulTile, aligned: bool, split: bool },
    /// `out[i] = Σ_z part[z * size + i]` for `p.s` slices `z`: the sum of the partial products of
    /// a split [`Builtin::Matmul`]. Arguments: out, part, uniform `{s}`; `luma.size.x` is the
    /// elements of `out`.
    MatmulSum,
    /// Counts the set bits of every tile of a bit-packed mask; see [`crate::filter`]. Arguments:
    /// counts, mask, uniform `{len, nwords, tiles}`. Element type `"u32"`.
    FilterCount,
    /// Writes the selected elements (`words` `u32` per element) or, with `index`, their indices;
    /// see [`crate::filter`]. Arguments: out, src (unless `index`), mask, scanned tile counts,
    /// uniform `{len, nwords, tiles}`. Element type `"u32"`.
    FilterScatter { words: u8, index: bool },
    /// Moves 32 x 32 tiles of a materialized axis permutation of one array through workgroup
    /// memory; see `jit::transpose`. Arguments: out, src, uniform `Params`. Element type `"u32"`
    /// (any 4-byte element) or `"vec2<f32>"` (`Df32`).
    Transpose,
}

/// One reduction pass over `a` for every op of `ops`: each workgroup reduces its share of the `p.n`
/// elements into `out[j * groups + group]`, `groups` being [`Tuning::reduce_groups`]; a workgroup
/// takes [`Tuning::reduce_block`] elements at a time, strided over the array. `segmented` passes
/// reduce the partial results of an earlier pass, op `j` over `a[j * p.n ..]`; otherwise every op
/// reads the same `p.n` elements. A [single-pass](ReduceList::single_pass) kernel binds the
/// counter and the final values (`result`) after `a` (see [`reduce_main`]).
fn reduce(ty: &str, ops: ReduceList, segmented: bool, tuning: &Tuning) -> String {
    // The double-float kernel takes the guard of its arithmetic (see `df_library`) as a uniform.
    let fields = ops.epilogue_fields();
    let (elem, params) = match ty {
        "df32" => (ReduceElem::Df32 { guard: "p.guard", fma: tuning.df_fma }, "struct Params { n: u32, guard: u32 }".to_owned()),
        other if fields.is_empty() => (ReduceElem::Plain(other), "struct Params { n: u32 }".to_owned()),
        other => (ReduceElem::Plain(other), format!("struct Params {{ n: u32, {fields}}}")),
    };
    let ty = elem.storage_ty();
    let single_pass = ops.is_single_pass().then_some(2);
    let uniform = if single_pass.is_some() { 4 } else { 2 };
    format!(
        "{params}
@group(1) @binding(0) var<storage, read_write> out: array<{ty}>;
@group(1) @binding(1) var<storage, read> a: array<{ty}>;
@group(1) @binding({uniform}) var<uniform> p: Params;

fn load_at(i: u32) -> {ty} {{ return a[i]; }}
{}",
        reduce_main(elem, ops, segmented, "load_at", tuning, single_pass, None)
    )
}

/// The first pass of a reduction of a bit-packed `bool` array over its words, so a pass reads
/// 32 times fewer bytes than one over the elements: every word is replaced by what it says (for
/// `Sum`, the number of its true elements, `countOneBits`; for `Max` whether any is; for `Min`
/// whether all are, `p.len` being the elements, so that the last word is compared with the
/// bits it holds), and the values are reduced as ordinary `u32`s.
fn reduce_bits(ops: ReduceList, tuning: &Tuning) -> String {
    let op = ops.as_slice().first().copied().unwrap_or(ReduceOp::Sum);
    let value = match op {
        ReduceOp::Sum | ReduceOp::SumSquares => "countOneBits(w)",
        ReduceOp::ArgMax | ReduceOp::ArgMin => unreachable!("boolean words have no arg reductions"),
        ReduceOp::Max => "select(0u, 1u, w != 0u)",
        ReduceOp::Min => "select(0u, 1u, w == select(0xFFFFFFFFu, (1u << (p.len & 31u)) - 1u, i == p.n - 1u && (p.len & 31u) != 0u))",
    };
    format!(
        "struct Params {{ n: u32, len: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<storage, read> a: array<u32>;
@group(1) @binding(2) var<uniform> p: Params;

fn load_at(i: u32) -> u32 {{ let w = a[i]; return {value}; }}
{}",
        reduce_main(ReduceElem::Plain("u32"), ops, false, "load_at", tuning, None, None)
    )
}

/// The element type a reduction pass works on.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReduceElem<'a> {
    /// A type WGSL adds, compares and squares itself.
    Plain(&'a str),
    /// A double-float (`vec2<f32>`) combined with the library of [`df_library`]; `guard` names
    /// the uniform that is always 0.
    Df32 { guard: &'a str, fma: bool },
}

impl<'a> ReduceElem<'a> {
    fn storage_ty(self) -> &'a str {
        match self {
            ReduceElem::Plain(ty) => ty,
            ReduceElem::Df32 { .. } => "vec2<f32>",
        }
    }

    /// The expression combining accumulators `u` and `v`.
    pub(crate) fn combine(self, op: ReduceOp) -> &'static str {
        match (self, op) {
            (ReduceElem::Plain(_), ReduceOp::Sum | ReduceOp::SumSquares) => "u + v",
            (ReduceElem::Plain(_), ReduceOp::Min) => "min(u, v)",
            (ReduceElem::Plain(_), ReduceOp::Max) => "max(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Sum | ReduceOp::SumSquares) => "df_add(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Min) => "df_min(u, v)",
            (ReduceElem::Df32 { .. }, ReduceOp::Max) => "df_max(u, v)",
            (_, ReduceOp::ArgMax | ReduceOp::ArgMin) => unreachable!("arg reductions have their own passes (`crate::arg`)"),
        }
    }

    /// The square of the expression `e`.
    pub(crate) fn square(self, e: &str) -> String {
        match self {
            ReduceElem::Plain(_) => format!("{e} * {e}"),
            ReduceElem::Df32 { .. } => format!("df_mul({e}, {e})"),
        }
    }

    /// Whether the workgroup built-ins (`subgroupAdd`, ...) work on the type.
    fn subgroups(self) -> bool {
        matches!(self, ReduceElem::Plain("f32" | "u32" | "i32" | "vec2<f32>" | "vec4<f32>"))
    }

    fn library(self) -> String {
        match self {
            ReduceElem::Plain(_) => String::new(),
            ReduceElem::Df32 { guard, fma } => df_library(guard, fma, false),
        }
    }
}

/// Double-float arithmetic on `vec2<f32>` `(hi, lo)`. Every intermediate of an error-free
/// transformation goes through `luma_o`, which ORs the bits with the uniform `guard` (always 0,
/// but unknown to the shader compiler): no reassociation or fast-math contraction can cancel
/// the transformations. (It is an OR, not an XOR, so that `luma_o(luma_o(x))`, as in a cast
/// feeding a cast, cannot fold back to `x`: the guard would cancel with itself.)
///
/// The error of a product is exact either way: with `fma` (the device's `fma` is fused, see
/// [`Tuning::df_fma`]) it is `fma(a, b, -p)`, otherwise Dekker's splitting of both factors costs
/// about ten times as many operations. Division and square root start from the hardware
/// quotient or root of the high words and add one correction computed from an exact remainder
/// (an accuracy of about 2^-45), instead of three Newton steps of full double-float products
/// and sums.
///
/// `sloppy_add` (`fast_math`) adds with one error-free sum of the high words and a plain sum of
/// the low ones, about half the operations: the error stays below 2^-47 of the larger operand,
/// but the relative error of a sum grows with the cancellation (a result 2^k times smaller than
/// the operands has about 2^(k-47)), where the default addition keeps 2^-46 of the result.
pub(crate) fn df_library(guard: &str, fma: bool, sloppy_add: bool) -> String {
    const ADD_ACCURATE: &str = "fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x); let t = df_two_sum(a.y, b.y);
    let c = df_quick(s.x, luma_o(s.y + t.x)); return df_quick(c.x, luma_o(c.y + t.y));
}
";
    const ADD_SLOPPY: &str = "fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x); return df_quick(s.x, luma_o(s.y + luma_o(a.y + b.y)));
}
";
    const COMMON: &str = "fn luma_o(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | {G}); }
fn luma_bits(x: f32) -> u32 { return bitcast<u32>(x) | {G}; }
fn df_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = luma_o(a + b); let bb = luma_o(s - a);
    let da = luma_o(a - luma_o(s - bb)); let db = luma_o(b - bb);
    return vec2<f32>(s, da + db);
}
fn df_quick(a: f32, b: f32) -> vec2<f32> { let s = luma_o(a + b); return vec2<f32>(s, luma_o(b - luma_o(s - a))); }
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = df_two_prod(a.x, b.x); return df_quick(p.x, luma_o(p.y + luma_o(a.x * b.y + a.y * b.x)));
}
fn df_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x;
    let p = df_two_prod(b.x, q1);
    let r = luma_o(luma_o(luma_o(a.x - p.x) - p.y) + luma_o(a.y - luma_o(b.y * q1)));
    return df_quick(q1, r / b.x);
}
fn df_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) { return vec2<f32>(sqrt(a.x), 0.0); }
    let s = sqrt(a.x);
    let p = df_two_prod(s, s);
    let e = luma_o(luma_o(luma_o(a.x - p.x) - p.y) + a.y);
    return df_quick(s, e / (2.0 * s));
}
fn df_lt(a: vec2<f32>, b: vec2<f32>) -> bool { return (a.x < b.x) | ((a.x == b.x) & (a.y < b.y)); }
fn df_le(a: vec2<f32>, b: vec2<f32>) -> bool { return (a.x < b.x) | ((a.x == b.x) & (a.y <= b.y)); }
fn df_eq(a: vec2<f32>, b: vec2<f32>) -> bool { return (a.x == b.x) & (a.y == b.y); }
fn df_min(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return select(b, a, df_lt(a, b)); }
fn df_max(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return select(a, b, df_lt(a, b)); }
fn df_abs(a: vec2<f32>) -> vec2<f32> { return select(a, -a, (a.x < 0.0) | ((a.x == 0.0) & (a.y < 0.0))); }
";
    const PRODUCT_FMA: &str = "fn df_two_prod(a: f32, b: f32) -> vec2<f32> { let p = luma_o(a * b); return vec2<f32>(p, fma(a, b, -p)); }
";
    const PRODUCT_DEKKER: &str = "fn df_split(a: f32) -> vec2<f32> { let t = luma_o(4097.0 * a); let hi = luma_o(t - luma_o(t - a)); return vec2<f32>(hi, luma_o(a - hi)); }
fn df_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = luma_o(a * b); let x = df_split(a); let y = df_split(b);
    let e1 = luma_o(luma_o(x.x * y.x) - p); let e2 = luma_o(e1 + luma_o(x.x * y.y)); let e3 = luma_o(e2 + luma_o(x.y * y.x));
    return vec2<f32>(p, luma_o(e3 + luma_o(x.y * y.y)));
}
";
    format!(
        "{COMMON}{}{}",
        if sloppy_add { ADD_SLOPPY } else { ADD_ACCURATE },
        if fma { PRODUCT_FMA } else { PRODUCT_DEKKER }
    )
    .replace("{G}", guard)
}

/// `exp`, `ln`, `tanh`, `log1p`, `expm1` and `pow` on `vec2<f32>` double-floats, on top of [`df_library`] (a kernel
/// that calls them defines both; see `codegen::library_for`).
///
/// Every function is accurate to a relative 2^-43 (`tests/df32_library.rs` measures 4.3e-14 at
/// most) for results of about `1e-25` and up. Below that the low word of a `Df32` is under the
/// smallest normal `f32` and loses bits whatever the function; a result below `2^-126` is 0 and
/// one above `f32::MAX` is infinity.
///
/// - `exp`: `x = k ln 2 + r`, `|r| <= ln 2 / 2`, with `ln 2` in three words (the first two of 16
///   bits, so that `k` times them is exact). `e^r - 1` is a Taylor series of `r / 16` (the terms
///   from the fourth power on in `f32`) squared four times as `(1 + p)^2 - 1 = 2p + p^2`, which
///   keeps the relative accuracy of the small values; the result is scaled by `2^k`.
/// - `ln`: `x = 2^e m`, `m` in `[1/sqrt 2, sqrt 2)`, and `ln m = 2 atanh((m - 1) / (m + 1))`, a
///   series in `z^2`, `|z| <= 0.172`: the relative accuracy holds near `x = 1`.
/// - `tanh`: `(e^{2|x|} - 1) / (e^{2|x|} + 1)` from the same `e^r - 1`, so small arguments keep
///   their relative accuracy; `1` from `|x| > 20`.
/// - `log1p`: for `1 + x` in `[1/sqrt 2, sqrt 2)` the same series with `z = x / (2 + x)` taken
///   from `x` itself (no cancellation), else `ln(1 + x)`.
/// - `expm1`: the `e^r - 1` of the reduction of `exp`, plus `2^k` minus 1 where `k != 0`.
/// - `pow(x, y)` is `2^(y e) * exp(y ln m)` for `x = 2^e m` with `y e` split exactly into an
///   integer and a fraction, so that the error does not grow with `y ln x` (it is that of an
///   `exp` and an `ln` and a product of at most `y ln m`, `|ln m| <= 0.35`). `NaN` for a negative
///   base, `x^0 = 1`, and the infinite and zero cases of IEEE `pow` for a base of 0 or infinity.
pub(crate) const DF_MATH_LIBRARY: &str = "struct DfLog { e: f32, l: vec2<f32> }
fn df_isnan(a: f32) -> bool { return (bitcast<u32>(a) & 0x7fffffffu) > 0x7f800000u; }
fn df_inf() -> f32 { return bitcast<f32>(0x7f800000u); }
fn df_scale(a: vec2<f32>, k: i32) -> vec2<f32> {
    let k1 = k >> 1u; let k2 = k - k1;
    let s1 = bitcast<f32>(u32(k1 + 127) << 23u); let s2 = bitcast<f32>(u32(k2 + 127) << 23u);
    let t = vec2<f32>(luma_o(a.x * s1), luma_o(a.y * s1));
    return vec2<f32>(luma_o(t.x * s2), luma_o(t.y * s2));
}
fn df_expm1_small(r: vec2<f32>) -> vec2<f32> {
    let u = r * 0.0625;
    let s = df_mul(u, u);
    let h = u.x;
    let c = 0.041666668 + h * (0.008333334 + h * (0.0013888889 + h * (0.00019841270 + h * 0.000024801587)));
    let v = df_add(vec2<f32>(0.16666667, -4.9670539e-9), vec2<f32>(h * c, 0.0));
    let w = df_add(vec2<f32>(0.5, 0.0), df_mul(u, v));
    var p = df_add(u, df_mul(s, w));
    for (var i = 0; i < 4; i++) { p = df_add(p * 2.0, df_mul(p, p)); }
    return p;
}
fn df_exp_reduce(x: vec2<f32>, k: f32) -> vec2<f32> {
    let a = df_two_sum(x.x - k * 0.693145752, x.y);
    return df_add(a, df_two_sum(-(k * 1.42861973e-6), -(k * -1.29053200e-11)));
}
fn df_exp_scaled(x: vec2<f32>, extra: f32) -> vec2<f32> {
    let k = round(x.x * 1.44269502);
    let e = df_add(vec2<f32>(1.0, 0.0), df_expm1_small(df_exp_reduce(x, k)));
    let v = df_scale(e, i32(clamp(k + extra, -250.0, 250.0)));
    if (v.x < 1.17549435e-38) { return vec2<f32>(0.0, 0.0); }
    if (v.x > 3.40282347e38) { return vec2<f32>(df_inf(), 0.0); }
    return v;
}
fn df_exp(x: vec2<f32>) -> vec2<f32> {
    if (df_isnan(x.x)) { return vec2<f32>(x.x, 0.0); }
    if (x.x > 88.722) { return vec2<f32>(df_inf(), 0.0); }
    if (x.x < -87.33) { return vec2<f32>(0.0, 0.0); }
    return df_exp_scaled(x, 0.0);
}
fn df_atanh2(z: vec2<f32>) -> vec2<f32> {
    let w = df_mul(z, z);
    let h = w.x;
    let c = 0.11111111 + h * (0.09090909 + h * (0.07692308 + h * (0.06666667 + h * (0.05882353 + h * 0.05263158))));
    let q7 = df_add(vec2<f32>(0.14285715, -6.3862120e-9), vec2<f32>(h * c, 0.0));
    let q5 = df_add(vec2<f32>(0.2, -2.9802323e-9), df_mul(w, q7));
    let q3 = df_add(vec2<f32>(0.33333334, -9.9341078e-9), df_mul(w, q5));
    let q1 = df_add(vec2<f32>(1.0, 0.0), df_mul(w, q3));
    return df_mul(z, q1) * 2.0;
}
fn df_log_split(a: vec2<f32>) -> DfLog {
    let small = a.x < 1.17549435e-38;
    let x = select(a, a * 1.84467441e19, small);
    let bits = bitcast<u32>(x.x);
    let big = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u) > 1.41421354;
    let e = i32(bits >> 23u) - 127 + select(0, 1, big);
    let m = df_scale(x, -e);
    let z = df_div(df_add(m, vec2<f32>(-1.0, 0.0)), df_add(m, vec2<f32>(1.0, 0.0)));
    return DfLog(f32(e - select(0, 64, small)), df_atanh2(z));
}
fn df_ln(x: vec2<f32>) -> vec2<f32> {
    if (df_isnan(x.x)) { return vec2<f32>(x.x, 0.0); }
    if (x.x < 0.0) { return vec2<f32>(bitcast<f32>(0x7fc00000u), 0.0); }
    if (x.x == 0.0) { return vec2<f32>(-df_inf(), 0.0); }
    if (x.x > 3.40282347e38) { return vec2<f32>(x.x, 0.0); }
    let s = df_log_split(x);
    return df_add(df_mul(vec2<f32>(s.e, 0.0), vec2<f32>(0.693147182, -1.90465421e-9)), s.l);
}
fn df_log1p(x: vec2<f32>) -> vec2<f32> {
    if (df_isnan(x.x)) { return vec2<f32>(x.x, 0.0); }
    if ((x.x < -1.0) | ((x.x == -1.0) & (x.y < 0.0))) { return vec2<f32>(bitcast<f32>(0x7fc00000u), 0.0); }
    if ((x.x == -1.0) & (x.y == 0.0)) { return vec2<f32>(-df_inf(), 0.0); }
    if (x.x > 3.40282347e38) { return vec2<f32>(x.x, 0.0); }
    if ((x.x > -0.29) & (x.x < 0.41)) { return df_atanh2(df_div(x, df_add(x, vec2<f32>(2.0, 0.0)))); }
    return df_ln(df_add(vec2<f32>(1.0, 0.0), x));
}
fn df_expm1(x: vec2<f32>) -> vec2<f32> {
    if (df_isnan(x.x)) { return vec2<f32>(x.x, 0.0); }
    if (x.x > 88.722) { return vec2<f32>(df_inf(), 0.0); }
    if (x.x < -87.33) { return vec2<f32>(-1.0, 0.0); }
    let k = round(x.x * 1.44269502);
    let p = df_expm1_small(df_exp_reduce(x, k));
    if (k == 0.0) { return p; }
    let e = df_scale(df_add(vec2<f32>(1.0, 0.0), p), i32(k));
    if (e.x > 3.40282347e38) { return vec2<f32>(df_inf(), 0.0); }
    return df_add(e, vec2<f32>(-1.0, 0.0));
}
fn df_tanh(x: vec2<f32>) -> vec2<f32> {
    if (df_isnan(x.x)) { return vec2<f32>(x.x, 0.0); }
    let neg = (bitcast<u32>(x.x) >> 31u) != 0u;
    let a = select(x, -x, neg);
    if (a.x > 20.0) { return select(vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0), neg); }
    let z = a * 2.0;
    let k = round(z.x * 1.44269502);
    let p = df_expm1_small(df_exp_reduce(z, k));
    var t = p;
    if (k != 0.0) { t = df_add(df_scale(df_add(vec2<f32>(1.0, 0.0), p), i32(k)), vec2<f32>(-1.0, 0.0)); }
    let th = df_div(t, df_add(t, vec2<f32>(2.0, 0.0)));
    return select(th, -th, neg);
}
fn df_pow(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let nan = vec2<f32>(bitcast<f32>(0x7fc00000u), 0.0);
    if (y.x == 0.0) { return vec2<f32>(1.0, 0.0); }
    if (df_isnan(x.x) | df_isnan(y.x) | (x.x < 0.0)) { return nan; }
    let zero = vec2<f32>(0.0, 0.0);
    let inf = vec2<f32>(df_inf(), 0.0);
    if (abs(y.x) > 1.0e30) {
        if ((x.x == 1.0) & (x.y == 0.0)) { return vec2<f32>(1.0, 0.0); }
        let grows = (x.x > 1.0) | ((x.x == 1.0) & (x.y > 0.0));
        return select(zero, inf, grows == (y.x > 0.0));
    }
    if (x.x == 0.0) { return select(inf, zero, y.x > 0.0); }
    if (x.x > 3.40282347e38) { return select(zero, inf, y.x > 0.0); }
    let s = df_log_split(x);
    let a = df_two_prod(y.x, s.e);
    let b = df_two_prod(y.y, s.e);
    if (abs(a.x) > 400.0) { return select(zero, inf, a.x > 0.0); }
    let n = round(a.x);
    let f = df_add(df_two_sum(a.x - n, b.x), df_two_sum(a.y, b.y));
    let arg = df_add(df_mul(f, vec2<f32>(0.693147182, -1.90465421e-9)), df_mul(y, s.l));
    if (abs(arg.x) > 100.0) { return select(zero, inf, n * 0.693147182 + arg.x > 0.0); }
    return df_exp_scaled(arg, n);
}
";

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

/// One stage of a reduction kernel (see [`reduce_phase`]): the `n` elements it reduces, which
/// workgroup this is among the `groups` sharing them, and where the values go.
struct Phase<'a> {
    /// WGSL expressions: the element count, the workgroup's index and the number of workgroups.
    n: &'a str,
    group: &'a str,
    groups: &'a str,
    /// Whether op `j` reduces `load(j * segment + i)` (the partial results of an earlier pass)
    /// instead of `load(i)`.
    segmented: bool,
    segment: &'a str,
    load: &'a str,
    /// The statement storing the WGSL expression (second argument) as the value of op `j`.
    store: &'a dyn Fn(usize, &str) -> String,
    /// The list whose epilogues are applied to the values before they are stored, if any.
    epilogue: Option<ReduceList>,
}

/// The statements of a stage of a reduction kernel computing every op of `ops`: every
/// invocation accumulates its elements of the strided blocks, then the workgroup combines the
/// accumulators, within subgroups with `subgroupAdd` and friends and across them through
/// workgroup memory when `subgroups`, by a shared-memory tree otherwise, and one invocation
/// stores the values. Expects `lid`, `zero`, the combining functions `c{j}` and, with
/// `subgroups`, `lane`, `sg` and the workgroup array `sh` to exist.
fn reduce_phase(elem: ReduceElem<'_>, ops: ReduceList, phase: &Phase<'_>, subgroups: bool, tuning: &Tuning) -> String {
    let (wg, block) = (REDUCE_WORKGROUP, tuning.reduce_block());
    let per_thread = tuning.reduce_per_thread;
    let Phase { n, group, groups, segmented, segment, load, store, epilogue } = phase;
    let mut inits = String::new();
    let mut loop_body = String::new();
    if let Some(roots) = ops.roots().filter(|_| !segmented) {
        (inits, loop_body) = merged_loop(elem, ops, &roots, phase, tuning);
    } else {
        // Without segments every op reduces the same elements: load each once.
        if !segmented {
            loop_body += &format!("            let e0 = {load}(base);\n");
            for s in 1..per_thread {
                loop_body += &format!(
                    "            let in{s} = base + {wg}u * {s}u < {n};\n            var e{s} = e0;\n            if (in{s}) {{ e{s} = {load}(base + {wg}u * {s}u); }}\n"
                );
            }
        }
        for (j, &op) in ops.as_slice().iter().enumerate() {
            let segment = if *segmented { format!("{j}u * {segment} + ") } else { String::new() };
            if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                unreachable!("arg reductions have their own passes (`crate::arg`)");
            }
            inits += &format!("    var acc{j} = zero;\n");
            for s in 0..per_thread {
                if *segmented {
                    if s == 0 {
                        loop_body += &format!("            let e{j}_0 = {load}({segment}base);\n");
                    } else {
                        loop_body += &format!(
                            "            let in{j}_{s} = base + {wg}u * {s}u < {n};\n            var e{j}_{s} = e{j}_0;\n            if (in{j}_{s}) {{ e{j}_{s} = {load}({segment}(base + {wg}u * {s}u)); }}\n"
                        );
                    }
                }
                let e = if *segmented { format!("e{j}_{s}") } else { format!("e{s}") };
                // Out-of-range slots: sums take zero, min/max reuse the first (valid) element.
                let value = match op {
                    ReduceOp::Sum => {
                        if s == 0 { e } else { format!("select(zero, {e}, base + {wg}u * {s}u < {n})") }
                    }
                    ReduceOp::SumSquares => {
                        let sq = elem.square(&e);
                        if s == 0 { sq } else { format!("select(zero, {sq}, base + {wg}u * {s}u < {n})") }
                    }
                    ReduceOp::Min | ReduceOp::Max => e,
                    ReduceOp::ArgMax | ReduceOp::ArgMin => unreachable!(),
                };
                loop_body += &format!("            let v{j}_{s} = {value};\n");
            }
            let values: Vec<String> = (0..per_thread).map(|s| format!("v{j}_{s}")).collect();
            let pw = pairwise(&format!("c{j}"), &values);
            loop_body += &match op {
                ReduceOp::Min | ReduceOp::Max => {
                    format!("            if (b == {group} * {block}u) {{ acc{j} = {pw}; }} else {{ acc{j} = c{j}(acc{j}, {pw}); }}\n")
                }
                _ => format!("            acc{j} = c{j}(acc{j}, {pw});\n"),
            };
        }
    }
    let mut finish = String::new();
    if subgroups {
        let mut lanes = String::new();
        let mut first = String::new();
        for (j, &op) in ops.as_slice().iter().enumerate() {
            let (group_op, ident) = match op {
                ReduceOp::Sum | ReduceOp::SumSquares => ("subgroupAdd", "zero".to_owned()),
                ReduceOp::Min => ("subgroupMin", format!("sh[{}u]", j as u32 * MAX_SUBGROUPS)),
                ReduceOp::Max => ("subgroupMax", format!("sh[{}u]", j as u32 * MAX_SUBGROUPS)),
                ReduceOp::ArgMax | ReduceOp::ArgMin => unreachable!("arg reductions have their own passes (`crate::arg`)"),
            };
            if matches!(op, ReduceOp::Min | ReduceOp::Max) {
                lanes += &format!("    let first{j} = subgroupBroadcastFirst(acc{j});\n    if ({group} * {block}u + lid >= {n}) {{ acc{j} = first{j}; }}\n");
            }
            lanes += &format!("    let r{j} = {group_op}(acc{j});\n    if (lane == 0u) {{ sh[{}u + sgid] = r{j}; }}\n", j as u32 * MAX_SUBGROUPS);
            let (lets, fin) = epilogue.map_or_else(|| (String::new(), format!("f{j}")), |list| list.epilogue_statements(j, &format!("f{j}"), &format!("g{j}_")));
            let store = store(j, &fin);
            first += &format!(
                "        var t{j} = {ident};
        for (var k = lane; k < nsg; k += sg) {{ t{j} = c{j}(t{j}, sh[{}u + k]); }}
        let f{j} = {group_op}(t{j});
        if (lane == 0u) {{ {lets}{store} }}
",
                j as u32 * MAX_SUBGROUPS
            );
        }
        finish += &format!(
            "    let sgid = lid / sg;
{lanes}    workgroupBarrier();
    if (sgid == 0u) {{
        let nsg = (min({n} - {group} * {block}u, {wg}u) + sg - 1u) / sg;
{first}    }}
"
        );
    } else {
        for j in 0..ops.as_slice().len() {
            let (lets, fin) = epilogue.map_or_else(|| (String::new(), "sh[0]".to_owned()), |list| list.epilogue_statements(j, "sh[0]", &format!("g{j}_")));
            let store = store(j, &fin);
            finish += &format!(
                "    sh[lid] = acc{j};
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid < s && {group} * {block}u + lid + s < {n}) {{ sh[lid] = c{j}(sh[lid], sh[lid + s]); }}
        workgroupBarrier();
    }}
    if (lid == 0u) {{ {lets}{store} }}
    workgroupBarrier();
",
                half = wg / 2
            );
        }
    }
    format!(
        "{inits}    for (var b = {group} * {block}u; b < {n}; b += {groups} * {block}u) {{
        let base = b + lid;
        if (base < {n}) {{
{loop_body}        }}
    }}
{finish}"
    )
}

/// The accumulation loop of a first pass over several operands (a [merged](ReduceList::merged) list, whose op `j`
/// reduces root `roots[j]` of the array `load(i)` returns): the accumulator declarations and the loop body. An
/// invocation computes the roots of `per_thread` elements one after the other, folding every op's values
/// in the tree [`pairwise`] makes of them, so that only a few values per op are live at once and the results are the
/// bits a separate reduction of each root gives. An element past the end is read at `base` instead (the value a
/// separate reduction substitutes for it).
fn merged_loop(elem: ReduceElem<'_>, ops: ReduceList, roots: &[u8], phase: &Phase<'_>, tuning: &Tuning) -> (String, String) {
    let (wg, block) = (REDUCE_WORKGROUP, tuning.reduce_block());
    let per_thread = tuning.reduce_per_thread as usize;
    let Phase { n, group, load, .. } = phase;
    let count = ops.as_slice().len();
    let mut inits = String::new();
    for j in 0..count {
        inits += &format!("    var acc{j} = zero;\n");
    }
    // The names of the value each op holds for the elements `lo..hi` once emitted.
    let name = |j: usize, lo: usize, hi: usize| if hi - lo == 1 { format!("v{j}_{lo}") } else { format!("w{j}_{lo}_{hi}") };
    fn emit(
        out: &mut String,
        lo: usize,
        hi: usize,
        leaf: &dyn Fn(&mut String, usize),
        node: &dyn Fn(&mut String, usize, usize, usize),
    ) {
        if hi - lo == 1 {
            leaf(out, lo);
            return;
        }
        // `pairwise` splits a list in the middle, the first half the shorter.
        let mid = lo + (hi - lo) / 2;
        emit(out, lo, mid, leaf, node);
        emit(out, mid, hi, leaf, node);
        node(out, lo, mid, hi);
    }
    let leaf = |out: &mut String, s: usize| {
        if s == 0 {
            *out += &format!("            let r0 = {load}(base);\n");
        } else {
            *out += &format!("            let in{s} = base + {wg}u * {s}u < {n};\n            let r{s} = {load}(select(base, base + {wg}u * {s}u, in{s}));\n");
        }
        for (j, &op) in ops.as_slice().iter().enumerate() {
            let e = format!("r{s}[{}]", roots[j]);
            let value = match op {
                ReduceOp::Sum => {
                    if s == 0 { e } else { format!("select(zero, {e}, in{s})") }
                }
                ReduceOp::SumSquares => {
                    let sq = elem.square(&e);
                    if s == 0 { sq } else { format!("select(zero, {sq}, in{s})") }
                }
                ReduceOp::Min | ReduceOp::Max => e,
                ReduceOp::ArgMax | ReduceOp::ArgMin => unreachable!("arg reductions have their own passes (`crate::arg`)"),
            };
            *out += &format!("            let v{j}_{s} = {value};\n");
        }
    };
    let combine = |out: &mut String, lo: usize, mid: usize, hi: usize| {
        for j in 0..count {
            *out += &format!("            let w{j}_{lo}_{hi} = c{j}({}, {});\n", name(j, lo, mid), name(j, mid, hi));
        }
    };
    let mut loop_body = String::new();
    emit(&mut loop_body, 0, per_thread, &leaf, &combine);
    for (j, &op) in ops.as_slice().iter().enumerate() {
        let pw = name(j, 0, per_thread);
        loop_body += &match op {
            ReduceOp::Min | ReduceOp::Max => {
                format!("            if (b == {group} * {block}u) {{ acc{j} = {pw}; }} else {{ acc{j} = c{j}(acc{j}, {pw}); }}\n")
            }
            _ => format!("            acc{j} = c{j}(acc{j}, {pw});\n"),
        };
    }
    (inits, loop_body)
}

/// The entry point of a reduction pass computing every op of `ops` (see [`reduce`]). Element
/// `i` comes from `{load}(i)`, and a `Params` uniform `p` with the element count `n` is bound.
/// Every invocation accumulates its elements of the strided blocks, then the workgroup combines
/// the accumulators: within subgroups with `subgroupAdd` and friends and across them through
/// workgroup memory when the device has subgroups, by a shared-memory tree otherwise. Workgroup
/// `group` of `groups` leaves the partial result of op `j` in `out[j * groups + group]`.
///
/// With `single_pass` (the binding of a `luma_counter: array<atomic<u32>>` of one word that is
/// zero; `luma_result: array<{ty}>` takes the next binding, see [`ReduceList::single_pass`])
/// the kernel also finishes the reduction, as the classic last-block reduction does: once its
/// partial results are written, a workgroup fences them (`storageBarrier`) and takes a ticket
/// from the counter; the workgroup holding the last ticket (every workgroup that has anything
/// to reduce takes one; those of the folded grid past `groups` leave at once, without) fences
/// again, reduces `out[j * groups ..]` exactly as a pass over the partial results of a first one
/// would (same accumulation order, so the values are the same bits), applies the epilogue of
/// `ops`, writes `luma_result[j]` and resets the counter for the next launch. Only for a
/// first pass (`!segmented`); the partial results then carry no epilogue.
///
/// A reduction with a [gate](ReduceList::with_gate) (one `f32` op, bindings `gate`.. `gate + GATE_BINDINGS`
/// after `luma_result`: `st`, `args`, `tmpl`, `hist`, `agg`, `stepv`) runs the gate on the final value, in
/// invocation 0 of the workgroup that has it (the last one of a single-pass kernel, the only one
/// of a kernel of one group). The gate's text defines `luma_gate(v: f32)`; [`GATE_STOPPED`] is added to it. A
/// workgroup that finds `st[0]` set (the loop stopped) leaves at once, workgroup 0 after
/// `luma_stopped()`, so that nothing of the stopped steps is computed or stored.
pub(crate) fn reduce_main(elem: ReduceElem<'_>, ops: ReduceList, segmented: bool, load: &str, tuning: &Tuning, single_pass: Option<u32>, gate: Option<u32>) -> String {
    debug_assert!(single_pass.is_none() || !segmented, "only a first pass can finish the reduction itself");
    debug_assert!(gate.is_none() || (!segmented && ops.gate().is_some()), "a gate needs a first pass with a gate");
    let ty = elem.storage_ty();
    let (wg, block) = (REDUCE_WORKGROUP, tuning.reduce_block());
    let mut combines = String::new();
    for (j, &op) in ops.as_slice().iter().enumerate() {
        let combine = elem.combine(op);
        combines += &format!("fn c{j}(u: {ty}, v: {ty}) -> {ty} {{ return {combine}; }}\n");
    }

    let subgroups = tuning.subgroups && elem.subgroups();
    let (shared, extra_args) = if subgroups {
        let n = ops.as_slice().len() as u32 * MAX_SUBGROUPS;
        (
            format!("var<workgroup> sh: array<{ty}, {n}>;"),
            "\n    @builtin(subgroup_invocation_id) lane: u32,\n    @builtin(subgroup_size) sg: u32,",
        )
    } else {
        (format!("var<workgroup> sh: array<{ty}, {wg}>;"), "")
    };
    let partial = |j: usize, value: &str| format!("out[{j}u * groups + group] = {value};");
    let body = reduce_phase(
        elem,
        ops,
        &Phase {
            n: "p.n",
            group: "group",
            groups: "groups",
            segmented,
            segment: "p.n",
            load,
            store: &partial,
            // A partial result is not the final value when the last workgroup finishes it.
            epilogue: if single_pass.is_some() { None } else { Some(ops) },
        },
        subgroups,
        tuning,
    );
    let gate_call = |value: &str| if gate.is_some() { format!("        luma_gate({value});\n") } else { String::new() };
    let (declare, last_block) = match single_pass {
        None => (String::new(), match gate {
            // One group: its only workgroup has the final value in `out`.
            Some(_) => format!("    if (lid == 0u) {{\n{}    }}\n", gate_call("out[0u]")),
            None => String::new(),
        }),
        Some(counter) => {
            let result = counter + 1;
            let value = |j: usize, value: &str| format!("luma_result[{j}u] = {value};");
            let rest = reduce_phase(
                elem,
                ops.rest(),
                &Phase { n: "groups", group: "0u", groups: "1u", segmented: true, segment: "groups", load: "luma_partial", store: &value, epilogue: Some(ops) },
                subgroups,
                tuning,
            );
            (
                format!(
                    "@group(1) @binding({counter}) var<storage, read_write> luma_counter: array<atomic<u32>>;
@group(1) @binding({result}) var<storage, read_write> luma_result: array<{ty}>;
var<workgroup> luma_ticket: u32;
fn luma_partial(i: u32) -> {ty} {{ return out[i]; }}
"
                ),
                format!(
                    "    storageBarrier();
    if (lid == 0u) {{ luma_ticket = atomicAdd(&luma_counter[0], 1u); }}
    if (workgroupUniformLoad(&luma_ticket) == groups - 1u) {{
        storageBarrier();
{rest}        if (lid == 0u) {{
            atomicStore(&luma_counter[0], 0u);
{}        }}
    }}
",
                    gate_call("luma_result[0u]")
                ),
            )
        }
    };
    let (gate_declare, gate_entry) = match (gate, ops.gate()) {
        (Some(first), Some(id)) => {
            let ([st, args, tmpl, hist, agg, stepv], source) = (std::array::from_fn(|k| first + k as u32), id.source());
            (
                format!(
                    "@group(1) @binding({st}) var<storage, read_write> st: array<u32>;
@group(1) @binding({args}) var<storage, read_write> args: array<u32>;
@group(1) @binding({tmpl}) var<storage, read> tmpl: array<u32>;
@group(1) @binding({hist}) var<storage, read_write> hist: array<f32>;
@group(1) @binding({agg}) var<storage, read_write> agg: array<f32>;
@group(1) @binding({stepv}) var<storage, read_write> stepv: array<f32>;
var<workgroup> luma_stop: u32;
{GATE_STOPPED}{source}"
                ),
                "    if (lid == 0u) { luma_stop = st[0]; }
    if (workgroupUniformLoad(&luma_stop) != 0u) {
        if (group == 0u && lid == 0u) { luma_stopped(); }
        return;
    }
",
            )
        }
        _ => (String::new(), ""),
    };
    format!(
        "{}{shared}
{declare}{gate_declare}
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
{gate_entry}    let zero = {ty}();
{body}{last_block}}}
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
            if i.is_multiple_of(4) {
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
        // Taps past the end of the chunk have zero weights; their inputs may be anything.
        fmas.push_str(&format!("            if (k + {kk}u < taps) {{\n"));
        for i in 0..r {
            fmas.push_str(&format!("                acc{i} = fma(v{}, w{kk}, acc{i});\n", i * s + kk));
        }
        fmas.push_str("            }\n");
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
    if (i < taps) {{ return kernel[c + i]; }}
    return 0.0;
}}

fn load(g: i32) -> f32 {{
    if (g >= 0 && u32(g) < params.input_len) {{ return input_data[u32(g)]; }}
    return 0.0;
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

/// How the Thomas solve of lines of blocks runs: the invocations per workgroup, the layout of
/// the scratch `cp` and whether two invocations solve a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ThomasPlan {
    pub(crate) workgroup: u8,
    /// Whether `cp` is tiled (see [`thomas_plan`]) rather than one cell after the other.
    pub(crate) tiled: bool,
    /// Whether two invocations solve a line, one sweeping from each end ([`tridiagonal_body`]),
    /// rather than one (see [`thomas_plan`]).
    pub(crate) twisted: bool,
}

/// Cells per tile of the tiled scratch: a warp.
const CP_TILE: u32 = 32;

/// The largest block size and the line count below which the twisted solve is chosen.
const TWISTED_BLOCK: u32 = 4;
const TWISTED_LINES: u32 = 8192;

/// The invocations per workgroup, the scratch layout and the line split of the Thomas solve of
/// `lines` lines of `block`-blocks, `stride` cells apart in memory.
///
/// With few lines the solve is latency bound: one invocation walks a long serial chain and the
/// GPU is nearly idle (RTX 3090, 708k cells: a strided 5x5 solve of 5568 lines of 128 cells had
/// 87 workgroups of 64 invocations, ~2 warps per multiprocessor, 771 us). A *twisted* solve runs
/// two invocations per line, one sweeping from each end ([`tridiagonal_body`]), so the chain is
/// halved and a workgroup of `workgroup` invocations covers `workgroup / 2` lines (`workgroup` is
/// even). It is chosen for blocks up to [`TWISTED_BLOCK`] and fewer than [`TWISTED_LINES`] lines,
/// where it measured faster on an RTX 3090 (stacked block-4, 1024 lines: 1516 -> 1100 us and
/// 1183 -> 709; 1024 lines of 100 cells along axis 2: 113 -> 71; 2048 block-2 lines of 32 cells:
/// 13.3 -> 10.4; 4096 scalar lines: 2.2 -> 2.0 ms; parts of 4488 lines of 70 cells, evaluated
/// (lazy): block 1 25.6 -> 22.5 us, block 2 76.8 -> 55.2 (104.5 -> 85.0), block 4 253.9 -> 186.3
/// (219 -> 182)). It lost with more lines or 5x5 blocks (parts, 13090 lines of 24 cells, block 4:
/// 150.5 -> 163.1 us (147.1 -> 161.6); 4488 lines of 70, block 5: 280.6 -> 300.9 (216.1 -> 232.4))
/// and was neutral on a fused 5x5 solve: there the plain solve, one invocation per line, stays.
///
/// Lines along the contiguous axis (or nearly: stride below [`Tuning::line_strided_from`]) have
/// their cells far apart for neighbouring invocations, so every access is a sector of its own
/// whatever the layout and the solve is latency bound. A small workgroup spreads the lines over
/// all multiprocessors (a workgroup runs on one) and lanes of a warp left idle cost nothing:
/// aiming for 512 workgroups, from 8 invocations to a warp of 32 (RTX 3090; 4096 lines,
/// `tri/solve_axis0`: 9.4 ms with 256, 3.9 with 32, 2.6 with 8; 1024 block-4 lines: 9.5, 2.1 ms
/// with 32, 8 after unrolling the kernel). Software rasterizers keep the elementwise size.
///
/// Along a strided axis neighbouring invocations solve neighbouring lines and read neighbouring
/// cells, so a warp's loads coalesce when the warp is full, and the solve is bound by memory
/// (5x5 blocks, 4488 lines of 70 cells, parts of 50 fields: 200 us without the elimination, 225
/// with it, 161 MB moved: 86 % of the RTX 3090's bandwidth) rather than by the dependent chain
/// of a cell. The workgroup is then a warp or more (blocks of 5, 4488 lines of 70, evaluated
/// parts, scratch in planes `k * cells + q`: 8 invocations 581 us, 16 380, 32 300, 64 295, 128
/// 290; 4096 lines of 100 cells, blocks of 1 to 5, 8 to 128: 16 to 32 for scalars, 128 for 2x2
/// blocks (106 us against 133 with 64), 64 for 3x3 to 5x5 (4x4: 206 against 228 with 32)), and
/// the scratch is tiled: the `cp` entries of the 32 cells `32 t ..` that a warp's cells fall in
/// are `block^2` runs of 32 floats one after the other, so each store and load of a cell's entry
/// is one contiguous 128 byte run and the 25 runs of a tile are one contiguous piece of memory,
/// where one cell after the other (`q * block^2 + k`) takes a sector per lane and entry (5x5
/// blocks, 4488 lines of 70, parts of 50 fields: 297 us one cell after the other, 234 with the
/// entries in planes `k * cells + q`, 220 tiled; 35904 lines: 2.1 ms, 2.3 ms, 2.0 ms). Below a
/// stride of 16 a warp's cells are in runs of too few neighbours (strides 2, 4, 8: tiled 1.5x,
/// 1.6x, 1.03x slower).
///
/// Stacked blocks (`parts` false) keep the narrow workgroups and the scratch of one cell after
/// the other: their coefficients are read `block^2` floats apart per cell, so a warp's loads do
/// not coalesce whatever the workgroup, and the wide tiled solve was 1.1x to 1.3x slower on
/// `bench_solve --only sweep/` (blocks of 2 to 5, 100 to 1024 strided lines of 32 to 1024 cells).
pub(crate) fn thomas_plan(tuning: &Tuning, block: u32, lines: u32, stride: u32, parts: bool) -> ThomasPlan {
    if tuning.profile == TuningProfile::Cpu {
        return ThomasPlan { workgroup: tuning.elementwise_workgroup as u8, tiled: false, twisted: false };
    }
    // Few lines of small blocks: latency bound (see above), two invocations per line win.
    let twisted = block <= TWISTED_BLOCK && lines < TWISTED_LINES;
    if parts && tuning.line_strided_from > 0 && stride >= tuning.line_strided_from && block <= UNROLLED_BLOCK {
        let wanted = match block {
            1 => 32,
            2 => 128,
            _ => 64,
        };
        return ThomasPlan { workgroup: wanted.min(tuning.elementwise_workgroup) as u8, tiled: block > 1, twisted };
    }
    ThomasPlan { workgroup: (lines / 512).max(1).next_power_of_two().clamp(8, 32).min(tuning.elementwise_workgroup) as u8, tiled: false, twisted }
}

/// The workgroups of the Thomas solve of `lines` lines with `workgroup` invocations per
/// workgroup: `workgroup / 2` lines each if `twisted` (see [`tridiagonal_body`]), else
/// `workgroup`.
pub(crate) fn thomas_workgroups(lines: u32, workgroup: u8, twisted: bool) -> u32 {
    let per_group = if twisted { u32::from(workgroup) / 2 } else { u32::from(workgroup) };
    lines.div_ceil(per_group)
}

/// Floats of the scratch `cp` of one system of `cells` cells of `block`-blocks, in the layout
/// [`thomas_plan`] chose: the tiled one rounds the cells up to whole tiles. `None` on overflow.
pub(crate) fn thomas_scratch_floats(cells: u64, block: u32, tiled: bool) -> Option<u64> {
    let cells = if tiled { cells.checked_next_multiple_of(u64::from(CP_TILE))? } else { cells };
    cells.checked_mul(u64::from(block * block))
}

/// The bindings of the Thomas kernel and the line addressing. `out` holds the forward-swept
/// right-hand sides and then the solution, `cp` the swept upper blocks. Cell `q` of the grid holds
/// block `lo/diag/up[q * B * B + row * B + col]` and `rhs[q * B + row]` (see [`Io::buffers`]); the
/// line `l` has `p.n` cells `line_base(l) + s * p.stride`.
fn tridiagonal_header(starts: bool, io: &Io) -> String {
    let (starts_binding, starts_decl) = if starts {
        ("@group(1) @binding(6) var<storage, read> starts: array<u32>;\n", 7)
    } else {
        ("", 6)
    };
    let (coef, rhs) = (io.coef_type(), io.rhs_type());
    format!(
        "struct Params {{ n: u32, stride: u32, lines: u32 }}
@group(1) @binding(0) var<storage, read_write> out: array<{rhs}>;
@group(1) @binding(1) var<storage, read_write> cp: array<f32>;
@group(1) @binding(2) var<storage, read> lo: array<{coef}>;
@group(1) @binding(3) var<storage, read> diag: array<{coef}>;
@group(1) @binding(4) var<storage, read> up: array<{coef}>;
@group(1) @binding(5) var<storage, read> rhs: array<{rhs}>;
{starts_binding}@group(1) @binding({starts_decl}) var<uniform> p: Params;
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
fn tridiagonal(block: u32, starts: bool, wg: u32, twisted: bool) -> String {
    if block <= UNROLLED_BLOCK {
        tridiagonal_unrolled(block as usize, starts, wg, twisted)
    } else {
        assert!(!twisted, "no twisted solve of blocks above {UNROLLED_BLOCK}");
        tridiagonal_loops(block, starts, wg)
    }
}

/// The largest block size the solve is unrolled for.
pub(crate) const UNROLLED_BLOCK: u32 = 5;

/// The straight-line solve of `b`-blocks (see [`tridiagonal_body`] for the algorithm). Per cell a
/// sweep holds the diagonal block `m`, the right-hand side `r = [coupling block | rhs]` (all
/// `b x (b + 1)` entries as scalars) and the previous cell's swept values `pc`/`pd` in registers;
/// it eliminates with partial pivoting (the pivot row is found with `select`s, rows are exchanged
/// with `select`s, nothing branches) and writes the swept values for the back substitution, which
/// carries the solution of the next cell in registers. For `b == 1` this is the plain Thomas
/// algorithm.
fn tridiagonal_unrolled(b: usize, starts: bool, wg: u32, twisted: bool) -> String {
    let mut src = tridiagonal_header(starts, &Io::buffers(b));
    src.push_str(&tridiagonal_body(b, starts, wg, 1, &Io::buffers(b), false, twisted));
    src
}

/// Gaussian elimination with partial pivoting (the first row of largest magnitude wins ties) of
/// the `b x b` matrix in the scalars `m{i}{j}` with the `w` right-hand side columns `r{i}{j}`,
/// then back substitution: the solution replaces `r`. Needs `ind` to be inside a block of its own
/// (declares `best*`, `pv*`, `v*`, ...).
fn eliminate(src: &mut String, ind: &str, b: usize, w: usize) {
    for col in 0..b {
        if col + 1 < b {
            let _ = writeln!(src, "{ind}var best{col} = abs(m{col}{col});\n{ind}var pv{col} = {col}u;");
            for i in col + 1..b {
                let _ = writeln!(
                    src,
                    "{ind}let a{col}{i} = abs(m{i}{col});\n{ind}let gt{col}{i} = a{col}{i} > best{col};\n{ind}best{col} = select(best{col}, a{col}{i}, gt{col}{i});\n{ind}pv{col} = select(pv{col}, {i}u, gt{col}{i});"
                );
            }
            for i in col + 1..b {
                let _ = writeln!(src, "{ind}let sw{col}{i} = pv{col} == {i}u;");
                let entries = (col..b).map(|j| format!("m{{}}{j}")).chain((0..w).map(|j| format!("r{{}}{j}")));
                for entry in entries {
                    let (top, other) = (entry.replace("{}", &col.to_string()), entry.replace("{}", &i.to_string()));
                    let _ = writeln!(
                        src,
                        "{ind}{{ let t = {top}; {top} = select({top}, {other}, sw{col}{i}); {other} = select({other}, t, sw{col}{i}); }}"
                    );
                }
            }
        }
        let _ = writeln!(src, "{ind}let v{col} = 1.0 / m{col}{col};");
        for i in col + 1..b {
            let _ = writeln!(src, "{ind}let f{col}{i} = m{i}{col} * v{col};");
            for j in col + 1..b {
                let _ = writeln!(src, "{ind}m{i}{j} -= f{col}{i} * m{col}{j};");
            }
            for j in 0..w {
                let _ = writeln!(src, "{ind}r{i}{j} -= f{col}{i} * r{col}{j};");
            }
        }
    }
    for j in 0..w {
        for i in (0..b).rev() {
            let mut x = format!("r{i}{j}");
            for k in i + 1..b {
                let _ = write!(x, " - m{i}{k} * r{k}{j}");
            }
            let _ = writeln!(src, "{ind}r{i}{j} = ({x}) * v{i};");
        }
    }
}

/// The entry point of the straight-line Thomas solve of `systems` systems of `b`-blocks, reading
/// and writing through `io`, after the bindings and `Params` (see [`tridiagonal_header`]; the
/// scratch `cp` holds `b * b` floats per cell and system: entry `k` of cell `q` at `q * b * b + k`,
/// or with `tiled` (see [`thomas_plan`]) at `(q / 32) * 32 * b * b + 32 * k + q % 32`, and the
/// system `sys` starts at `thomas_scratch_floats(lines * n, b, tiled)`).
///
/// With `twisted` two invocations solve a line (twisted factorisation), halving the serial chain
/// (see [`thomas_plan`] for when it pays): of the rows `s0..n` of line `l`,
/// invocation A sweeps the top half `s0..=mid` downwards (`mid = s0 + (n - s0 - 1) / 2`) exactly as
/// the Thomas algorithm does, `x_r = f'_r - C'_r x_{r+1}`, and invocation B the bottom half
/// `mid+1..n` upwards, `x_r = g_r - E_r x_{r-1}` (`M = D_r - U_r E_{r+1}`, `E_r = M^-1 L_r`,
/// `g_r = M^-1 (f_r - U_r g_{r+1})`; `E`/`g` take the places of `C'`/`f'` in the scratch and
/// `out`). The two swept values at the cut go through workgroup memory and both invocations
/// solve `(I - C'_mid E_{mid+1}) x_mid = f'_mid - C'_mid g_{mid+1}`, then each back substitutes
/// its half. A one row line is solved by A alone. A workgroup of `wg` invocations covers `wg / 2`
/// lines: invocation `t` is side `t / (wg / 2)` of line `group * wg / 2 + t % (wg / 2)`, so a warp
/// still touches neighbouring lines (coalescing and the tiled scratch work as for one
/// invocation per line) and every cell's scratch is written and read by one invocation. All
/// invocations reach the barrier (`busy` instead of an early return).
///
/// Without `twisted` one invocation sweeps the whole line (`mid = n - 1`, no exchange, no
/// barrier), `wg` lines per workgroup.
pub(crate) fn tridiagonal_body(b: usize, starts: bool, wg: u32, systems: u32, io: &Io, tiled: bool, twisted: bool) -> String {
    assert!(!twisted || (wg >= 2 && wg.is_multiple_of(2)), "twisted Thomas workgroup of {wg} invocations: two per line need an even size");
    let w = b + 1;
    let bb = b * b;
    // Invocations per line, lines per workgroup, floats a line exchanges.
    let (hb, st) = (if twisted { wg / 2 } else { wg }, bb + b);
    let s0 = if starts { "min(starts[l], p.n)" } else { "0u" };
    let (guard, sys, line, scratch) = if systems > 1 {
        (format!("p.lines * {systems}u"), "    let sys = lg / p.lines;\n", "lg % p.lines", if tiled { format!("sys * (((p.lines * p.n + 31u) & ~31u) * {bb}u)") } else { format!("sys * (p.lines * p.n * {bb}u)") })
    } else {
        ("p.lines".to_owned(), "    let sys = 0u;\n", "lg", "0u".to_owned())
    };
    let mut src = String::from("fn line_base(l: u32) -> u32 { return (l % p.stride) + (l / p.stride) * p.stride * p.n; }\n");
    // The index of entry `k` of the scratch of cell `q`, from `cq` (see above).
    let cp_at = |k: usize| if tiled { format!("cq + {}u", 32 * k) } else { format!("cq + {k}u") };
    let cq_expr = if tiled { format!("so + (q >> 5u) * {}u + (q & 31u)", 32 * bb) } else { format!("so + q * {bb}u") };
    let (exchange, side, step) = if twisted { (format!("var<workgroup> xch: array<f32, {}>;\n", wg * st as u32), format!("tid / {hb}u"), 2) } else { (String::new(), "0u".to_owned(), 1) };
    let _ = write!(
        src,
        "{exchange}@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
    let gi = gid.x + gid.y * nwg.x * {wg}u;
    let tid = gi % {wg}u;
    let side = {side};
    let lt = tid % {hb}u;
    let lg = (gi / {wg}u) * {hb}u + lt;
    let live = lg < {guard};
{sys}    let l = {line};
    let so = {scratch};
    let base = line_base(l);
    var s0 = p.n;
    if (live) {{
        s0 = {s0};
        for (var s = side; s < s0; s += {step}u) {{
            let q = base + s * p.stride;
"
    );
    let zeros: Vec<String> = vec!["0.0".to_owned(); b];
    src.push_str(&io.store("            ", "q", &zeros));
    src.push_str("        }\n    }\n    let busy = live && s0 < p.n;\n");
    if twisted {
        src.push_str(
            "    let mid = s0 + (max(p.n - s0, 1u) - 1u) / 2u;
    let hasb = mid + 1u < p.n;
    var nrows = 0u;
    if (busy) { nrows = select(mid + 1u - s0, p.n - 1u - mid, side == 1u); }
",
        );
    } else {
        src.push_str("    let mid = p.n - 1u;\n    var nrows = 0u;\n    if (busy) { nrows = p.n - s0; }\n");
    }
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "    var pc{i}{j} = 0.0;");
        }
        let _ = writeln!(src, "    var pd{i} = 0.0;");
    }
    // The sweep of this invocation's half: A from `s0` down, B from `n - 1` up. The block that
    // the row's right-hand block is taken from (A: `up`, B: `lo`) is read always, the one that
    // couples to the previous row of the sweep (A: `lo`, B: `up`) from the second row on.
    let _ = write!(src, "    for (var k = 0u; k < nrows; k++) {{\n        let s = select(s0 + k, p.n - 1u - k, side == 1u);\n        let q = base + s * p.stride;\n        let cq = {cq_expr};\n");
    src.push_str(&io.load_core("        ", "q"));
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        var m{i}{j} = {};\n        var r{i}{j} = 0.0;\n        var l{i}{j} = 0.0;", io.diag("q", i * b + j));
        }
        let _ = writeln!(src, "        var r{i}{b} = {};", io.rhs("q", i));
    }
    for (id, own, other) in [(0, "up", "lo"), (1, "lo", "up")].into_iter().take(if twisted { 2 } else { 1 }) {
        let load = |name: &str, ind: &str| if name == "up" { io.load_up(ind, "q") } else { io.load_lo(ind, "q") };
        let entry = |name: &str, k: usize| if name == "up" { io.up("q", k) } else { io.lo("q", k) };
        let head = if !twisted { "        {" } else if id == 0 { "        if" } else { " else if" };
        if twisted {
            let _ = writeln!(src, "{head} (side == {id}u) {{");
        } else {
            let _ = writeln!(src, "{head}");
        }
        src.push_str(&load(own, "            "));
        for i in 0..b {
            for j in 0..b {
                let _ = writeln!(src, "            r{i}{j} = {};", entry(own, i * b + j));
            }
        }
        src.push_str("            if (k > 0u) {\n");
        src.push_str(&load(other, "                "));
        for i in 0..b {
            for j in 0..b {
                let _ = writeln!(src, "                l{i}{j} = {};", entry(other, i * b + j));
            }
        }
        src.push_str("            }\n        }");
    }
    src.push('\n');
    src.push_str("        if (k > 0u) {\n");
    for i in 0..b {
        for j in 0..b {
            let sum = (0..b).map(|k| format!("l{i}{k} * pc{k}{j}")).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "            m{i}{j} -= {sum};");
        }
        let sum = (0..b).map(|k| format!("l{i}{k} * pd{k}")).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "            r{i}{b} -= {sum};");
    }
    src.push_str("        }\n");
    eliminate(&mut src, "        ", b, w);
    for i in 0..b {
        for j in 0..b {
            let _ = writeln!(src, "        cp[{}] = r{i}{j};\n        pc{i}{j} = r{i}{j};", cp_at(i * b + j));
        }
        let _ = writeln!(src, "        pd{i} = r{i}{b};");
    }
    let swept: Vec<String> = (0..b).map(|i| format!("r{i}{b}")).collect();
    src.push_str(&io.store("        ", "q", &swept));
    src.push_str("    }\n");
    // The solution of a one row line (or of the whole line's last row) is its swept right-hand
    // side, still in `pd`.
    let write_x = |src: &mut String| {
        for i in 0..b {
            let _ = writeln!(src, "    var x{i} = pd{i};");
        }
    };
    if twisted {
        // The swept values at the cut: C' and f' of A's last row, E and g of B's first row.
        src.push_str("    if (busy) {\n");
        for i in 0..b {
            for j in 0..b {
                let _ = writeln!(src, "        xch[tid * {st}u + {}u] = pc{i}{j};", i * b + j);
            }
            let _ = writeln!(src, "        xch[tid * {st}u + {}u] = pd{i};", bb + i);
        }
        src.push_str("    }\n    workgroupBarrier();\n");
        write_x(&mut src);
        src.push_str("    var back = 0u;\n    if (busy && hasb) {\n");
        let _ = writeln!(src, "        let ca = lt * {st}u;\n        let cb = {}u + ca;", hb as usize * st);
        for i in 0..b {
            for j in 0..b {
                let sum = (0..b).map(|k| format!("xch[ca + {}u] * xch[cb + {}u]", i * b + k, k * b + j)).collect::<Vec<_>>().join(" + ");
                let _ = writeln!(src, "        var m{i}{j} = {}.0 - ({sum});", u8::from(i == j));
            }
            let sum = (0..b).map(|k| format!("xch[ca + {}u] * xch[cb + {}u]", i * b + k, bb + k)).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "        var r{i}0 = xch[ca + {}u] - ({sum});", bb + i);
        }
        eliminate(&mut src, "        ", b, 1);
        // `r` is x_mid; x_{mid+1} = g - E x_mid. Each side keeps its own row.
        for i in 0..b {
            let sum = (0..b).map(|k| format!("xch[cb + {}u] * r{k}0", i * b + k)).collect::<Vec<_>>().join(" + ");
            let _ = writeln!(src, "        x{i} = select(r{i}0, xch[cb + {}u] - ({sum}), side == 1u);", bb + i);
        }
        src.push_str("        let q = base + (mid + side) * p.stride;\n");
        let solved: Vec<String> = (0..b).map(|i| format!("x{i}")).collect();
        src.push_str(&io.store("        ", "q", &solved));
        src.push_str("        back = select(mid - s0, p.n - mid - 2u, side == 1u);\n    }\n");
    } else {
        write_x(&mut src);
        src.push_str("    var back = 0u;\n    if (busy) { back = mid - s0; }\n");
    }
    // Back substitution of the rest of the half: A from `mid - 1` down, B from `mid + 2` up.
    let _ = write!(
        src,
        "    for (var k = 0u; k < back; k++) {{\n        let s = select(mid - 1u - k, mid + 2u + k, side == 1u);\n        let q = base + s * p.stride;\n        let cq = {cq_expr};\n"
    );
    src.push_str(&io.load_out("        ", "q", "yo"));
    for i in 0..b {
        let sum = (0..b).map(|k| format!("cp[{}] * x{k}", cp_at(i * b + k))).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(src, "        let y{i} = yo{i} - ({sum});");
    }
    let solved: Vec<String> = (0..b).map(|i| format!("y{i}")).collect();
    src.push_str(&io.store("        ", "q", &solved));
    for i in 0..b {
        let _ = writeln!(src, "        x{i} = y{i};");
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

/// The kernel a [`Builtin`] runs: built directly as a Kernel IR module, or as WGSL text.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BuiltinKernel {
    Module(KernelModule),
    Wgsl(Cow<'static, str>),
}

/// `gid.x + gid.y * nwg.x * wg`: the element of the invocation of a launch folded over x and y.
fn element_index(f: &mut FnBuilder<'_>, wg: u32) -> Val {
    let gid = f.builtin(BuiltinVar::GlobalInvocationId);
    let nwg = f.builtin(BuiltinVar::NumWorkgroups);
    let (gx, gy, nx) = (f.field(gid, 0, KirType::U32), f.field(gid, 1, KirType::U32), f.field(nwg, 0, KirType::U32));
    let width = f.lit_u32(wg);
    let index = gx + gy * nx * width;
    f.let_val("i", index)
}

/// `if (i >= luma.size.x) { return; }`
fn guard_size(f: &mut FnBuilder<'_>, i: Val) {
    let size = f.grid_size(0);
    let outside = f.ge(i, size);
    f.if_then(outside, |f| f.ret(None));
}

/// The scalar or vector type named by the WGSL `ty` of an element.
fn element_type(ty: &str) -> KirType {
    match ty {
        "f32" => KirType::F32,
        "u32" => KirType::U32,
        "i32" => KirType::I32,
        "f64" => KirType::Scalar(ScalarTy::F64),
        "vec2<f32>" => KirType::vec(ScalarTy::F32, 2),
        other => unreachable!("no fill for elements of type {other}"),
    }
}

/// [`Builtin::Fill`]: `out[i] = p.s`.
fn fill_module(ty: &str, wg: u32) -> KernelModule {
    let ty = element_type(ty);
    let mut mb = ModuleBuilder::new("fill", "main", [wg, 1, 1]);
    let params = mb.add_struct_layout("Params", &[("s", ty.clone())]);
    let out = mb.add_storage_read_write("out", ty.clone());
    let p = mb.add_uniform("p", KirType::Struct(params));
    let mut f = mb.entry_builder();
    let i = element_index(&mut f, wg);
    guard_size(&mut f, i);
    let value = f.uniform_field(p, 0, ty);
    f.store(out, i, value);
    f.finish();
    finish(mb, None)
}

/// [`Builtin::FillBits`]: the words of a bool array filled with `p.s`, the bits past `p.len` zero.
fn fill_bits_module(wg: u32) -> KernelModule {
    let mut mb = ModuleBuilder::new("fill_bits", "main", [wg, 1, 1]);
    let params = mb.add_uniform_u32_struct("Params", &["s", "len"]);
    let out = mb.add_storage_read_write("out", KirType::U32);
    let p = mb.add_uniform("p", KirType::Struct(params));
    let mut f = mb.entry_builder();
    let i = element_index(&mut f, wg);
    guard_size(&mut f, i);
    let (s, len) = (f.uniform_field(p, 0, KirType::U32), f.uniform_field(p, 1, KirType::U32));
    let (zero, one, mask) = (f.lit_u32(0), f.lit_u32(1), f.lit_u32(31));
    let low = f.let_val("low", len & mask);
    let last = f.grid_size(0) - one;
    let partial = f.and(f.eq(i, last), f.ne(low, zero));
    let bits = f.shl(one, low) - one;
    let full = f.lit_u32(0xFFFF_FFFF);
    let tail = f.select(partial, bits, full);
    let word = f.select(f.ne(s, zero), tail, zero);
    f.store(out, i, word);
    f.finish();
    finish(mb, None)
}

/// [`Builtin::NonFinite`]: `out[i] = 1` where the bits of `a[i]` are those of an `Inf` or a `NaN`.
fn non_finite_module(wg: u32) -> KernelModule {
    let mut mb = ModuleBuilder::new("non_finite", "main", [wg, 1, 1]);
    let out = mb.add_storage_read_write("out", KirType::U32);
    let a = mb.add_storage_read("a", KirType::U32);
    let mut f = mb.entry_builder();
    let i = element_index(&mut f, wg);
    guard_size(&mut f, i);
    let exponent = f.lit_u32(0x7f80_0000);
    let special = f.eq(f.load(a, i) & exponent, exponent);
    let (zero, one) = (f.lit_u32(0), f.lit_u32(1));
    let flag = f.select(special, one, zero);
    f.store(out, i, flag);
    f.finish();
    finish(mb, None)
}

/// [`Builtin::Conv1d`]: spatial 1D convolution (cross-correlation). Launched with one invocation
/// per output element, folded over x and y workgroups. Buffers may be larger than the arrays they
/// hold, so lengths come from the parameters, never from the length of a binding.
fn conv1d_module() -> KernelModule {
    const WG: u32 = 64;
    let mut mb = ModuleBuilder::new("conv1d", "main", [WG, 1, 1]);
    let params = mb.add_uniform_u32_struct("Params", &["kernel_len", "stride", "padding", "input_len", "output_len"]);
    let output = mb.add_storage_read_write("output", KirType::F32);
    let input = mb.add_storage_read("input_data", KirType::F32);
    let kernel = mb.add_storage_read("kernel", KirType::F32);
    let p = mb.add_uniform("params", KirType::Struct(params));
    let mut f = mb.entry_builder();
    let i = element_index(&mut f, WG);
    let field = |f: &FnBuilder<'_>, at: u32| f.uniform_field(p, at, KirType::U32);
    let (kernel_len, stride, padding, input_len, output_len) =
        (field(&f, 0), field(&f, 1), field(&f, 2), field(&f, 3), field(&f, 4));
    let outside = f.ge(i, output_len);
    f.if_then(outside, |f| f.ret(None));
    let first = f.convert(i * stride, KirType::I32) - f.convert(padding, KirType::I32);
    let in_start = f.let_val("in_start", first);
    let zero = f.lit_f32(0.0);
    let sum = f.var_decl("sum", KirType::F32, Some(zero));
    let (start, step) = (f.lit_u32(0), f.lit_u32(1));
    f.for_range("k", start, kernel_len, step, |f, k| {
        let at = f.let_val("in_idx", in_start + f.convert(k, KirType::I32));
        let inside = f.and(f.ge(at, f.lit_i32(0)), f.lt(f.convert(at, KirType::U32), input_len));
        f.if_then(inside, |f| {
            let term = f.load(input, f.convert(at, KirType::U32)) * f.load(kernel, k);
            let next = f.read_var(sum) + term;
            f.assign_var(sum, next);
        });
    });
    let total = f.read_var(sum);
    f.store(output, i, total);
    f.finish();
    finish(mb, None)
}

/// Writes the indirect arguments of the next step of `iterate_until`: the template, or zeros
/// once the residual is within the tolerance (a NaN residual never is) so that the steps after
/// convergence dispatch no workgroups.
fn until_gate_module() -> KernelModule {
    let mut mb = ModuleBuilder::new("until_gate", "main", [64, 1, 1]);
    let params = mb.add_struct_layout("Params", &[("tol", KirType::F32), ("step", KirType::U32), ("words", KirType::U32)]);
    let res = mb.add_storage_read("res", KirType::F32);
    let st = mb.add_storage_read_write("st", KirType::U32);
    let args = mb.add_storage_read_write("args", KirType::U32);
    let tmpl = mb.add_storage_read("tmpl", KirType::U32);
    let p = mb.add_uniform("p", KirType::Struct(params));
    let mut f = mb.entry_builder();
    let gid = f.builtin(BuiltinVar::GlobalInvocationId);
    let g = f.field(gid, 0, KirType::U32);
    let g = f.let_val("g", g);
    let (zero, one) = (f.lit_u32(0), f.lit_u32(1));
    let residual = f.load(res, zero);
    let residual = f.let_val("residual", residual);
    // Fast math may not honour NaN comparisons, so the bits are tested too.
    let bits = f.bitcast(residual, KirType::U32);
    let finite = f.le(bits & f.lit_u32(0x7fff_ffff), f.lit_u32(0x7f80_0000));
    let tol = f.uniform_field(p, 0, KirType::F32);
    let reached = f.and(finite, f.le(residual, tol));
    let hit = f.let_val("hit", reached);
    let stopped = f.ne(f.load(st, zero), zero);
    let done = f.let_val("done", f.or(stopped, hit));
    let first = f.and(f.eq(g, zero), f.eq(f.load(st, zero), zero));
    f.if_then(first, |f| {
        f.if_then(hit, |f| f.store(st, zero, one));
        let step = f.uniform_field(p, 1, KirType::U32);
        f.store(st, one, step + one);
    });
    let words = f.uniform_field(p, 2, KirType::U32);
    let within = f.lt(g, words);
    f.if_then(within, |f| {
        let value = f.select(done, zero, f.load(tmpl, g));
        f.store(args, g, value);
    });
    f.finish();
    finish(mb, None)
}

/// [`Builtin::MatmulSum`]: sums the partial products of a split product. A workgroup of 32x8
/// invocations covers 32 outputs, row `ty` of it summing every eighth slice from `ty`.
fn matmul_sum_module() -> KernelModule {
    let mut mb = ModuleBuilder::new("matmul_sum", "main", [32, 8, 1]);
    let params = mb.add_uniform_u32_struct("Params", &["s"]);
    let out = mb.add_storage_read_write("out", KirType::F32);
    let part = mb.add_storage_read("part", KirType::F32);
    let p = mb.add_uniform("p", KirType::Struct(params));
    let tile = mb.add_shared("tile", KirType::F32, 256);
    let mut f = mb.entry_builder();
    let lid = f.builtin(BuiltinVar::LocalInvocationId);
    let wid = f.builtin(BuiltinVar::WorkgroupId);
    let nwg = f.builtin(BuiltinVar::NumWorkgroups);
    let (tx, ty) = (f.field(lid, 0, KirType::U32), f.field(lid, 1, KirType::U32));
    let group = f.field(wid, 0, KirType::U32) + f.field(wid, 1, KirType::U32) * f.field(nwg, 0, KirType::U32);
    let i = f.let_val("i", group * f.lit_u32(32) + tx);
    let mn = f.grid_size(0);
    let slices = f.uniform_field(p, 0, KirType::U32);
    let zero = f.lit_f32(0.0);
    let acc = f.var_decl("acc", KirType::F32, Some(zero));
    let inside = f.lt(i, mn);
    f.if_then(inside, |f| {
        let step = f.lit_u32(8);
        f.for_range("z", ty, slices, step, |f, z| {
            let term = f.load(part, z * f.grid_size(0) + i);
            let next = f.read_var(acc) + term;
            f.assign_var(acc, next);
        });
    });
    let slot = ty * f.lit_u32(32) + tx;
    let partial = f.read_var(acc);
    f.shared_store(tile, slot, partial);
    f.barrier(BarrierScope::Workgroup);
    let first = f.and(f.eq(ty, f.lit_u32(0)), inside);
    f.if_then(first, |f| {
        let at = |f: &FnBuilder<'_>, row: u32| f.shared_load(tile, tx + f.lit_u32(row * 32));
        let (a, b) = (at(f, 0) + at(f, 1), at(f, 2) + at(f, 3));
        let (c, d) = (at(f, 4) + at(f, 5), at(f, 6) + at(f, 7));
        let total = (a + b) + (c + d);
        f.store(out, i, total);
    });
    f.finish();
    finish(mb, None)
}

impl Builtin {
    /// This kernel for elements of type `ty` (`"f32"` or `"u32"`), with the parameters `tuning`
    /// gives for the GPU.
    pub(crate) fn kernel(self, ty: &str, tuning: &Tuning) -> BuiltinKernel {
        let wg = tuning.elementwise_workgroup;
        let wgsl = |source: String| BuiltinKernel::Wgsl(source.into());
        match self {
            Builtin::Fill => BuiltinKernel::Module(fill_module(ty, wg)),
            Builtin::FillBits => BuiltinKernel::Module(fill_bits_module(wg)),
            Builtin::ReduceBits(ops) => wgsl(reduce_bits(ops, tuning)),
            Builtin::Reduce(ops, segmented) => wgsl(reduce(ty, ops, segmented, tuning)),
            Builtin::Conv1d => BuiltinKernel::Module(conv1d_module()),
            Builtin::Conv1dTiled(stride, per_thread) => wgsl(conv1d_tiled(stride, per_thread, tuning.conv_workgroup)),
            Builtin::Fft(pass) => wgsl(crate::fft::source(pass)),
            Builtin::UntilGate => BuiltinKernel::Module(until_gate_module()),
            Builtin::Tridiagonal { block, starts, workgroup, twisted } => wgsl(tridiagonal(block as u32, starts, u32::from(workgroup), twisted)),
            Builtin::CyclicTridiagonal { block, starts, workgroup, shift, shared_rows } => {
                wgsl(crate::cyclic::source(u32::from(block), starts, u32::from(workgroup), u32::from(shift), shared_rows))
            }
            Builtin::SolveParts(spec) => wgsl(spec.source()),
            Builtin::Radix { pass, first } => wgsl(crate::sort::source(pass, first)),
            Builtin::Matmul { tile, aligned, split } => wgsl(crate::matmul::builtin_source(tile, aligned, split)),
            Builtin::MatmulSum => BuiltinKernel::Module(matmul_sum_module()),
            Builtin::FilterCount => wgsl(crate::filter::count_source()),
            Builtin::FilterScatter { words, index } => wgsl(crate::filter::scatter_source(words, index)),
            Builtin::Transpose => wgsl(crate::jit::transpose::source(ty)),
            Builtin::NonFinite => BuiltinKernel::Module(non_finite_module(wg)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitwise_operations_only_exist_for_u32() {
        let bitwise = [BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor, BinOp::Shl, BinOp::Shr, BinOp::Rotl];
        for op in bitwise {
            assert!(op.expr_for(ElemTy::U32).is_some(), "{op:?}");
            for ty in [ElemTy::F32, ElemTy::F64, ElemTy::Df32] {
                assert!(op.expr_for(ty).is_none(), "{op:?} on {ty:?}");
            }
        }
        for op in [UnOp::BitNot, UnOp::Shl(3), UnOp::Shr(31), UnOp::Rotl(0)] {
            assert!(op.expr_for(ElemTy::U32).is_some());
            for ty in [ElemTy::F32, ElemTy::F64, ElemTy::Df32] {
                assert!(op.expr_for(ty).is_none(), "{op:?} on {ty:?}");
            }
        }
        assert_eq!(UnOp::Rotl(5).expr(), "((x << 5u) | (x >> 27u))");
        assert_eq!(UnOp::Rotl(0).expr(), "x");
        assert_eq!(UnOp::by_constant(BinOp::Shr, 33), Some(UnOp::Shr(1)));
        // There is no negation of an unsigned integer.
        assert!(UnOp::Neg.expr_for(ElemTy::U32).is_none());
        // `i32` has the integer operations only: no float math, no rotations.
        for op in [UnOp::Sqrt, UnOp::Exp, UnOp::Floor, UnOp::Rotl(3)] {
            assert!(op.expr_for(ElemTy::I32).is_none(), "{op:?}");
        }
        assert!(BinOp::Rotl.expr_for(ElemTy::I32).is_none());
        assert!(BinOp::Atan2.expr_for(ElemTy::I32).is_none());
        assert!(UnOp::Neg.expr_for(ElemTy::I32).is_some());
    }

    /// Every reduction kernel (first, later and single-pass passes) is valid WGSL, with subgroup
    /// operations and without.
    #[test]
    fn reduction_kernels_validate() {
        let sqrt = Epilogue::default().then(EpilogueStep::Unary(UnOp::Sqrt)).unwrap();
        let lists = [
            ReduceList::new(&[ReduceOp::Sum]),
            ReduceList::new(&[ReduceOp::Min]),
            ReduceList::new(&[ReduceOp::Max]),
            ReduceList::new(&[ReduceOp::SumSquares]).with_epilogue(sqrt),
            ReduceList::new(&[ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max, ReduceOp::SumSquares]),
        ];
        for subgroups in [false, true] {
            let tuning = Tuning { subgroups, ..Tuning::for_tests() };
            let features = if subgroups { luma_core::Features::SUBGROUP } else { luma_core::Features::empty() };
            for ty in ["f32", "u32", "i32", "vec4<f32>", "df32"] {
                for list in lists {
                    let list = if ty == "f32" { list } else { list.with_epilogue(Epilogue::default()) };
                    let kernels = [
                        ("first", reduce(ty, list, false, &tuning)),
                        ("single-pass", reduce(ty, list.single_pass(), false, &tuning)),
                        ("later", reduce(ty, list.rest(), true, &tuning)),
                    ];
                    for (what, wgsl) in kernels {
                        luma_core::kir::parse_wgsl(&wgsl, "main", None, features, [256; 3], 256, 64, true)
                            .unwrap_or_else(|error| panic!("{what} {ty} {list:?} (subgroups {subgroups}): {error}\n{wgsl}"));
                    }
                }
            }
        }
    }
}
