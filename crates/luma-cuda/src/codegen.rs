//! KIR -> CUDA C++ for NVRTC.
//!
//! The emitted source is header-free: no `#include`, every vector type, operator and helper it
//! uses is defined inline, so NVRTC needs nothing on disk. Pure Rust and OS independent.
//!
//! Kernel ABI of `luma_main` (the order `launch_kernel` must pass the parameters in):
//! `LumaGrid luma` by value, `const luma_u32x3* luma_indirect` (null for direct dispatches), one
//! parameter per *active* binding in ascending original binding order (see [`CudaArgAbi`]), then
//! one `uint32_t _luma_len{b}` per entry of [`CudaTranslated::array_lengths`].

use std::collections::{BTreeSet, HashSet};

use luma_core::BindingKind;
use luma_core::kir::*;

/// How one active binding is passed to `luma_main`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaArgAbi {
    /// A device pointer to the storage buffer.
    DevicePtr,
    /// The uniform bytes passed by value as a kernel parameter of `size` bytes.
    InlineUniform { size: usize },
    /// A device pointer to the uniform bytes (too large to pass by value).
    UniformPtr,
}

#[derive(Clone, Debug)]
pub struct CudaTranslated {
    pub source: String,
    /// One entry per active binding, in ascending original binding order.
    pub param_abis: Vec<CudaArgAbi>,
    /// Bindings whose element count is appended, after the binding arguments, as
    /// `uint32_t _luma_len{binding}` parameters (ascending). The count is the bound byte range
    /// divided by the binding's element stride.
    pub array_lengths: Vec<u32>,
}

const LANES: [&str; 4] = ["x", "y", "z", "w"];
const PARAM_BYTES: usize = 4096;
/// Largest uniform passed by value, whatever else is left of the parameter area.
const MAX_INLINE_UNIFORM: usize = 3584;

fn sc(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::Bool => "bool",
        ScalarTy::U32 => "uint32_t",
        ScalarTy::I32 => "int32_t",
        ScalarTy::F16 => "luma_h",
        ScalarTy::F32 => "float",
        ScalarTy::F64 => "double",
    }
}

fn sidx(s: ScalarTy) -> u8 {
    match s {
        ScalarTy::Bool => 0,
        ScalarTy::U32 => 1,
        ScalarTy::I32 => 2,
        ScalarTy::F16 => 3,
        ScalarTy::F32 => 4,
        ScalarTy::F64 => 5,
    }
}

fn sfrom(i: u8) -> ScalarTy {
    match i {
        0 => ScalarTy::Bool,
        1 => ScalarTy::U32,
        2 => ScalarTy::I32,
        3 => ScalarTy::F16,
        4 => ScalarTy::F32,
        _ => ScalarTy::F64,
    }
}

fn vname(s: ScalarTy, lanes: u8) -> String {
    format!("luma_{}x{}", s.wgsl_name(), lanes)
}

/// Size in bytes of the C++ type that stands for `ty`.
fn cpp_size(ty: &KirType, structs: &[StructDecl]) -> u32 {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => s.size_bytes(),
        KirType::Vec { elem, lanes } => elem.size_bytes() * u32::from(*lanes),
        KirType::Array { len, stride, .. } => len * stride,
        KirType::Struct(i) => structs[*i as usize].size,
    }
}

/// Alignment in bytes of the C++ type that stands for `ty` (matches the emitted declarations).
fn cpp_align(ty: &KirType, structs: &[StructDecl]) -> u32 {
    match ty {
        KirType::Scalar(s) | KirType::Atomic(s) => s.size_bytes(),
        KirType::Vec { elem, lanes } => match lanes {
            2 => 2 * elem.size_bytes(),
            4 => 4 * elem.size_bytes(),
            _ => elem.size_bytes(),
        },
        KirType::Array { elem, .. } => cpp_align(elem, structs),
        KirType::Struct(i) => structs[*i as usize].align.max(1),
    }
}

fn lit_f32(v: f32) -> String {
    if v.is_nan() {
        "__int_as_float(0x7fc00000)".into()
    } else if v.is_infinite() {
        if v > 0.0 { "__int_as_float(0x7f800000)".into() } else { "__int_as_float(0xff800000)".into() }
    } else if v < 0.0 || (v == 0.0 && v.is_sign_negative()) {
        format!("({v:?}f)")
    } else {
        format!("{v:?}f")
    }
}

fn lit_f64(v: f64) -> String {
    if v.is_nan() {
        "__longlong_as_double(0x7ff8000000000000LL)".into()
    } else if v.is_infinite() {
        if v > 0.0 { "__longlong_as_double(0x7ff0000000000000LL)".into() } else { "__longlong_as_double(0xfff0000000000000LL)".into() }
    } else if v < 0.0 || (v == 0.0 && v.is_sign_negative()) {
        format!("({v:?})")
    } else {
        format!("{v:?}")
    }
}

fn un_name(op: IrUnOp) -> &'static str {
    use IrUnOp::*;
    match op {
        Neg => "neg",
        Not => "not",
        BitNot => "bnot",
        Abs => "abs",
        Sqrt => "sqrt",
        Exp => "exp",
        Ln => "ln",
        Sin => "sin",
        Cos => "cos",
        Tan => "tan",
        Tanh => "tanh",
        Sign => "sign",
        Floor => "floor",
        Ceil => "ceil",
        Trunc => "trunc",
        Round => "round",
        CountOneBits => "popc",
        FirstTrailingBit => "ftb",
        IsNan => "isnan",
        IsInf => "isinf",
        OpaqueGuard => "opaque",
        InverseSqrt => "rsqrt",
        Exp2 => "exp2",
        Log2 => "log2",
        Asin => "asin",
        Acos => "acos",
        Atan => "atan",
        FirstLeadingBit => "flb",
        CountLeadingZeros => "clz",
        CountTrailingZeros => "ctz",
        All => "all",
        Any => "any",
    }
}

fn bin_name(op: IrBinOp) -> &'static str {
    use IrBinOp::*;
    match op {
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        Div => "div",
        Rem => "rem",
        AddExact => "addx",
        SubExact => "subx",
        MulExact => "mulx",
        Min => "min",
        Max => "max",
        Pow => "pow",
        Atan2 => "atan2",
        Copysign => "copysign",
        BitAnd => "band",
        BitOr => "bor",
        BitXor => "bxor",
        Shl => "shl",
        Shr => "shr",
        Rotl => "rotl",
        MulHiU32 => "mulhi",
        Eq => "eq",
        Ne => "ne",
        Lt => "lt",
        Le => "le",
        Gt => "gt",
        Ge => "ge",
        And => "and",
        Or => "or",
    }
}

fn ret(e: impl AsRef<str>) -> String {
    format!("return {};", e.as_ref())
}

/// Body of `luma_<op>(a)` for scalar `s`, with its result type.
fn scalar_un(op: IrUnOp, s: ScalarTy) -> Option<(ScalarTy, String)> {
    use IrUnOp::*;
    use ScalarTy::*;
    if s == F16 {
        let (r, _) = scalar_un(op, F32)?;
        let name = format!("luma_{}", un_name(op));
        return Some(if r == F32 {
            (F16, ret(format!("luma_to_h({name}((float)a))")))
        } else {
            (r, ret(format!("{name}((float)a)")))
        });
    }
    let f = |base: &str| -> String { if s == F32 { format!("{base}f") } else { base.to_string() } };
    let (zero, one) = if s == F64 { ("0.0", "1.0") } else { ("0.0f", "1.0f") };
    let r = |t: ScalarTy, e: &str| Some((t, ret(e)));
    match (op, s) {
        (Not, Bool) => r(Bool, "!a"),
        (Neg, F32 | F64) => r(s, "-a"),
        (Neg, I32) => r(I32, "(int32_t)(0u - (uint32_t)a)"),
        (Neg, U32) => r(U32, "0u - a"),
        (BitNot, U32 | I32) => r(s, "~a"),
        (Abs, F32 | F64) => r(s, &format!("{}(a)", f("fabs"))),
        (Abs, I32) => r(I32, "a < 0 ? (int32_t)(0u - (uint32_t)a) : a"),
        (Abs, U32) => r(U32, "a"),
        (Sqrt | Exp | Ln | Sin | Cos | Tan | Tanh | Floor | Ceil | Trunc | InverseSqrt | Exp2 | Log2 | Asin | Acos | Atan | Round, F32 | F64) => {
            let base = match op {
                Sqrt => "sqrt",
                Exp => "exp",
                Ln => "log",
                Sin => "sin",
                Cos => "cos",
                Tan => "tan",
                Tanh => "tanh",
                Floor => "floor",
                Ceil => "ceil",
                Trunc => "trunc",
                InverseSqrt => "rsqrt",
                Exp2 => "exp2",
                Log2 => "log2",
                Asin => "asin",
                Acos => "acos",
                Atan => "atan",
                _ => "rint",
            };
            r(s, &format!("{}(a)", f(base)))
        }
        (Sign, F32 | F64) => r(s, &format!("a > {zero} ? {one} : (a < {zero} ? -{one} : {zero})")),
        (Sign, I32) => r(I32, "(int32_t)((a > 0) - (a < 0))"),
        (Sign, U32) => r(U32, "a > 0u ? 1u : 0u"),
        (CountOneBits, U32) => r(U32, "(uint32_t)__popc(a)"),
        (CountOneBits, I32) => r(I32, "(int32_t)__popc((uint32_t)a)"),
        (FirstTrailingBit, U32) => r(U32, "a == 0u ? 0xffffffffu : (uint32_t)(__ffs((int)a) - 1)"),
        (FirstTrailingBit, I32) => r(I32, "(int32_t)((uint32_t)a == 0u ? 0xffffffffu : (uint32_t)(__ffs(a) - 1))"),
        (FirstLeadingBit, U32) => r(U32, "a == 0u ? 0xffffffffu : 31u - (uint32_t)__clz((int)a)"),
        (FirstLeadingBit, I32) => Some((
            I32,
            "uint32_t v = a < 0 ? ~(uint32_t)a : (uint32_t)a; return (int32_t)(v == 0u ? 0xffffffffu : 31u - (uint32_t)__clz((int)v));".into(),
        )),
        (CountLeadingZeros, U32) => r(U32, "(uint32_t)__clz((int)a)"),
        (CountLeadingZeros, I32) => r(I32, "(int32_t)__clz(a)"),
        (CountTrailingZeros, U32) => r(U32, "a == 0u ? 32u : (uint32_t)(__ffs((int)a) - 1)"),
        (CountTrailingZeros, I32) => r(I32, "(int32_t)(a == 0 ? 32 : __ffs(a) - 1)"),
        (IsNan, F32 | F64) => r(Bool, "a != a"),
        (IsInf, F32) => r(Bool, "fabsf(a) == __int_as_float(0x7f800000)"),
        (IsInf, F64) => r(Bool, "fabs(a) == __longlong_as_double(0x7ff0000000000000LL)"),
        (OpaqueGuard, F32) => Some((F32, "asm volatile(\"\" : \"+f\"(a)); return a;".into())),
        (OpaqueGuard, F64) => Some((F64, "asm volatile(\"\" : \"+d\"(a)); return a;".into())),
        (OpaqueGuard, U32 | I32) => Some((s, "asm volatile(\"\" : \"+r\"(a)); return a;".into())),
        (OpaqueGuard, Bool) => r(Bool, "a"),
        _ => None,
    }
}

/// Body of `luma_<op>(a, b)` with operand scalars `l`, `rs`, with its result type.
fn scalar_bin(op: IrBinOp, l: ScalarTy, rs: ScalarTy) -> Option<(ScalarTy, String)> {
    use IrBinOp::*;
    use ScalarTy::*;
    if l == F16 {
        let (r, _) = scalar_bin(op, F32, F32)?;
        let name = format!("luma_{}", bin_name(op));
        return Some(if r == F32 {
            (F16, ret(format!("luma_to_h({name}((float)a, (float)b))")))
        } else {
            (r, ret(format!("{name}((float)a, (float)b)")))
        });
    }
    let fl = |base: &str| -> String { if l == F32 { format!("{base}f") } else { base.to_string() } };
    let r = |t: ScalarTy, e: &str| Some((t, ret(e)));
    if op.is_comparison() {
        let o = match op {
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            _ => ">=",
        };
        return r(Bool, &format!("a {o} b"));
    }
    match (op, l) {
        (Add | Sub | Mul, F32 | F64 | U32) => {
            let o = match op {
                Add => "+",
                Sub => "-",
                _ => "*",
            };
            r(l, &format!("a {o} b"))
        }
        (Add | Sub | Mul, I32) => {
            let o = match op {
                Add => "+",
                Sub => "-",
                _ => "*",
            };
            r(I32, &format!("(int32_t)((uint32_t)a {o} (uint32_t)b)"))
        }
        (Div, F32 | F64) => r(l, "a / b"),
        (Div, U32) => r(U32, "a / (b == 0u ? 1u : b)"),
        (Div, I32) => r(I32, "a / ((b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 1 : b)"),
        (Rem, F32 | F64) => r(l, &format!("{}(a, b)", fl("fmod"))),
        (Rem, U32) => r(U32, "a % (b == 0u ? 1u : b)"),
        (Rem, I32) => r(I32, "a % ((b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 1 : b)"),
        (AddExact, F32) => r(F32, "__fadd_rn(a, b)"),
        (SubExact, F32) => r(F32, "__fsub_rn(a, b)"),
        (MulExact, F32) => r(F32, "__fmul_rn(a, b)"),
        (AddExact, F64) => r(F64, "__dadd_rn(a, b)"),
        (SubExact, F64) => r(F64, "__dsub_rn(a, b)"),
        (MulExact, F64) => r(F64, "__dmul_rn(a, b)"),
        (Min, F32 | F64) => r(l, &format!("{}(a, b)", fl("fmin"))),
        (Max, F32 | F64) => r(l, &format!("{}(a, b)", fl("fmax"))),
        (Min, U32 | I32) => r(l, "a < b ? a : b"),
        (Max, U32 | I32) => r(l, "a > b ? a : b"),
        (Pow, F32 | F64) => r(l, &format!("{}(a, b)", fl("pow"))),
        (Atan2, F32 | F64) => r(l, &format!("{}(a, b)", fl("atan2"))),
        (Copysign, F32 | F64) => r(l, &format!("{}(a, b)", fl("copysign"))),
        (BitAnd, U32 | I32) => r(l, "a & b"),
        (BitOr, U32 | I32) => r(l, "a | b"),
        (BitXor, U32 | I32) => r(l, "a ^ b"),
        (BitAnd, Bool) => r(Bool, "(bool)(a & b)"),
        (BitOr, Bool) => r(Bool, "(bool)(a | b)"),
        (BitXor, Bool) => r(Bool, "a != b"),
        (Shl, U32) if rs == U32 => r(U32, "a << (b & 31u)"),
        (Shl, I32) if rs == U32 => r(I32, "(int32_t)((uint32_t)a << (b & 31u))"),
        (Shr, U32 | I32) if rs == U32 => r(l, "a >> (b & 31u)"),
        (Rotl, U32) if rs == U32 => r(U32, "(a << (b & 31u)) | (a >> ((32u - (b & 31u)) & 31u))"),
        (Rotl, I32) if rs == U32 => {
            r(I32, "(int32_t)(((uint32_t)a << (b & 31u)) | ((uint32_t)a >> ((32u - (b & 31u)) & 31u)))")
        }
        (MulHiU32, U32) => r(U32, "__umulhi(a, b)"),
        (And, Bool) => r(Bool, "a && b"),
        (Or, Bool) => r(Bool, "a || b"),
        _ => None,
    }
}

fn infix_bin(op: IrBinOp, s: ScalarTy) -> Option<&'static str> {
    use IrBinOp::*;
    use ScalarTy::*;
    match (op, s) {
        (Add, F32 | F64 | U32) => Some("+"),
        (Sub, F32 | F64 | U32) => Some("-"),
        (Mul, F32 | F64 | U32) => Some("*"),
        (Div, F32 | F64) => Some("/"),
        (Eq, F32 | F64 | U32 | I32 | Bool) => Some("=="),
        (Ne, F32 | F64 | U32 | I32 | Bool) => Some("!="),
        (Lt, F32 | F64 | U32 | I32) => Some("<"),
        (Le, F32 | F64 | U32 | I32) => Some("<="),
        (Gt, F32 | F64 | U32 | I32) => Some(">"),
        (Ge, F32 | F64 | U32 | I32) => Some(">="),
        (And, Bool) => Some("&&"),
        (Or, Bool) => Some("||"),
        _ => None,
    }
}

/// Scalar value conversion (`Convert`); `v` is a scalar expression of type `ss`.
fn conv_scalar(ss: ScalarTy, ds: ScalarTy, v: &str) -> String {
    use ScalarTy::*;
    if ss == ds {
        return v.to_string();
    }
    match (ss, ds) {
        (F16, Bool) => format!("((float){v} != 0.0f)"),
        (_, Bool) => format!("({v} != 0)"),
        (_, F16) => format!("luma_to_h((float)({v}))"),
        (F16, d) => format!("(({})(float)({v}))", sc(d)),
        (_, d) => format!("(({})({v}))", sc(d)),
    }
}

#[derive(Default, Clone)]
struct Res {
    grid: bool,
    binds: BTreeSet<u32>,
    shared: BTreeSet<u32>,
    privs: BTreeSet<u32>,
    lens: BTreeSet<u32>,
    calls: BTreeSet<u32>,
}

impl Res {
    fn merge(&mut self, o: &Res) -> bool {
        let before = (self.binds.len(), self.shared.len(), self.privs.len(), self.lens.len());
        let g = self.grid;
        self.grid |= o.grid;
        self.binds.extend(o.binds.iter().copied());
        self.shared.extend(o.shared.iter().copied());
        self.privs.extend(o.privs.iter().copied());
        self.lens.extend(o.lens.iter().copied());
        before != (self.binds.len(), self.shared.len(), self.privs.len(), self.lens.len()) || g != self.grid
    }
}

fn tgt_res(t: &AtomicTarget, r: &mut Res) {
    match t {
        AtomicTarget::Storage { binding, .. } => {
            r.binds.insert(*binding);
        }
        AtomicTarget::Shared { shared, .. } => {
            r.shared.insert(*shared);
        }
    }
}

fn lv_res(lv: &LValue, r: &mut Res) {
    match lv {
        LValue::Local(_) => {}
        LValue::Private(p) => {
            r.privs.insert(*p);
        }
        LValue::Storage { binding, .. } => {
            r.binds.insert(*binding);
        }
        LValue::Shared { shared, .. } => {
            r.shared.insert(*shared);
        }
        LValue::Index { base, .. } | LValue::Field { base, .. } => lv_res(base, r),
    }
}

fn scan(block: &[Stmt]) -> Res {
    let mut r = Res::default();
    walk_stmts(block, &mut |s| {
        match s {
            Stmt::Store { binding, .. } => {
                r.binds.insert(*binding);
            }
            Stmt::SharedStore { shared, .. } => {
                r.shared.insert(*shared);
            }
            Stmt::Assign { dst, .. } => lv_res(dst, &mut r),
            Stmt::AtomicStore { target, .. } | Stmt::AtomicRmw { target, .. } | Stmt::AtomicCompareExchange { target, .. } => tgt_res(target, &mut r),
            _ => {}
        }
        s.for_each_expr(&mut |e| {
            e.walk(&mut |x| match x {
                Expr::UniformRead { binding, .. } | Expr::Load { binding, .. } | Expr::StorageRead { binding, .. } => {
                    r.binds.insert(*binding);
                }
                Expr::SharedLoad { shared, .. } | Expr::SharedRead { shared, .. } => {
                    r.shared.insert(*shared);
                }
                Expr::Private(p) => {
                    r.privs.insert(*p);
                }
                Expr::GridSize(_) => r.grid = true,
                Expr::ArrayLength { binding } => {
                    r.binds.insert(*binding);
                    r.lens.insert(*binding);
                }
                Expr::AtomicLoad { target, .. } => tgt_res(target, &mut r),
                Expr::Call { func, .. } => {
                    r.calls.insert(func.0);
                }
                _ => {}
            })
        });
    });
    r
}

struct FnCtx<'b> {
    locals: &'b [(String, KirType, bool)],
    params: &'b [(String, KirType)],
}

struct Em<'a> {
    m: &'a KernelModule,
    active: Vec<bool>,
    abi: Vec<Option<CudaArgAbi>>,
    res: Vec<Res>,
    vecs: BTreeSet<(u8, u8)>,
    scalar_defs: Vec<String>,
    vec_defs: Vec<String>,
    keys: HashSet<String>,
    labels: u32,
    init_shared: bool,
}

fn line(out: &mut String, ind: usize, s: &str) {
    for _ in 0..ind {
        out.push_str("    ");
    }
    out.push_str(s);
    out.push('\n');
}

impl<'a> Em<'a> {
    fn ct(&mut self, ty: &KirType) -> String {
        match ty {
            KirType::Scalar(s) | KirType::Atomic(s) => sc(*s).to_string(),
            KirType::Vec { elem, lanes } => {
                self.vecs.insert((sidx(*elem), *lanes));
                vname(*elem, *lanes)
            }
            KirType::Array { elem, len, stride } => format!("luma_arr<{}, {len}u, {stride}u>", self.ct(elem)),
            KirType::Struct(i) => format!("S{i}"),
        }
    }

    fn expr_ty(&self, e: &Expr, cx: &FnCtx) -> KirType {
        match e {
            Expr::LitBool(_) => KirType::BOOL,
            Expr::LitU32(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => KirType::U32,
            Expr::LitI32(_) => KirType::I32,
            Expr::LitF16(_) => KirType::Scalar(ScalarTy::F16),
            Expr::LitF32(_) => KirType::F32,
            Expr::LitF64(_) => KirType::Scalar(ScalarTy::F64),
            Expr::ZeroVal(t) => t.clone(),
            Expr::Local(l) => cx.locals[l.0 as usize].1.clone(),
            Expr::FnParam(i) => cx.params[*i as usize].1.clone(),
            Expr::Private(p) => self.m.privates[*p as usize].ty.clone(),
            Expr::UniformRead { ty, .. }
            | Expr::Load { ty, .. }
            | Expr::StorageRead { ty, .. }
            | Expr::SharedLoad { ty, .. }
            | Expr::SharedRead { ty, .. }
            | Expr::Access { ty, .. }
            | Expr::Swizzle { ty, .. }
            | Expr::Compose { ty, .. }
            | Expr::Unary { ty, .. }
            | Expr::Binary { ty, .. }
            | Expr::Ternary { ty, .. }
            | Expr::Subgroup { ty, .. } => ty.clone(),
            Expr::Splat { elem, lanes, .. } => KirType::vec(*elem, *lanes),
            Expr::Cast { dst_ty, .. } => dst_ty.clone(),
            Expr::Call { ty, .. } => ty.clone().unwrap_or(KirType::BOOL),
            Expr::AtomicLoad { ty, .. } => KirType::Scalar(*ty),
        }
    }

    fn step_ty(&self, base: &KirType, step: &AccessStep) -> KirType {
        match (base, step) {
            (KirType::Struct(s), AccessStep::Field(i)) => self.m.structs[*s as usize].fields[*i as usize].ty.clone(),
            (KirType::Vec { elem, .. }, _) => KirType::Scalar(*elem),
            (KirType::Array { elem, .. }, AccessStep::Index(_)) => (**elem).clone(),
            _ => base.clone(),
        }
    }

    /// Applies an access path to the C++ value `s` of type `ty`.
    fn path(&mut self, mut s: String, mut ty: KirType, steps: &[AccessStep], cx: &FnCtx) -> (String, KirType) {
        for step in steps {
            let next = self.step_ty(&ty, step);
            s = match (&ty, step) {
                (KirType::Struct(_), AccessStep::Field(i)) => format!("{s}.f{i}"),
                (KirType::Vec { .. }, AccessStep::Field(i)) => format!("{s}.{}", LANES[*i as usize]),
                (KirType::Vec { elem, lanes }, AccessStep::Index(e)) => {
                    let idx = self.expr(e, cx);
                    self.reg_vget(*elem, *lanes);
                    format!("luma_vget({s}, {idx})")
                }
                (_, AccessStep::Index(e)) => {
                    let idx = self.expr(e, cx);
                    format!("{s}[{idx}]")
                }
                (_, AccessStep::Field(i)) => format!("{s}.f{i}"),
            };
            ty = next;
        }
        (s, ty)
    }

    fn bind_elem(&mut self, binding: u32, idx: &str) -> String {
        let m = self.m;
        let d = &m.bindings[binding as usize];
        let t = self.ct(&d.elem_ty);
        let nat = cpp_size(&d.elem_ty, &m.structs);
        let stride = d.element_stride.unwrap_or(nat);
        if stride == nat {
            format!("b{binding}[{idx}]")
        } else {
            let c = if d.kind == BindingKind::StorageRead { "const " } else { "" };
            format!("(*({c}{t}*)((const char*)b{binding} + (unsigned long long)(uint32_t)({idx}) * {stride}ull))").replace("(const char*)", if c.is_empty() { "(char*)" } else { "(const char*)" })
        }
    }

    fn storage_path(&mut self, binding: u32, steps: &[AccessStep], cx: &FnCtx) -> (String, KirType) {
        let m = self.m;
        let d = &m.bindings[binding as usize];
        if d.is_unsized_array {
            match steps.split_first() {
                Some((AccessStep::Index(i), rest)) => {
                    let idx = self.expr(i, cx);
                    let base = self.bind_elem(binding, &idx);
                    self.path(base, d.elem_ty.clone(), rest, cx)
                }
                _ => panic!("luma-cuda codegen invariant: an unsized storage array binding `{}` must be indexed first", d.name),
            }
        } else {
            self.path(format!("(*b{binding})"), d.elem_ty.clone(), steps, cx)
        }
    }

    fn shared_elem(&self, shared: u32, idx: &str) -> String {
        match self.m.shared[shared as usize].len {
            // Untrusted kernels keep WGSL's clamped indexing; trusted ones index directly.
            Some(n) if self.init_shared => format!("sh{shared}[luma_ci({idx}, {n}u)]"),
            Some(_) => format!("sh{shared}[{idx}]"),
            None => format!("sh{shared}"),
        }
    }

    fn shared_root_ty(&self, shared: u32) -> KirType {
        let d = &self.m.shared[shared as usize];
        match d.len {
            Some(len) => {
                let (size, align) = d.elem_ty.layout(&self.m.structs);
                KirType::Array { elem: Box::new(d.elem_ty.clone()), len, stride: size.next_multiple_of(align) }
            }
            None => d.elem_ty.clone(),
        }
    }

    fn shared_path(&mut self, shared: u32, steps: &[AccessStep], cx: &FnCtx) -> (String, KirType) {
        let d_len = self.m.shared[shared as usize].len;
        let elem = self.m.shared[shared as usize].elem_ty.clone();
        if d_len.is_some() {
            match steps.split_first() {
                Some((AccessStep::Index(i), rest)) => {
                    let idx = self.expr(i, cx);
                    let base = self.shared_elem(shared, &idx);
                    self.path(base, elem, rest, cx)
                }
                _ => (format!("sh{shared}"), self.shared_root_ty(shared)),
            }
        } else {
            self.path(format!("sh{shared}"), elem, steps, cx)
        }
    }

    fn lvalue(&mut self, lv: &LValue, cx: &FnCtx) -> (String, KirType) {
        match lv {
            LValue::Local(l) => (format!("l{}", l.0), cx.locals[l.0 as usize].1.clone()),
            LValue::Private(p) => (format!("pv{p}"), self.m.privates[*p as usize].ty.clone()),
            LValue::Storage { binding, path } => self.storage_path(*binding, path, cx),
            LValue::Shared { shared, path } => self.shared_path(*shared, path, cx),
            LValue::Index { base, index } => {
                let (b, bty) = self.lvalue(base, cx);
                let idx = self.expr(index, cx);
                match bty {
                    KirType::Vec { elem, lanes } => {
                        self.reg_lane(elem, lanes);
                        (format!("luma_lane({b}, {idx})"), KirType::Scalar(elem))
                    }
                    KirType::Array { elem, .. } => (format!("{b}[{idx}]"), *elem),
                    other => (format!("{b}[{idx}]"), other),
                }
            }
            LValue::Field { base, field } => {
                let (b, bty) = self.lvalue(base, cx);
                match bty {
                    KirType::Vec { elem, .. } => (format!("{b}.{}", LANES[*field as usize]), KirType::Scalar(elem)),
                    KirType::Struct(s) => (format!("{b}.f{field}"), self.m.structs[s as usize].fields[*field as usize].ty.clone()),
                    other => (format!("{b}.f{field}"), other),
                }
            }
        }
    }

    fn atomic_ptr(&mut self, t: &AtomicTarget, cx: &FnCtx) -> String {
        match t {
            AtomicTarget::Storage { binding, index: Some(i) } => {
                let idx = self.expr(i, cx);
                format!("(&{})", self.bind_elem(*binding, &idx))
            }
            AtomicTarget::Storage { binding, index: None } => format!("b{binding}"),
            AtomicTarget::Shared { shared, index: Some(i) } => {
                let idx = self.expr(i, cx);
                format!("(&{})", self.shared_elem(*shared, &idx))
            }
            AtomicTarget::Shared { shared, index: None } => format!("(&sh{shared})"),
        }
    }

    fn atomic_call(op: IrAtomicOp, s: ScalarTy, p: &str, v: &str) -> String {
        use IrAtomicOp::*;
        use ScalarTy::*;
        let t = sc(s);
        let ptr = format!("(({t}*)({p}))");
        match (s, op) {
            (U32 | I32, Add) => format!("atomicAdd({ptr}, {v})"),
            (U32 | I32, Sub) => format!("atomicSub({ptr}, {v})"),
            (U32 | I32, Min) => format!("atomicMin({ptr}, {v})"),
            (U32 | I32, Max) => format!("atomicMax({ptr}, {v})"),
            (U32 | I32, And) => format!("atomicAnd({ptr}, {v})"),
            (U32 | I32, Or) => format!("atomicOr({ptr}, {v})"),
            (U32 | I32, Xor) => format!("atomicXor({ptr}, {v})"),
            (U32 | I32, Exchange) => format!("atomicExch({ptr}, {v})"),
            (F32, Add) => format!("atomicAdd({ptr}, {v})"),
            (F32, Sub) => format!("atomicAdd({ptr}, -({v}))"),
            (F32, Min) => format!("luma_atomic_fmin({ptr}, {v})"),
            (F32, Max) => format!("luma_atomic_fmax({ptr}, {v})"),
            (F32, Exchange) => format!("atomicExch({ptr}, {v})"),
            (F32, And) => format!("__uint_as_float(atomicAnd((uint32_t*)({p}), __float_as_uint({v})))"),
            (F32, Or) => format!("__uint_as_float(atomicOr((uint32_t*)({p}), __float_as_uint({v})))"),
            (F32, Xor) => format!("__uint_as_float(atomicXor((uint32_t*)({p}), __float_as_uint({v})))"),
            (F64, Add) => format!("atomicAdd({ptr}, {v})"),
            (F64, Sub) => format!("atomicAdd({ptr}, -({v}))"),
            (F64, Min) => format!("luma_atomic_dmin({ptr}, {v})"),
            (F64, Max) => format!("luma_atomic_dmax({ptr}, {v})"),
            (F64, Exchange) => format!("__longlong_as_double((long long)atomicExch((unsigned long long*)({p}), (unsigned long long)__double_as_longlong({v})))"),
            (F64, And) => format!("__longlong_as_double((long long)atomicAnd((unsigned long long*)({p}), (unsigned long long)__double_as_longlong({v})))"),
            (F64, Or) => format!("__longlong_as_double((long long)atomicOr((unsigned long long*)({p}), (unsigned long long)__double_as_longlong({v})))"),
            (F64, Xor) => format!("__longlong_as_double((long long)atomicXor((unsigned long long*)({p}), (unsigned long long)__double_as_longlong({v})))"),
            (Bool | F16, _) => panic!("luma-cuda codegen invariant: atomic {op:?} on `{}` is not a valid KIR atomic type", s.wgsl_name()),
        }
    }

    // ----- helper registration ------------------------------------------------------------

    fn reg_scalar_un(&mut self, op: IrUnOp, s: ScalarTy) {
        let name = format!("luma_{}", un_name(op));
        let key = format!("un|{name}|{}", s.wgsl_name());
        if self.keys.contains(&key) {
            return;
        }
        if s == ScalarTy::F16 {
            self.reg_scalar_un(op, ScalarTy::F32);
        }
        let (r, body) = scalar_un(op, s)
            .unwrap_or_else(|| panic!("luma-cuda codegen invariant: unary op {op:?} is not defined on `{}`", s.wgsl_name()));
        self.keys.insert(key);
        self.scalar_defs.push(format!("__device__ __forceinline__ {} {name}({} a) {{ {body} }}\n", sc(r), sc(s)));
    }

    fn reg_scalar_bin(&mut self, op: IrBinOp, l: ScalarTy, rs: ScalarTy) {
        let name = format!("luma_{}", bin_name(op));
        let key = format!("bin|{name}|{}|{}", l.wgsl_name(), rs.wgsl_name());
        if self.keys.contains(&key) {
            return;
        }
        if l == ScalarTy::F16 {
            self.reg_scalar_bin(op, ScalarTy::F32, ScalarTy::F32);
        }
        let (r, body) = scalar_bin(op, l, rs).unwrap_or_else(|| {
            panic!("luma-cuda codegen invariant: binary op {op:?} is not defined on `{}` and `{}`", l.wgsl_name(), rs.wgsl_name())
        });
        self.keys.insert(key);
        self.scalar_defs.push(format!("__device__ __forceinline__ {} {name}({} a, {} b) {{ {body} }}\n", sc(r), sc(l), sc(rs)));
    }

    fn reg_scalar_fma(&mut self, s: ScalarTy) {
        let key = format!("fma|{}", s.wgsl_name());
        if self.keys.contains(&key) {
            return;
        }
        let body = match s {
            ScalarTy::F32 => ret("__fmaf_rn(a, b, c)"),
            ScalarTy::F64 => ret("__fma_rn(a, b, c)"),
            ScalarTy::F16 => {
                self.reg_scalar_fma(ScalarTy::F32);
                ret("luma_to_h(luma_fma((float)a, (float)b, (float)c))")
            }
            other => panic!("luma-cuda codegen invariant: fma is not defined on `{}`", other.wgsl_name()),
        };
        self.keys.insert(key);
        let t = sc(s);
        self.scalar_defs.push(format!("__device__ __forceinline__ {t} luma_fma({t} a, {t} b, {t} c) {{ {body} }}\n"));
    }

    fn reg_scalar_clamp(&mut self, s: ScalarTy) {
        let key = format!("clamp|{}", s.wgsl_name());
        if self.keys.contains(&key) {
            return;
        }
        self.reg_scalar_bin(IrBinOp::Max, s, s);
        self.reg_scalar_bin(IrBinOp::Min, s, s);
        self.keys.insert(key);
        let t = sc(s);
        self.scalar_defs.push(format!("__device__ __forceinline__ {t} luma_clamp({t} a, {t} lo, {t} hi) {{ return luma_min(luma_max(a, lo), hi); }}\n"));
    }

    fn reg_sat(&mut self, ss: ScalarTy, ds: ScalarTy) -> String {
        let name = format!("luma_sat_{}_{}", ss.wgsl_name(), ds.wgsl_name());
        let key = format!("sat|{name}");
        if self.keys.insert(key) {
            let (xt, lim_hi, lim_lo, lim_u) = if ss == ScalarTy::F64 {
                ("double", "2147483648.0", "-2147483648.0", "4294967296.0")
            } else {
                ("float", "2147483648.0f", "-2147483648.0f", "4294967296.0f")
            };
            let zero = if ss == ScalarTy::F64 { "0.0" } else { "0.0f" };
            let body = match ds {
                ScalarTy::I32 => format!(
                    "{xt} x = ({xt})a; return x != x ? 0 : (x >= {lim_hi} ? 2147483647 : (x <= {lim_lo} ? (-2147483647 - 1) : (int32_t)x));"
                ),
                ScalarTy::U32 => format!(
                    "{xt} x = ({xt})a; return x != x ? 0u : (x <= {zero} ? 0u : (x >= {lim_u} ? 0xffffffffu : (uint32_t)x));"
                ),
                _ => format!("return ({})(({xt})a);", sc(ds)),
            };
            self.scalar_defs.push(format!("__device__ __forceinline__ {} {name}({} a) {{ {body} }}\n", sc(ds), sc(ss)));
        }
        name
    }

    fn reg_ew(&mut self, name: &str, args: &[KirType], rty: &KirType, call: &dyn Fn(&[String]) -> String) {
        let ats: Vec<String> = args.iter().map(|a| self.ct(a)).collect();
        let key = format!("ew|{name}|{}", ats.join(","));
        if !self.keys.insert(key) {
            return;
        }
        let rt = self.ct(rty);
        let n = rty.lanes() as usize;
        let params = ats.iter().enumerate().map(|(i, t)| format!("{t} a{i}")).collect::<Vec<_>>().join(", ");
        let lanes: Vec<String> = (0..n)
            .map(|l| {
                let xs: Vec<String> =
                    args.iter().enumerate().map(|(j, a)| if a.lanes() > 1 { format!("a{j}.{}", LANES[l]) } else { format!("a{j}") }).collect();
                call(&xs)
            })
            .collect();
        self.vec_defs.push(format!("__device__ __forceinline__ {rt} {name}({params}) {{ return {rt}{{{}}}; }}\n", lanes.join(", ")));
    }

    fn reg_vget(&mut self, elem: ScalarTy, lanes: u8) {
        let vt = vname(elem, lanes);
        if !self.keys.insert(format!("vget|{vt}")) {
            return;
        }
        self.vecs.insert((sidx(elem), lanes));
        let mut e = format!("a.{}", LANES[lanes as usize - 1]);
        for l in (0..lanes as usize - 1).rev() {
            e = format!("i == {l}u ? a.{} : ({e})", LANES[l]);
        }
        self.vec_defs.push(format!("__device__ __forceinline__ {} luma_vget({vt} a, uint32_t i) {{ return {e}; }}\n", sc(elem)));
    }

    fn reg_lane(&mut self, elem: ScalarTy, lanes: u8) {
        let vt = vname(elem, lanes);
        if !self.keys.insert(format!("lane|{vt}")) {
            return;
        }
        self.vecs.insert((sidx(elem), lanes));
        let mut e = format!("a.{}", LANES[lanes as usize - 1]);
        for l in (0..lanes as usize - 1).rev() {
            e = format!("i == {l}u ? a.{} : ({e})", LANES[l]);
        }
        self.vec_defs.push(format!("__device__ __forceinline__ {}& luma_lane({vt}& a, uint32_t i) {{ return {e}; }}\n", sc(elem)));
    }

    // ----- expressions --------------------------------------------------------------------

    fn bin_call(&mut self, op: IrBinOp, lt: &KirType, rt: &KirType, rty: &KirType, a: &str, b: &str) -> String {
        let ls = lt.elem_scalar().unwrap_or(ScalarTy::F32);
        let rs = rt.elem_scalar().unwrap_or(ScalarTy::F32);
        let both_scalar = lt.lanes() == 1 && rt.lanes() == 1;
        if both_scalar && let Some(o) = infix_bin(op, ls) {
            return format!("({a} {o} {b})");
        }
        self.reg_scalar_bin(op, ls, rs);
        let name = format!("luma_{}", bin_name(op));
        if !both_scalar {
            self.reg_ew(&name, &[lt.clone(), rt.clone()], rty, &|x| format!("{name}({}, {})", x[0], x[1]));
        }
        format!("{name}({a}, {b})")
    }

    fn un_call(&mut self, op: IrUnOp, aty: &KirType, rty: &KirType, a: &str) -> String {
        let s = aty.elem_scalar().unwrap_or(ScalarTy::F32);
        let name = format!("luma_{}", un_name(op));
        if aty.lanes() == 1 {
            match (op, s) {
                (IrUnOp::Not, ScalarTy::Bool) => return format!("(!{a})"),
                (IrUnOp::Neg, ScalarTy::F32 | ScalarTy::F64) => return format!("(-{a})"),
                (IrUnOp::BitNot, ScalarTy::U32 | ScalarTy::I32) => return format!("(~{a})"),
                _ => {}
            }
            self.reg_scalar_un(op, s);
            return format!("{name}({a})");
        }
        if matches!(op, IrUnOp::All | IrUnOp::Any) {
            let vt = self.ct(aty);
            if self.keys.insert(format!("{name}|{vt}")) {
                let j = if op == IrUnOp::All { " && " } else { " || " };
                let e = (0..aty.lanes() as usize).map(|l| format!("a.{}", LANES[l])).collect::<Vec<_>>().join(j);
                self.vec_defs.push(format!("__device__ __forceinline__ bool {name}({vt} a) {{ return {e}; }}\n"));
            }
            return format!("{name}({a})");
        }
        self.reg_scalar_un(op, s);
        self.reg_ew(&name, std::slice::from_ref(aty), rty, &|x| format!("{name}({})", x[0]));
        format!("{name}({a})")
    }

    fn bitcast(&mut self, v: &str, ss: &KirType, dst: &KirType) -> String {
        use ScalarTy::*;
        if let (KirType::Scalar(a), KirType::Scalar(b)) = (ss, dst) {
            return match (a, b) {
                (x, y) if x == y => v.to_string(),
                (F32, U32) => format!("__float_as_uint({v})"),
                (F32, I32) => format!("__float_as_int({v})"),
                (U32, F32) => format!("__uint_as_float({v})"),
                (I32, F32) => format!("__int_as_float({v})"),
                (U32, I32) => format!("((int32_t)({v}))"),
                (I32, U32) => format!("((uint32_t)({v}))"),
                _ => format!("luma_bitcast<{}>({v})", self.ct(dst)),
            };
        }
        format!("luma_bitcast<{}>({v})", self.ct(dst))
    }

    fn cast(&mut self, kind: CastKind, v: &str, src: &KirType, dst: &KirType) -> String {
        let ss = src.elem_scalar().unwrap_or(ScalarTy::F32);
        let ds = dst.elem_scalar().unwrap_or(ScalarTy::F32);
        match kind {
            CastKind::Bitcast => self.bitcast(v, src, dst),
            CastKind::Convert | CastKind::FloatToIntSat => {
                let sat = kind == CastKind::FloatToIntSat && ss.is_float() && ds.is_int();
                let sat_name = if sat { Some(self.reg_sat(ss, ds)) } else { None };
                let conv = |x: &str| match &sat_name {
                    Some(n) => format!("{n}({x})"),
                    None => conv_scalar(ss, ds, x),
                };
                if src.lanes() == 1 {
                    conv(v)
                } else {
                    let name = format!("luma_{}_{}", if sat { "satv" } else { "cvt" }, vname(ds, dst.lanes()));
                    self.reg_ew(&name, std::slice::from_ref(src), dst, &|x| conv(&x[0]));
                    format!("{name}({v})")
                }
            }
        }
    }

    fn sg_scalar(name: &str, s: ScalarTy, xs: &[String]) -> String {
        if s == ScalarTy::F16 {
            let args = xs.iter().enumerate().map(|(i, x)| if i == 0 { format!("(float)({x})") } else { x.clone() }).collect::<Vec<_>>().join(", ");
            format!("luma_to_h({name}({args}))")
        } else {
            format!("{name}({})", xs.join(", "))
        }
    }

    fn expr(&mut self, e: &Expr, cx: &FnCtx) -> String {
        let m = self.m;
        match e {
            Expr::LitBool(b) => b.to_string(),
            Expr::LitU32(v) => format!("{v}u"),
            Expr::LitI32(v) => {
                if *v == i32::MIN {
                    "(-2147483647 - 1)".into()
                } else if *v < 0 {
                    format!("({v})")
                } else {
                    v.to_string()
                }
            }
            Expr::LitF16(v) => format!("luma_to_h({})", lit_f32(*v)),
            Expr::LitF32(v) => lit_f32(*v),
            Expr::LitF64(v) => lit_f64(*v),
            Expr::ZeroVal(t) => format!("{}{{}}", self.ct(t)),
            Expr::Local(l) => format!("l{}", l.0),
            Expr::FnParam(i) => format!("a{i}"),
            Expr::Private(p) => format!("pv{p}"),
            Expr::GridSize(a) => format!("luma.size[{a}]"),
            Expr::UniformRead { binding, path, .. } => {
                let d = &m.bindings[*binding as usize];
                self.path(format!("u{binding}"), d.elem_ty.clone(), path, cx).0
            }
            Expr::Load { binding, index, .. } => {
                let i = self.expr(index, cx);
                self.bind_elem(*binding, &i)
            }
            Expr::StorageRead { binding, path, .. } => self.storage_path(*binding, path, cx).0,
            Expr::SharedLoad { shared, index, .. } => {
                let i = self.expr(index, cx);
                self.shared_elem(*shared, &i)
            }
            Expr::SharedRead { shared, path, .. } => self.shared_path(*shared, path, cx).0,
            Expr::Access { base, step, .. } => {
                let bty = self.expr_ty(base, cx);
                let b = format!("({})", self.expr(base, cx));
                self.path(b, bty, std::slice::from_ref(step), cx).0
            }
            Expr::Swizzle { base, indices, len, .. } => {
                let bty = self.expr_ty(base, cx);
                let b = self.expr(base, cx);
                let (elem, lanes) = match &bty {
                    KirType::Vec { elem, lanes } => (*elem, *lanes),
                    _ => (ScalarTy::F32, 1),
                };
                let idx = &indices[..*len as usize];
                if idx.len() == 1 {
                    format!("({b}).{}", LANES[idx[0] as usize])
                } else {
                    let vt = vname(elem, lanes);
                    let rt = self.ct(&KirType::vec(elem, idx.len() as u8));
                    let tag: String = idx.iter().map(|i| i.to_string()).collect();
                    let name = format!("luma_swz{tag}");
                    if self.keys.insert(format!("{name}|{vt}")) {
                        let comps = idx.iter().map(|i| format!("a.{}", LANES[*i as usize])).collect::<Vec<_>>().join(", ");
                        self.vec_defs.push(format!("__device__ __forceinline__ {rt} {name}({vt} a) {{ return {rt}{{{comps}}}; }}\n"));
                    }
                    format!("{name}({b})")
                }
            }
            Expr::Compose { ty, comps } => {
                let vt = self.ct(ty);
                let tys: Vec<KirType> = comps.iter().map(|c| self.expr_ty(c, cx)).collect();
                let vals: Vec<String> = comps.iter().map(|c| self.expr(c, cx)).collect();
                match ty {
                    // The wrapper holds raw bytes: elements are assigned one by one.
                    KirType::Array { .. } => format!("luma_arr_of<{vt}>({})", vals.join(", ")),
                    // Named-field assignment keeps the explicit `_pad` members out of it.
                    KirType::Struct(si) => {
                        let d = &m.structs[*si as usize];
                        assert_eq!(d.fields.len(), vals.len(), "luma-cuda codegen invariant: struct compose arity");
                        let name = format!("luma_mk_S{si}");
                        if self.keys.insert(format!("mk|{name}")) {
                            let params = d.fields.iter().enumerate().map(|(k, f)| format!("{} f{k}", self.ct(&f.ty))).collect::<Vec<_>>().join(", ");
                            let sets = (0..d.fields.len()).map(|k| format!("r.f{k} = f{k};")).collect::<Vec<_>>().join(" ");
                            self.scalar_defs.push(format!("__device__ __forceinline__ {vt} {name}({params}) {{ {vt} r = {{}}; {sets} return r; }}\n"));
                        }
                        format!("{name}({})", vals.join(", "))
                    }
                    _ if tys.iter().all(|t| t.lanes() == 1) => format!("{vt}{{{}}}", vals.join(", ")),
                    _ => {
                        let (elem, lanes) = match ty {
                            KirType::Vec { elem, lanes } => (*elem, *lanes),
                            other => panic!("luma-cuda codegen invariant: compose of {other:?}"),
                        };
                        let et = sc(elem);
                        if self.keys.insert(format!("fill|{vt}")) {
                            let sets = (0..lanes as usize).map(|l| format!("r.{} = t[{l}];", LANES[l])).collect::<Vec<_>>().join(" ");
                            self.vec_defs.push(format!("__device__ __forceinline__ void luma_fill({vt}& r, const {et}* t) {{ {sets} }}\n"));
                        }
                        for t in &tys {
                            if let KirType::Vec { elem: e2, lanes: l2 } = t {
                                let wt = vname(*e2, *l2);
                                if self.keys.insert(format!("put|{wt}")) {
                                    let puts = (0..*l2 as usize).map(|l| format!("*o++ = v.{};", LANES[l])).collect::<Vec<_>>().join(" ");
                                    self.vec_defs.push(format!("__device__ __forceinline__ void luma_put({}*& o, {wt} v) {{ {puts} }}\n", sc(*e2)));
                                }
                            }
                        }
                        format!("luma_comp<{vt}, {et}>({})", vals.join(", "))
                    }
                }
            }
            Expr::Splat { elem, lanes, val } => {
                let v = self.expr(val, cx);
                let vt = self.ct(&KirType::vec(*elem, *lanes));
                let name = format!("luma_splat_{}", vname(*elem, *lanes));
                if self.keys.insert(format!("splat|{vt}")) {
                    let comps = vec!["v"; *lanes as usize].join(", ");
                    self.vec_defs.push(format!("__device__ __forceinline__ {vt} {name}({} v) {{ return {vt}{{{comps}}}; }}\n", sc(*elem)));
                }
                format!("{name}({v})")
            }
            Expr::Unary { op, arg, ty } => {
                let aty = self.expr_ty(arg, cx);
                let a = self.expr(arg, cx);
                self.un_call(*op, &aty, ty, &a)
            }
            Expr::Binary { op, lhs, rhs, ty } => {
                let lt = self.expr_ty(lhs, cx);
                let rt = self.expr_ty(rhs, cx);
                let a = self.expr(lhs, cx);
                let b = self.expr(rhs, cx);
                self.bin_call(*op, &lt, &rt, ty, &a, &b)
            }
            Expr::Ternary { op, a, b, c, ty } => {
                let ta = self.expr_ty(a, cx);
                let tb = self.expr_ty(b, cx);
                let tc = self.expr_ty(c, cx);
                let ea = self.expr(a, cx);
                let eb = self.expr(b, cx);
                let ec = self.expr(c, cx);
                let s = ty.elem_scalar().unwrap_or(ScalarTy::F32);
                match op {
                    IrTernaryOp::Select => {
                        if ta.lanes() == 1 {
                            format!("({ea} ? {eb} : {ec})")
                        } else {
                            self.reg_ew("luma_select", &[ta, tb, tc], ty, &|x| format!("({} ? {} : {})", x[0], x[1], x[2]));
                            format!("luma_select({ea}, {eb}, {ec})")
                        }
                    }
                    IrTernaryOp::Fma => {
                        self.reg_scalar_fma(s);
                        if ty.lanes() > 1 {
                            self.reg_ew("luma_fma", &[ta, tb, tc], ty, &|x| format!("luma_fma({}, {}, {})", x[0], x[1], x[2]));
                        }
                        format!("luma_fma({ea}, {eb}, {ec})")
                    }
                    IrTernaryOp::Clamp => {
                        self.reg_scalar_clamp(s);
                        if ty.lanes() > 1 {
                            self.reg_ew("luma_clamp", &[ta, tb, tc], ty, &|x| format!("luma_clamp({}, {}, {})", x[0], x[1], x[2]));
                        }
                        format!("luma_clamp({ea}, {eb}, {ec})")
                    }
                }
            }
            Expr::Cast { kind, expr, src_ty, dst_ty } => {
                let v = self.expr(expr, cx);
                self.cast(*kind, &v, src_ty, dst_ty)
            }
            Expr::Call { func, args, .. } => {
                let mut xs: Vec<String> = args.iter().map(|a| self.expr(a, cx)).collect();
                xs.extend(self.ctx_args(&self.res[func.0 as usize].clone()));
                format!("f{}({})", func.0, xs.join(", "))
            }
            Expr::AtomicLoad { target, ty } => {
                let p = self.atomic_ptr(target, cx);
                format!("(*(volatile {}*)({p}))", sc(*ty))
            }
            Expr::Subgroup { op, arg, lane, ty } => {
                let aty = arg.as_ref().map(|a| self.expr_ty(a, cx)).unwrap_or(KirType::BOOL);
                let a = arg.as_ref().map(|a| self.expr(a, cx));
                let l = lane.as_ref().map(|l| self.expr(l, cx));
                if *op == IrSubgroupOp::Ballot {
                    self.vecs.insert((sidx(ScalarTy::U32), 4));
                    return format!("luma_ballot({})", a.unwrap_or_else(|| "true".into()));
                }
                let a = a.unwrap_or_default();
                let name = match (op, &l) {
                    (IrSubgroupOp::Add, _) => "luma_sg_add",
                    (IrSubgroupOp::Min, _) => "luma_sg_min",
                    (IrSubgroupOp::Max, _) => "luma_sg_max",
                    (IrSubgroupOp::ExclusiveAdd, _) => "luma_sg_xadd",
                    (_, Some(_)) => "luma_sg_shuffle",
                    _ => "luma_sg_first",
                };
                let s = aty.elem_scalar().unwrap_or(ScalarTy::F32);
                let mut xs = vec![a];
                if let Some(l) = l {
                    xs.push(l);
                }
                if aty.lanes() == 1 {
                    Self::sg_scalar(name, s, &xs)
                } else {
                    let mut tys = vec![aty.clone()];
                    if xs.len() > 1 {
                        tys.push(KirType::U32);
                    }
                    self.reg_ew(name, &tys, ty, &|x| Self::sg_scalar(name, s, x));
                    format!("{name}({})", xs.join(", "))
                }
            }
            Expr::ArrayLength { binding } => format!("_luma_len{binding}"),
        }
    }

    /// The arguments a helper function needs to be handed besides its own.
    fn ctx_args(&self, r: &Res) -> Vec<String> {
        let mut v = Vec::new();
        if r.grid {
            v.push("luma".into());
        }
        for b in &r.binds {
            let d = &self.m.bindings[*b as usize];
            v.push(if d.kind == BindingKind::Uniform { format!("u{b}") } else { format!("b{b}") });
        }
        for s in &r.shared {
            v.push(format!("sh{s}"));
        }
        for p in &r.privs {
            v.push(format!("pv{p}"));
        }
        for l in &r.lens {
            v.push(format!("_luma_len{l}"));
        }
        v
    }

    fn ctx_params(&mut self, r: &Res) -> Vec<String> {
        let m = self.m;
        let mut v = Vec::new();
        if r.grid {
            v.push("const LumaGrid& luma".to_string());
        }
        for b in &r.binds {
            let d = &m.bindings[*b as usize];
            let t = self.ct(&d.elem_ty);
            v.push(match d.kind {
                BindingKind::Uniform => format!("const {t}& u{b}"),
                BindingKind::StorageRead => format!("const {t}* b{b}"),
                BindingKind::StorageReadWrite => format!("{t}* b{b}"),
            });
        }
        for s in &r.shared {
            let d = &m.shared[*s as usize];
            let t = self.ct(&d.elem_ty);
            v.push(if d.len.is_some() { format!("{t}* sh{s}") } else { format!("{t}& sh{s}") });
        }
        for p in &r.privs {
            let t = self.ct(&m.privates[*p as usize].ty);
            v.push(format!("{t}& pv{p}"));
        }
        for l in &r.lens {
            v.push(format!("uint32_t _luma_len{l}"));
        }
        v
    }

    // ----- statements ---------------------------------------------------------------------

    fn stmts(&mut self, block: &[Stmt], cx: &FnCtx, out: &mut String, ind: usize) {
        for s in block {
            self.stmt(s, cx, out, ind);
        }
    }

    fn stmt(&mut self, s: &Stmt, cx: &FnCtx, out: &mut String, ind: usize) {
        match s {
            Stmt::Let { local, init } => {
                let v = self.expr(init, cx);
                line(out, ind, &format!("l{} = {v};", local.0));
            }
            Stmt::Var { local, init } => {
                let v = match init {
                    Some(e) => self.expr(e, cx),
                    None => format!("{}{{}}", self.ct(&cx.locals[local.0 as usize].1)),
                };
                line(out, ind, &format!("l{} = {v};", local.0));
            }
            Stmt::Assign { dst, val } => {
                let (lv, _) = self.lvalue(dst, cx);
                let v = self.expr(val, cx);
                line(out, ind, &format!("{lv} = {v};"));
            }
            Stmt::Store { binding, index, val } => {
                let i = self.expr(index, cx);
                let v = self.expr(val, cx);
                let place = self.bind_elem(*binding, &i);
                line(out, ind, &format!("{place} = {v};"));
            }
            Stmt::SharedStore { shared, index, val } => {
                let i = self.expr(index, cx);
                let v = self.expr(val, cx);
                let place = self.shared_elem(*shared, &i);
                line(out, ind, &format!("{place} = {v};"));
            }
            Stmt::If { cond, then_body, else_body } => {
                let c = self.expr(cond, cx);
                line(out, ind, &format!("if ({c}) {{"));
                self.stmts(then_body, cx, out, ind + 1);
                if else_body.is_empty() {
                    line(out, ind, "}");
                } else {
                    line(out, ind, "} else {");
                    self.stmts(else_body, cx, out, ind + 1);
                    line(out, ind, "}");
                }
            }
            Stmt::Loop { body, continuing, break_if } => {
                let id = self.labels;
                self.labels += 1;
                line(out, ind, "{");
                line(out, ind + 1, &format!("bool lc{id} = false;"));
                line(out, ind + 1, "for (;;) {");
                line(out, ind + 2, &format!("if (lc{id}) {{"));
                self.stmts(continuing, cx, out, ind + 3);
                if let Some(b) = break_if {
                    let c = self.expr(b, cx);
                    line(out, ind + 3, &format!("if ({c}) break;"));
                }
                line(out, ind + 2, "}");
                line(out, ind + 2, &format!("lc{id} = true;"));
                self.stmts(body, cx, out, ind + 2);
                line(out, ind + 1, "}");
                line(out, ind, "}");
            }
            Stmt::Switch { selector, cases, default } => {
                let sel = self.expr(selector, cx);
                line(out, ind, &format!("switch ({sel}) {{"));
                for (c, body, fall) in cases {
                    let lbl = match c {
                        SwitchCase::I32(v) => format!("case {v}:"),
                        SwitchCase::U32(v) => format!("case {v}u:"),
                    };
                    line(out, ind + 1, &format!("{lbl} {{"));
                    self.stmts(body, cx, out, ind + 2);
                    line(out, ind + 1, "}");
                    if !fall {
                        line(out, ind + 1, "break;");
                    }
                }
                line(out, ind + 1, "default: {");
                self.stmts(default, cx, out, ind + 2);
                line(out, ind + 1, "}");
                line(out, ind + 1, "break;");
                line(out, ind, "}");
            }
            Stmt::Break => line(out, ind, "break;"),
            Stmt::Continue => line(out, ind, "continue;"),
            Stmt::Return(None) => line(out, ind, "return;"),
            Stmt::Return(Some(e)) => {
                let v = self.expr(e, cx);
                line(out, ind, &format!("return {v};"));
            }
            Stmt::Barrier(scope) => match scope {
                BarrierScope::Workgroup => line(out, ind, "__syncthreads();"),
                BarrierScope::Storage | BarrierScope::StorageAndWorkgroup => line(out, ind, "__threadfence(); __syncthreads();"),
            },
            Stmt::AtomicStore { target, val } => {
                let vt = self.expr_ty(val, cx);
                let t = self.ct(&vt);
                let p = self.atomic_ptr(target, cx);
                let v = self.expr(val, cx);
                line(out, ind, &format!("*(volatile {t}*)({p}) = {v};"));
            }
            Stmt::AtomicRmw { dst, op, elem_ty, target, val } => {
                let p = self.atomic_ptr(target, cx);
                let v = self.expr(val, cx);
                let call = Self::atomic_call(*op, *elem_ty, &p, &v);
                match dst {
                    Some(d) => line(out, ind, &format!("l{} = {call};", d.0)),
                    None => line(out, ind, &format!("(void){call};")),
                }
            }
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, elem_ty, target, cmp, val } => {
                let p = self.atomic_ptr(target, cx);
                let c = self.expr(cmp, cx);
                let v = self.expr(val, cx);
                let (o, x) = (dst_old.0, dst_exchanged.0);
                // Target, comparator and value are each evaluated once; the swap is judged on the
                // returned and expected *bits*, so NaN payloads and signed zeros compare exactly.
                let text = match elem_ty {
                    ScalarTy::U32 | ScalarTy::I32 => {
                        let t = sc(*elem_ty);
                        format!("{{ {t} c = {c}; {t} n = {v}; {t} r = atomicCAS(({t}*)({p}), c, n); l{o} = r; l{x} = (r == c); }}")
                    }
                    ScalarTy::F32 => format!(
                        "{{ uint32_t c = __float_as_uint({c}); uint32_t n = __float_as_uint({v}); uint32_t r = atomicCAS((uint32_t*)({p}), c, n); l{o} = __uint_as_float(r); l{x} = (r == c); }}"
                    ),
                    ScalarTy::F64 => format!(
                        "{{ unsigned long long c = (unsigned long long)__double_as_longlong({c}); unsigned long long n = (unsigned long long)__double_as_longlong({v}); unsigned long long r = atomicCAS((unsigned long long*)({p}), c, n); l{o} = __longlong_as_double((long long)r); l{x} = (r == c); }}"
                    ),
                    other => panic!("luma-cuda codegen invariant: compare-exchange on `{}` is not a valid KIR atomic type", other.wgsl_name()),
                };
                line(out, ind, &text);
            }
            Stmt::Discard(e) => {
                let v = self.expr(e, cx);
                line(out, ind, &format!("(void)({v});"));
            }
        }
    }

    /// Locals are declared without initializers: `Stmt::Var` writes its own typed zero or init,
    /// and everything else (`Let`, builtins, atomic results) is assigned before it is read.
    fn decl_locals(&mut self, locals: &[(String, KirType, bool)], out: &mut String, ind: usize) {
        for (i, (_, ty, _)) in locals.iter().enumerate() {
            let t = self.ct(ty);
            line(out, ind, &format!("{t} l{i};"));
        }
    }

    // ----- module-level pieces --------------------------------------------------------------

    fn struct_def(&mut self, i: usize, done: &mut Vec<bool>, out: &mut String) {
        if done[i] {
            return;
        }
        done[i] = true;
        let m = self.m;
        let d = &m.structs[i];
        fn deps(t: &KirType, f: &mut dyn FnMut(usize)) {
            match t {
                KirType::Struct(j) => f(*j as usize),
                KirType::Array { elem, .. } => deps(elem, f),
                _ => {}
            }
        }
        let mut need = Vec::new();
        for fld in &d.fields {
            deps(&fld.ty, &mut |j| need.push(j));
        }
        for j in need {
            self.struct_def(j, done, out);
        }
        let mut body = String::new();
        let mut cur = 0u32;
        for (k, fld) in d.fields.iter().enumerate() {
            if fld.offset > cur {
                body.push_str(&format!("    uint8_t _pad{k}[{}];\n", fld.offset - cur));
            }
            let t = self.ct(&fld.ty);
            body.push_str(&format!("    {t} f{k};\n"));
            cur = fld.offset + cpp_size(&fld.ty, &m.structs);
        }
        if d.size > cur {
            body.push_str(&format!("    uint8_t _tail[{}];\n", d.size - cur));
        } else if d.fields.is_empty() {
            body.push_str("    uint8_t _tail[1];\n");
        }
        out.push_str(&format!("struct alignas({}) S{i} {{\n{body}}};\n", d.align.max(1)));
    }

    fn vec_struct(s: ScalarTy, lanes: u8) -> String {
        let sz = s.size_bytes();
        let comps = LANES[..lanes as usize].join(", ");
        let al = match lanes {
            2 => format!("alignas({}) ", 2 * sz),
            4 => format!("alignas({}) ", 4 * sz),
            _ => String::new(),
        };
        format!("struct {al}{} {{ {} {comps}; }};\n", vname(s, lanes), sc(s))
    }

    /// Bytes of the kernel parameter area, laying each parameter out at its natural alignment.
    fn param_bytes(&self, n_lens: usize) -> usize {
        let m = self.m;
        // LumaGrid (16 bytes, align 4), then `luma_indirect` (8, align 8).
        let mut off = 16usize.next_multiple_of(4);
        off = off.next_multiple_of(8) + 8;
        for (i, b) in m.bindings.iter().enumerate() {
            if !self.active[i] {
                continue;
            }
            let (size, align) = match self.abi[i] {
                Some(CudaArgAbi::InlineUniform { size }) => (size, cpp_align(&b.elem_ty, &m.structs) as usize),
                _ => (8, 8),
            };
            off = off.next_multiple_of(align) + size;
        }
        for _ in 0..n_lens {
            off = off.next_multiple_of(4) + 4;
        }
        off
    }

    fn plan_abi(&mut self, lens: &[u32]) {
        let m = self.m;
        let mut uniforms: Vec<(usize, usize)> = (0..m.bindings.len())
            .filter(|&i| self.active[i] && m.bindings[i].kind == BindingKind::Uniform)
            .map(|i| (cpp_size(&m.bindings[i].elem_ty, &m.structs) as usize, i))
            .collect();
        for (i, b) in m.bindings.iter().enumerate() {
            if self.active[i] {
                self.abi[i] = Some(if b.kind == BindingKind::Uniform { CudaArgAbi::UniformPtr } else { CudaArgAbi::DevicePtr });
            }
        }
        // Smallest first: each uniform goes inline only if it is at most MAX_INLINE_UNIFORM bytes
        // and the whole parameter area, alignment padding included, stays within PARAM_BYTES.
        uniforms.sort();
        for (size, i) in uniforms {
            if size > MAX_INLINE_UNIFORM {
                continue;
            }
            self.abi[i] = Some(CudaArgAbi::InlineUniform { size });
            if self.param_bytes(lens.len()) > PARAM_BYTES {
                self.abi[i] = Some(CudaArgAbi::UniformPtr);
            }
        }
    }

    fn run(mut self) -> CudaTranslated {
        let m = self.m;
        let live = m.reachable_functions();

        // Resources per function, closed over calls.
        let direct: Vec<Res> = m.functions.iter().map(|f| scan(&f.body)).collect();
        let mut total = direct.clone();
        loop {
            let mut changed = false;
            for i in 0..total.len() {
                for c in direct[i].calls.clone() {
                    let other = total[c as usize].clone();
                    if total[i].merge(&other) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        self.res = total;
        let mut entry_res = scan(&m.entry_body);
        for c in entry_res.calls.clone() {
            let o = self.res[c as usize].clone();
            entry_res.merge(&o);
        }
        let lens: Vec<u32> = entry_res.lens.iter().copied().filter(|&b| self.active.get(b as usize).copied().unwrap_or(false)).collect();
        self.plan_abi(&lens);

        // Force-register what the prologue itself needs.
        self.vecs.insert((sidx(ScalarTy::U32), 3));

        // Module structs.
        let mut structs = String::new();
        let mut done = vec![false; m.structs.len()];
        for i in 0..m.structs.len() {
            self.struct_def(i, &mut done, &mut structs);
        }

        // Functions.
        let mut protos = String::new();
        let mut defs = String::new();
        for (i, f) in m.functions.iter().enumerate() {
            if !live[i] {
                continue;
            }
            let rt = match &f.ret_ty {
                Some(t) => self.ct(t),
                None => "void".to_string(),
            };
            let r = self.res[i].clone();
            let mut params: Vec<String> = Vec::new();
            for (k, (_, t)) in f.params.iter().enumerate() {
                params.push(format!("{} a{k}", self.ct(t)));
            }
            params.extend(self.ctx_params(&r));
            let sig = format!("__device__ __forceinline__ {rt} f{i}({})", params.join(", "));
            protos.push_str(&format!("{sig};\n"));
            let cx = FnCtx { locals: &f.locals, params: &f.params };
            defs.push_str(&format!("{sig} {{\n"));
            self.decl_locals(&f.locals, &mut defs, 1);
            self.stmts(&f.body, &cx, &mut defs, 1);
            defs.push_str("}\n");
        }

        // Kernel.
        let mut kparams: Vec<String> = vec!["LumaGrid luma".into(), "const luma_u32x3* __restrict__ luma_indirect".into()];
        let mut prologue = String::new();
        let mut param_abis = Vec::new();
        for (i, d) in m.bindings.iter().enumerate() {
            if !self.active[i] {
                continue;
            }
            let t = self.ct(&d.elem_ty);
            let abi = self.abi[i].unwrap_or(CudaArgAbi::DevicePtr);
            param_abis.push(abi);
            match (d.kind, abi) {
                (BindingKind::StorageRead, _) => kparams.push(format!("const {t}* __restrict__ b{i}")),
                (BindingKind::StorageReadWrite, _) => kparams.push(format!("{t}* b{i}")),
                (BindingKind::Uniform, CudaArgAbi::InlineUniform { .. }) => kparams.push(format!("const {t} u{i}")),
                (BindingKind::Uniform, _) => {
                    kparams.push(format!("const {t}* __restrict__ u{i}_p"));
                    line(&mut prologue, 1, &format!("const {t}& u{i} = *u{i}_p;"));
                }
            }
        }
        for l in &lens {
            kparams.push(format!("uint32_t _luma_len{l}"));
        }

        let mut kernel = String::new();
        let [wx, wy, wz] = m.workgroup_size;
        kernel.push_str(&format!("extern \"C\" __global__ void __launch_bounds__(LUMA_WG_INVOCATIONS) luma_main({}) {{\n", kparams.join(", ")));
        line(
            &mut kernel,
            1,
            "if (luma_indirect != nullptr && (blockIdx.x >= luma_indirect->x || blockIdx.y >= luma_indirect->y || blockIdx.z >= luma_indirect->z)) return;",
        );
        kernel.push_str(&prologue);
        let cx = FnCtx { locals: &m.entry_locals, params: &[] };
        // Workgroup memory, zero-initialised cooperatively.
        let mut any_shared = false;
        for (i, d) in m.shared.iter().enumerate() {
            let t = self.ct(&d.elem_ty);
            match d.len {
                Some(n) => line(&mut kernel, 1, &format!("__shared__ {t} sh{i}[{n}];")),
                None => line(&mut kernel, 1, &format!("__shared__ {t} sh{i};")),
            }
            if self.init_shared {
                line(&mut kernel, 1, &format!("for (uint32_t z = LUMA_LINEAR; z < sizeof(sh{i}); z += LUMA_WG_INVOCATIONS) ((unsigned char*)&sh{i})[z] = 0;"));
                any_shared = true;
            }
        }
        if any_shared {
            line(&mut kernel, 1, "__syncthreads();");
        }
        for (i, p) in m.privates.iter().enumerate() {
            let t = self.ct(&p.ty);
            match &p.init {
                Some(e) => {
                    let v = self.expr(e, &cx);
                    line(&mut kernel, 1, &format!("{t} pv{i} = {v};"));
                }
                None => line(&mut kernel, 1, &format!("{t} pv{i} = {{}};")),
            }
        }
        self.decl_locals(&m.entry_locals, &mut kernel, 1);
        for (b, l) in &m.entry_builtins {
            let v = match b {
                BuiltinVar::GlobalInvocationId => {
                    "luma_u32x3{blockIdx.x * LUMA_WG_X + threadIdx.x, blockIdx.y * LUMA_WG_Y + threadIdx.y, blockIdx.z * LUMA_WG_Z + threadIdx.z}"
                        .to_string()
                }
                BuiltinVar::LocalInvocationId => "luma_u32x3{threadIdx.x, threadIdx.y, threadIdx.z}".to_string(),
                BuiltinVar::LocalInvocationIndex => "LUMA_LINEAR".to_string(),
                BuiltinVar::WorkgroupId => "luma_u32x3{blockIdx.x, blockIdx.y, blockIdx.z}".to_string(),
                BuiltinVar::NumWorkgroups => {
                    "luma_u32x3{luma_indirect != nullptr ? luma_indirect->x : gridDim.x, luma_indirect != nullptr ? luma_indirect->y : gridDim.y, luma_indirect != nullptr ? luma_indirect->z : gridDim.z}"
                        .to_string()
                }
                BuiltinVar::SubgroupInvocationId => "luma_lane_id()".to_string(),
                BuiltinVar::SubgroupSize => "32u".to_string(),
            };
            line(&mut kernel, 1, &format!("l{} = {v};", l.0));
        }
        self.stmts(&m.entry_body, &cx, &mut kernel, 1);
        kernel.push_str("}\n");

        // Assemble.
        let mut vec_types = String::new();
        for (s, l) in self.vecs.clone() {
            vec_types.push_str(&Self::vec_struct(sfrom(s), l));
        }
        let mut body = String::new();
        body.push_str(&structs);
        for d in &self.scalar_defs {
            body.push_str(d);
        }
        for d in &self.vec_defs {
            body.push_str(d);
        }
        body.push_str(&protos);
        body.push_str(&defs);
        body.push_str(&kernel);

        let mut src = String::new();
        src.push_str(&format!(
            "// {}\n#define LUMA_WG_X {wx}u\n#define LUMA_WG_Y {wy}u\n#define LUMA_WG_Z {wz}u\n#define LUMA_WG_INVOCATIONS {}u\n#define LUMA_LINEAR (threadIdx.x + LUMA_WG_X * (threadIdx.y + LUMA_WG_Y * threadIdx.z))\n",
            m.label.replace(['\n', '\r'], " "),
            wx * wy * wz
        ));
        src.push_str(PRELUDE_CORE);
        let all = format!("{vec_types}{body}");
        if all.contains("luma_h") {
            src.push_str(PRELUDE_HALF);
        }
        if all.contains("luma_arr<") {
            src.push_str(PRELUDE_ARR);
        }
        if all.contains("luma_ci(") {
            src.push_str(PRELUDE_CI);
        }
        if all.contains("luma_bitcast<") {
            src.push_str(PRELUDE_BITCAST);
        }
        if all.contains("luma_comp<") {
            src.push_str(PRELUDE_COMP);
        }
        if all.contains("luma_atomic_") {
            src.push_str(PRELUDE_ATOMIC);
        }
        if all.contains("luma_sg_") || all.contains("luma_lane_id") || all.contains("luma_ballot") {
            src.push_str(PRELUDE_SUBGROUP);
        }
        src.push_str(&vec_types);
        if all.contains("luma_ballot(") {
            src.push_str(PRELUDE_BALLOT);
        }
        src.push_str(&body);

        CudaTranslated { source: src, param_abis, array_lengths: lens }
    }
}

const PRELUDE_CORE: &str = r#"typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
struct LumaGrid { uint32_t size[3]; uint32_t _pad; };
"#;

const PRELUDE_HALF: &str = r#"struct luma_h {
    uint16_t bits;
    __device__ __forceinline__ operator float() const { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits)); return f; }
};
__device__ __forceinline__ luma_h luma_to_h(float f) { luma_h h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h.bits) : "f"(f)); return h; }
"#;

const PRELUDE_ARR: &str = r#"template<class T, uint32_t N, uint32_t S> struct luma_arr {
    alignas(alignof(T)) uint8_t raw[(N * S) ? (N * S) : 1u];
    __device__ __forceinline__ T& operator[](uint32_t i) { uint32_t j = i < N ? i : (N ? N - 1u : 0u); return *reinterpret_cast<T*>(raw + (unsigned long long)j * S); }
    __device__ __forceinline__ const T& operator[](uint32_t i) const { uint32_t j = i < N ? i : (N ? N - 1u : 0u); return *reinterpret_cast<const T*>(raw + (unsigned long long)j * S); }
};
template<class A, class... E> __device__ __forceinline__ A luma_arr_of(E... e) { A r = {}; uint32_t i = 0u; ((r[i++] = e), ...); return r; }
"#;

const PRELUDE_CI: &str = "__device__ __forceinline__ uint32_t luma_ci(uint32_t i, uint32_t n) { return i < n ? i : n - 1u; }\n";

const PRELUDE_BITCAST: &str = r#"template<class To, class From> __device__ __forceinline__ To luma_bitcast(From f) { To t; __builtin_memcpy(&t, &f, sizeof(To)); return t; }
"#;

const PRELUDE_COMP: &str = r#"template<class E> __device__ __forceinline__ void luma_put(E*& o, E v) { *o++ = v; }
template<class V, class E, class... A> __device__ __forceinline__ V luma_comp(A... a) { E t[4]; E* p = t; (luma_put(p, a), ...); V r; luma_fill(r, t); return r; }
"#;

const PRELUDE_ATOMIC: &str = r#"__device__ __forceinline__ float luma_atomic_fmin(float* p, float v) {
    uint32_t* u = (uint32_t*)p; uint32_t old = *(volatile uint32_t*)u, assumed;
    do { assumed = old; float nv = fminf(__uint_as_float(assumed), v); if (__float_as_uint(nv) == assumed) break; old = atomicCAS(u, assumed, __float_as_uint(nv)); } while (assumed != old);
    return __uint_as_float(old);
}
__device__ __forceinline__ float luma_atomic_fmax(float* p, float v) {
    uint32_t* u = (uint32_t*)p; uint32_t old = *(volatile uint32_t*)u, assumed;
    do { assumed = old; float nv = fmaxf(__uint_as_float(assumed), v); if (__float_as_uint(nv) == assumed) break; old = atomicCAS(u, assumed, __float_as_uint(nv)); } while (assumed != old);
    return __uint_as_float(old);
}
__device__ __forceinline__ double luma_atomic_dmin(double* p, double v) {
    unsigned long long* u = (unsigned long long*)p; unsigned long long old = *(volatile unsigned long long*)u, assumed;
    do { assumed = old; double nv = fmin(__longlong_as_double((long long)assumed), v); if ((unsigned long long)__double_as_longlong(nv) == assumed) break; old = atomicCAS(u, assumed, (unsigned long long)__double_as_longlong(nv)); } while (assumed != old);
    return __longlong_as_double((long long)old);
}
__device__ __forceinline__ double luma_atomic_dmax(double* p, double v) {
    unsigned long long* u = (unsigned long long*)p; unsigned long long old = *(volatile unsigned long long*)u, assumed;
    do { assumed = old; double nv = fmax(__longlong_as_double((long long)assumed), v); if ((unsigned long long)__double_as_longlong(nv) == assumed) break; old = atomicCAS(u, assumed, (unsigned long long)__double_as_longlong(nv)); } while (assumed != old);
    return __longlong_as_double((long long)old);
}
"#;

const PRELUDE_SUBGROUP: &str = r#"__device__ __forceinline__ uint32_t luma_lane_id() { uint32_t l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
template<class T> __device__ __forceinline__ T luma_sg_ad(T a, T b) { return a + b; }
__device__ __forceinline__ int32_t luma_sg_ad(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
template<class T> __device__ __forceinline__ T luma_sg_mn(T a, T b) { return b < a ? b : a; }
__device__ __forceinline__ float luma_sg_mn(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double luma_sg_mn(double a, double b) { return fmin(a, b); }
template<class T> __device__ __forceinline__ T luma_sg_mx(T a, T b) { return b > a ? b : a; }
__device__ __forceinline__ float luma_sg_mx(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double luma_sg_mx(double a, double b) { return fmax(a, b); }
template<int OP, class T> __device__ __forceinline__ T luma_sg_op(T a, T b) { return OP == 0 ? luma_sg_ad(a, b) : (OP == 1 ? luma_sg_mn(a, b) : luma_sg_mx(a, b)); }
template<int OP, class T> __device__ __forceinline__ T luma_sg_red(T v) {
    unsigned m = __activemask();
    if (m == 0xffffffffu) {
        // Full warp: xor butterfly (pairwise tree). Each pair combines commutatively, so all lanes agree bit for bit.
        for (int o = 16; o > 0; o >>= 1) v = luma_sg_op<OP>(v, __shfl_xor_sync(m, v, o));
        return v;
    }
    unsigned rest = m & (m - 1u);
    T acc = __shfl_sync(m, v, __ffs(m) - 1);
    while (rest) {
        int l = __ffs(rest) - 1; rest &= rest - 1u;
        acc = luma_sg_op<OP>(acc, __shfl_sync(m, v, l));
    }
    return acc;
}
template<class T> __device__ __forceinline__ T luma_sg_add(T v) { return luma_sg_red<0>(v); }
template<class T> __device__ __forceinline__ T luma_sg_min(T v) { return luma_sg_red<1>(v); }
template<class T> __device__ __forceinline__ T luma_sg_max(T v) { return luma_sg_red<2>(v); }
template<class T> __device__ __forceinline__ T luma_sg_xadd(T v) {
    unsigned m = __activemask(); unsigned lane = luma_lane_id();
    if (m == 0xffffffffu) {
        // Full warp: Hillis-Steele inclusive scan with shfl_up, then shift by one lane (no subtraction, so floats stay exact).
        T s = v;
        for (int o = 1; o < 32; o <<= 1) { T t = __shfl_up_sync(m, s, o); if (lane >= (unsigned)o) s = luma_sg_ad(s, t); }
        T ex = __shfl_up_sync(m, s, 1);
        return lane == 0u ? T() : ex;
    }
    T acc = T(); unsigned rest = m;
    while (rest) {
        int l = __ffs(rest) - 1; rest &= rest - 1u;
        T x = __shfl_sync(m, v, l);
        if ((unsigned)l < lane) acc = luma_sg_ad(acc, x);
    }
    return acc;
}
template<class T> __device__ __forceinline__ T luma_sg_shuffle(T v, uint32_t l) { return __shfl_sync(__activemask(), v, l); }
template<class T> __device__ __forceinline__ T luma_sg_first(T v) { unsigned m = __activemask(); return __shfl_sync(m, v, __ffs(m) - 1); }
"#;

const PRELUDE_BALLOT: &str = r#"__device__ __forceinline__ luma_u32x4 luma_ballot(bool p) { luma_u32x4 r; r.x = __ballot_sync(__activemask(), p); r.y = 0u; r.z = 0u; r.w = 0u; return r; }
"#;

/// Translates `module` to NVRTC-ready CUDA C++. `used` is the reachability mask of bindings
/// (`None` keeps all); inactive bindings get no kernel parameter. Workgroup memory is zeroed
/// (see [`emit_with_workgroup_init`]).
pub fn emit(module: &KernelModule, used: Option<&[bool]>) -> CudaTranslated {
    emit_with_workgroup_init(module, used, true)
}

/// Like [`emit`]; with `initialize == false` no workgroup-memory zeroing loop or barrier is
/// emitted (for trusted generated kernels that write before they read). With `true` every
/// `__shared__` variable is cleared byte-wise over its whole `sizeof` (so half, bool, atomic
/// and aggregate representations, tail padding included, start at zero) and a `__syncthreads()`
/// follows. The kernel ABI is identical either way.
pub fn emit_with_workgroup_init(module: &KernelModule, used: Option<&[bool]>, initialize: bool) -> CudaTranslated {
    let n = module.bindings.len();
    let active: Vec<bool> = (0..n).map(|i| used.is_none_or(|u| u.get(i).copied().unwrap_or(false))).collect();
    Em {
        m: module,
        active,
        abi: vec![None; n],
        res: Vec::new(),
        vecs: BTreeSet::new(),
        scalar_defs: Vec::new(),
        vec_defs: Vec::new(),
        keys: HashSet::new(),
        labels: 0,
        init_shared: initialize,
    }
    .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(f: impl FnOnce(&mut ModuleBuilder)) -> KernelModule {
        let mut mb = ModuleBuilder::new("t", "main", [64, 1, 1]);
        f(&mut mb);
        mb.finish()
    }

    #[test]
    fn mulhi_exact_indirect_and_no_includes() {
        let m = build(|mb| {
            let p = mb.add_uniform_u32_struct("P", &["n", "k"]);
            let params = mb.add_uniform("p", KirType::Struct(p));
            let out = mb.add_storage_read_write("out", KirType::U32);
            let _unused = mb.add_storage_read("unused", KirType::F32);
            let mut b = mb.entry_builder();
            let gid = b.builtin(BuiltinVar::GlobalInvocationId);
            let i = b.field(gid, 0, KirType::U32);
            let n = b.uniform_field(params, 0, KirType::U32);
            let hi = b.mul_hi_u32(i, n);
            let f = b.bitcast(hi, KirType::F32);
            let s = b.binary(IrBinOp::AddExact, f, f);
            let p2 = b.binary(IrBinOp::MulExact, s, f);
            let bits = b.bitcast(p2, KirType::U32);
            b.store(out, i, bits);
            b.finish();
        });
        let t = emit(&m, None);
        assert!(!t.source.contains("#include"));
        assert!(t.source.contains("__umulhi("));
        assert!(t.source.contains("__fadd_rn("));
        assert!(t.source.contains("__fmul_rn("));
        assert!(t.source.contains("luma_indirect"));
        assert!(t.source.contains("__launch_bounds__(LUMA_WG_INVOCATIONS)"));
        assert!(t.source.contains("#define LUMA_WG_INVOCATIONS 64u"));
        assert_eq!(t.param_abis, vec![CudaArgAbi::InlineUniform { size: 8 }, CudaArgAbi::DevicePtr, CudaArgAbi::DevicePtr]);
        let masked = emit(&m, Some(&[true, true, false]));
        assert_eq!(masked.param_abis.len(), 2);
        assert!(!masked.source.contains("b2"));
    }

    #[test]
    fn struct_padding_and_vec3() {
        let m = build(|mb| {
            let s = mb.add_struct_layout("P", &[("a", KirType::U32), ("v", KirType::vec(ScalarTy::F32, 4))]);
            let s3 = mb.add_struct_layout("Q", &[("v", KirType::vec(ScalarTy::F32, 3)), ("t", KirType::U32)]);
            let u = mb.add_uniform("p", KirType::Struct(s));
            let _q = mb.add_uniform("q", KirType::Struct(s3));
            let out = mb.add_storage_read_write("out", KirType::F32);
            let mut b = mb.entry_builder();
            let v = b.uniform_field(u, 1, KirType::vec(ScalarTy::F32, 4));
            let x = b.field(v, 0, KirType::F32);
            let z = b.lit_u32(0);
            b.store(out, z, x);
            b.finish();
        });
        let t = emit(&m, None).source;
        assert!(t.contains("uint8_t _pad1[12];"), "{t}");
        assert!(t.contains("struct alignas(16) S0"));
        assert!(t.contains("luma_f32x3 f0;"));
        assert!(t.contains("uint32_t f1;"));
        assert!(!t.contains("_pad1[12];\n    luma_f32x3"));
    }

    #[test]
    fn inline_uniform_budget_and_fallback() {
        let m = build(|mb| {
            let arr = |n| KirType::Array { elem: Box::new(KirType::U32), len: n, stride: 4 };
            mb.add_uniform("a", arr(750));
            mb.add_uniform("b", arr(750));
            mb.add_uniform("big", arr(2000));
            mb.add_storage_read_write("out", KirType::U32);
            mb.entry_builder().finish();
        });
        let t = emit(&m, None);
        assert_eq!(
            t.param_abis,
            vec![CudaArgAbi::InlineUniform { size: 3000 }, CudaArgAbi::UniformPtr, CudaArgAbi::UniformPtr, CudaArgAbi::DevicePtr]
        );
        assert!(t.source.contains("const luma_arr<uint32_t, 2000u, 4u>* __restrict__ u2_p"));
        assert!(t.source.contains("const luma_arr<uint32_t, 750u, 4u>& u1 = *u1_p;"));
    }

    #[test]
    fn vector_select_and_array_length() {
        let v4 = KirType::vec(ScalarTy::F32, 4);
        let m = build(|mb| {
            let a = mb.add_storage_read("a", v4.clone());
            let out = mb.add_storage_read_write("out", v4.clone());
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let x = b.load(a, z);
            let y = b.load(a, z);
            let c = b.lt(x, y);
            let s = b.select(c, x, y);
            b.store(out, z, s);
            let n = b.val(Expr::ArrayLength { binding: out }, KirType::U32);
            b.store(out, z, b.splat(ScalarTy::F32, 4, b.convert(n, KirType::F32)));
            b.finish();
        });
        let t = emit(&m, None);
        assert!(t.source.contains("luma_select("));
        assert!(t.source.contains("luma_boolx4"));
        assert!(t.source.contains("luma_f32x4 luma_select(luma_boolx4 a0, luma_f32x4 a1, luma_f32x4 a2)"));
        assert_eq!(t.array_lengths, vec![1]);
        assert!(t.source.contains("uint32_t _luma_len1)"));
        assert_eq!(t.param_abis, vec![CudaArgAbi::DevicePtr, CudaArgAbi::DevicePtr]);
    }

    #[test]
    fn atomic_add_ballot_and_arrays() {
        let m = build(|mb| {
            let acc = mb.add_storage_read_write("acc", KirType::F32);
            let out = mb.add_storage_read_write("out", KirType::U32);
            mb.add_private("arr", KirType::Array { elem: Box::new(KirType::U32), len: 4, stride: 16 }, None);
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let one = b.lit_f32(1.0);
            b.atomic_rmw(None, IrAtomicOp::Add, ScalarTy::F32, AtomicTarget::Storage { binding: acc, index: Some(z.expr().boxed()) }, one);
            b.atomic_rmw(None, IrAtomicOp::Min, ScalarTy::F32, AtomicTarget::Storage { binding: acc, index: Some(z.expr().boxed()) }, one);
            let pred = b.lit_bool(true);
            let ballot = b.val(
                Expr::Subgroup { op: IrSubgroupOp::Ballot, arg: Some(pred.expr().boxed()), lane: None, ty: KirType::vec(ScalarTy::U32, 4) },
                KirType::vec(ScalarTy::U32, 4),
            );
            let w = b.field(ballot, 0, KirType::U32);
            b.store(out, z, w);
            b.finish();
        });
        let t = emit(&m, None).source;
        assert!(t.contains("atomicAdd("));
        assert!(t.contains("luma_atomic_fmin("));
        assert!(t.contains("__ballot_sync("));
        assert!(t.contains("luma_arr<uint32_t, 4u, 16u> pv0 = {};"));
        assert!(t.contains("template<class T, uint32_t N, uint32_t S> struct luma_arr"));
    }

    #[test]
    fn workgroup_init_is_optional() {
        let m = build(|mb| {
            mb.add_shared("tile", KirType::Scalar(ScalarTy::F16), 8);
            mb.add_shared_scalar("flag", KirType::BOOL);
            mb.add_storage_read_write("out", KirType::U32);
            let mut b = mb.entry_builder();
            b.barrier(BarrierScope::Workgroup);
            b.finish();
        });
        let on = emit(&m, None);
        assert!(on.source.contains("z < sizeof(sh0)"));
        assert!(on.source.contains("z < sizeof(sh1)"));
        let off = emit_with_workgroup_init(&m, None, false);
        assert!(!off.source.contains("sizeof(sh"));
        assert_eq!(on.param_abis, off.param_abis);
        assert_eq!(on.array_lengths, off.array_lengths);
    }

    #[test]
    fn aggregate_compose_uses_constructors() {
        let m = build(|mb| {
            let s = mb.add_struct_layout("S", &[("a", KirType::U32), ("v", KirType::vec(ScalarTy::F32, 4))]);
            let out = mb.add_storage_read_write("out", KirType::U32);
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let v = b.compose(KirType::vec(ScalarTy::F32, 4), &[b.lit_f32(1.0), b.lit_f32(2.0), b.lit_f32(3.0), b.lit_f32(4.0)]);
            let st = b.compose(KirType::Struct(s), &[z, v]);
            let arr_ty = KirType::Array { elem: Box::new(KirType::U32), len: 2, stride: 8 };
            let arr = b.compose(arr_ty, &[z, z]);
            let e = b.index(arr, z, KirType::U32);
            let a = b.field(st, 0, KirType::U32);
            b.store(out, z, a + e);
            b.finish();
        });
        let t = emit(&m, None).source;
        // Struct: named-field constructor that never touches the `_pad` members.
        assert!(t.contains("luma_mk_S0(uint32_t f0, luma_f32x4 f1)"), "{t}");
        assert!(t.contains("r.f0 = f0; r.f1 = f1;"));
        assert!(!t.contains("S0{"));
        assert!(t.contains("luma_mk_S0("));
        // The vector inside the struct is a plain brace-initialised vector, not a luma_comp.
        assert!(t.contains("luma_f32x4{1.0f, 2.0f, 3.0f, 4.0f}"));
        assert!(!t.contains("luma_comp<"));
        // Array: element-wise assignment through the wrapper.
        assert!(t.contains("luma_arr_of<luma_arr<uint32_t, 2u, 8u>>("));
        assert!(t.contains("A luma_arr_of(E... e)"));
    }

    #[test]
    fn float_atomics_use_bit_identity() {
        let m = build(|mb| {
            let acc = mb.add_storage_read_write("acc", KirType::F32);
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let one = b.lit_f32(1.0);
            let target = || AtomicTarget::Storage { binding: acc, index: Some(z.expr().boxed()) };
            b.atomic_rmw(None, IrAtomicOp::Xor, ScalarTy::F32, target(), one);
            b.atomic_rmw(None, IrAtomicOp::And, ScalarTy::F32, target(), one);
            b.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::F32, target(), one);
            let old = b.var_decl("old", KirType::F32, None);
            let swapped = b.var_decl("swapped", KirType::BOOL, None);
            b.push_stmt(Stmt::AtomicCompareExchange {
                dst_old: old,
                dst_exchanged: swapped,
                elem_ty: ScalarTy::F32,
                target: target(),
                cmp: one.expr(),
                val: one.expr(),
            });
            b.finish();
        });
        let t = emit(&m, None).source;
        assert!(t.contains("__uint_as_float(atomicXor((uint32_t*)("));
        assert!(t.contains("__uint_as_float(atomicAnd((uint32_t*)("));
        assert!(t.contains("__uint_as_float(atomicOr((uint32_t*)("));
        assert!(t.contains("uint32_t c = __float_as_uint(1.0f); uint32_t n = __float_as_uint(1.0f);"), "{t}");
        assert!(t.contains("= __uint_as_float(r);"));
        assert!(t.contains("(r == c)"));
        assert!(!t.contains("unsupported"));
    }

    #[test]
    fn warp_reductions_are_butterflies() {
        let m = build(|mb| {
            let out = mb.add_storage_read_write("out", KirType::F32);
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let x = b.lit_f32(1.5);
            let sum = b.val(Expr::Subgroup { op: IrSubgroupOp::Add, arg: Some(x.expr().boxed()), lane: None, ty: KirType::F32 }, KirType::F32);
            let ex = b.val(Expr::Subgroup { op: IrSubgroupOp::ExclusiveAdd, arg: Some(x.expr().boxed()), lane: None, ty: KirType::F32 }, KirType::F32);
            b.store(out, z, sum + ex);
            b.finish();
        });
        let t = emit(&m, None).source;
        assert!(t.contains("luma_sg_add(1.5f)"));
        assert!(t.contains("luma_sg_xadd(1.5f)"));
        assert!(t.contains("v = luma_sg_op<OP>(v, __shfl_xor_sync(m, v, o));"));
        assert!(t.contains("__shfl_up_sync(m, s, o)"));
        assert!(t.contains("(int32_t)((uint32_t)a + (uint32_t)b)"));
    }

    #[test]
    fn shared_clamp_only_for_untrusted() {
        let m = build(|mb| {
            let tile = mb.add_shared("tile", KirType::U32, 8);
            mb.add_storage_read_write("out", KirType::U32);
            let mut b = mb.entry_builder();
            let z = b.lit_u32(0);
            let one = b.lit_u32(1);
            b.shared_store(tile, z, one);
            let v = b.shared_load(tile, z);
            b.store(0, z, v);
            b.finish();
        });
        let untrusted = emit(&m, None).source;
        assert!(untrusted.contains("sh0[luma_ci(0u, 8u)]"));
        assert!(untrusted.contains("z < sizeof(sh0)"));
        let trusted = emit_with_workgroup_init(&m, None, false).source;
        assert!(trusted.contains("sh0[0u]"));
        assert!(!trusted.contains("luma_ci("));
        assert!(!trusted.contains("sizeof(sh0)"));
    }

    #[test]
    fn uniform_inline_limit_edges() {
        let arr = |n| KirType::Array { elem: Box::new(KirType::U32), len: n, stride: 4 };
        let at_limit = build(|mb| {
            mb.add_uniform("u", arr(896));
            mb.add_storage_read_write("out", KirType::U32);
            mb.entry_builder().finish();
        });
        assert_eq!(emit(&at_limit, None).param_abis, vec![CudaArgAbi::InlineUniform { size: 3584 }, CudaArgAbi::DevicePtr]);
        let over = build(|mb| {
            mb.add_uniform("u", arr(900));
            mb.add_storage_read_write("out", KirType::U32);
            mb.entry_builder().finish();
        });
        assert_eq!(emit(&over, None).param_abis, vec![CudaArgAbi::UniformPtr, CudaArgAbi::DevicePtr]);
        // Alignment padding counts: an align-16 uniform after an 8-byte pointer pads to 32.
        let aligned = build(|mb| {
            let s = mb.add_struct_layout("A", &[("v", KirType::vec(ScalarTy::F32, 4)), ("pad", arr(892))]);
            mb.add_storage_read_write("out", KirType::U32);
            mb.add_uniform("u", KirType::Struct(s));
            mb.entry_builder().finish();
        });
        let t = emit(&aligned, None);
        assert_eq!(t.param_abis, vec![CudaArgAbi::DevicePtr, CudaArgAbi::InlineUniform { size: 3584 }]);
        assert!(t.source.contains("const S0 u1"));
    }
}
