//! The Kernel IR: a small imperative, backend-neutral description of one compute kernel.
//!
//! A [`KernelModule`] is a tree of [`Stmt`]s over [`Expr`]essions with explicit locals; there is no
//! SSA and no pointer type. Memory is reached by (binding | shared | private) plus an access
//! path, which every target (WGSL, MSL, CUDA C++) can print directly.

use crate::hal::{BindingInfo, BindingKind};

/// Index into the locals table of the enclosing function (or of `entry_locals`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// Index into [`KernelModule::functions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarTy {
    Bool,
    U32,
    I32,
    F16,
    F32,
    F64,
}

impl ScalarTy {
    /// Size in memory; `Bool` is not host-shareable and counts as a 4-byte word.
    pub const fn size_bytes(self) -> u32 {
        match self {
            ScalarTy::Bool | ScalarTy::U32 | ScalarTy::I32 | ScalarTy::F32 => 4,
            ScalarTy::F16 => 2,
            ScalarTy::F64 => 8,
        }
    }

    pub const fn align_bytes(self) -> u32 {
        self.size_bytes()
    }

    pub const fn is_float(self) -> bool {
        matches!(self, ScalarTy::F16 | ScalarTy::F32 | ScalarTy::F64)
    }

    pub const fn is_int(self) -> bool {
        matches!(self, ScalarTy::U32 | ScalarTy::I32)
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, ScalarTy::I32 | ScalarTy::F16 | ScalarTy::F32 | ScalarTy::F64)
    }

    pub const fn wgsl_name(self) -> &'static str {
        match self {
            ScalarTy::Bool => "bool",
            ScalarTy::U32 => "u32",
            ScalarTy::I32 => "i32",
            ScalarTy::F16 => "f16",
            ScalarTy::F32 => "f32",
            ScalarTy::F64 => "f64",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KirType {
    Scalar(ScalarTy),
    Vec { elem: ScalarTy, lanes: u8 }, // 2, 3, 4
    Array { elem: Box<KirType>, len: u32, stride: u32 },
    /// Index into [`KernelModule::structs`].
    Struct(u32),
    Atomic(ScalarTy), // U32 or I32
}

impl KirType {
    pub const BOOL: KirType = KirType::Scalar(ScalarTy::Bool);
    pub const U32: KirType = KirType::Scalar(ScalarTy::U32);
    pub const I32: KirType = KirType::Scalar(ScalarTy::I32);
    pub const F32: KirType = KirType::Scalar(ScalarTy::F32);

    pub const fn vec(elem: ScalarTy, lanes: u8) -> KirType {
        KirType::Vec { elem, lanes }
    }

    /// The scalar type of a `Scalar`.
    pub fn scalar(&self) -> Option<ScalarTy> {
        match self {
            KirType::Scalar(s) => Some(*s),
            _ => None,
        }
    }

    /// The scalar a scalar, vector or atomic is made of.
    pub fn elem_scalar(&self) -> Option<ScalarTy> {
        match self {
            KirType::Scalar(s) | KirType::Atomic(s) => Some(*s),
            KirType::Vec { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    /// 2 to 4 for vectors, 1 for everything else.
    pub fn lanes(&self) -> u8 {
        match self {
            KirType::Vec { lanes, .. } => *lanes,
            _ => 1,
        }
    }

    /// `(size, align)` in bytes under WGSL's host-shareable layout.
    pub fn layout(&self, structs: &[StructDecl]) -> (u32, u32) {
        match self {
            KirType::Scalar(s) | KirType::Atomic(s) => (s.size_bytes(), s.align_bytes()),
            KirType::Vec { elem, lanes } => {
                let e = elem.size_bytes();
                (e * u32::from(*lanes), e * if *lanes == 2 { 2 } else { 4 })
            }
            KirType::Array { elem, len, stride } => (len * stride, elem.layout(structs).1),
            KirType::Struct(i) => {
                let s = &structs[*i as usize];
                (s.size, s.align)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub ty: KirType,
    pub offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructField>,
    /// Byte size including trailing padding.
    pub size: u32,
    pub align: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingDecl {
    pub name: String,
    pub kind: BindingKind,
    /// For an unsized array binding the element type, otherwise the type of the whole binding.
    pub elem_ty: KirType,
    /// `var<storage> b: array<T>`: indexed by [`Expr::Load`] / [`Stmt::Store`], length unknown.
    pub is_unsized_array: bool,
    pub element_stride: Option<u32>,
    pub min_size: Option<u32>,
}

impl BindingDecl {
    pub fn info(&self) -> BindingInfo {
        BindingInfo {
            name: self.name.clone(),
            kind: self.kind,
            element_stride: self.element_stride,
            min_size: self.min_size,
        }
    }
}

/// Workgroup memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedDecl {
    pub name: String,
    pub elem_ty: KirType,
    /// `Some(n)` for `array<elem_ty, n>`, `None` for a single `elem_ty`.
    pub len: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrivateDecl {
    pub name: String,
    pub ty: KirType,
    pub init: Option<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinVar {
    GlobalInvocationId,
    LocalInvocationId,
    LocalInvocationIndex,
    WorkgroupId,
    NumWorkgroups,
    SubgroupInvocationId,
    SubgroupSize,
}

impl BuiltinVar {
    pub fn ty(self) -> KirType {
        match self {
            BuiltinVar::GlobalInvocationId | BuiltinVar::LocalInvocationId | BuiltinVar::WorkgroupId | BuiltinVar::NumWorkgroups => {
                KirType::vec(ScalarTy::U32, 3)
            }
            BuiltinVar::LocalInvocationIndex | BuiltinVar::SubgroupInvocationId | BuiltinVar::SubgroupSize => KirType::U32,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KirFunction {
    pub name: String,
    pub params: Vec<(String, KirType)>,
    pub ret_ty: Option<KirType>,
    /// `(name, type, mutable)`; indexed by [`LocalId`].
    pub locals: Vec<(String, KirType, bool)>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelModule {
    pub label: String,
    pub entry: String,
    pub workgroup_size: [u32; 3],
    /// Set when the kernel relies on IEEE-exact `f32` arithmetic (error-free transformations):
    /// targets must not reassociate or contract, i.e. no fast-math.
    pub uses_exact_f32: bool,
    pub structs: Vec<StructDecl>,
    /// Indexed by binding number (the kernel argument position).
    pub bindings: Vec<BindingDecl>,
    pub shared: Vec<SharedDecl>,
    pub privates: Vec<PrivateDecl>,
    pub functions: Vec<KirFunction>,
    /// Builtin inputs of the entry point and the locals that hold them.
    pub entry_builtins: Vec<(BuiltinVar, LocalId)>,
    /// `(name, type, mutable)`; indexed by [`LocalId`], includes the builtin locals.
    pub entry_locals: Vec<(String, KirType, bool)>,
    pub entry_body: Vec<Stmt>,
}

/// One step of an access path. On a vector, `Field(i)` selects lane `i`.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessStep {
    Field(u32),
    Index(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum AtomicTarget {
    /// `binding[index]` (or the binding itself when `index` is `None`).
    Storage { binding: u32, index: Option<Box<Expr>> },
    /// `shared[index]` (or the variable itself when `index` is `None`).
    Shared { shared: u32, index: Option<Box<Expr>> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierScope {
    Workgroup,
    Storage,
    StorageAndWorkgroup,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LValue {
    Local(LocalId),
    Private(u32),
    Storage { binding: u32, path: Vec<AccessStep> },
    Shared { shared: u32, path: Vec<AccessStep> },
    Index { base: Box<LValue>, index: Box<Expr> },
    /// A struct field, or a vector lane.
    Field { base: Box<LValue>, field: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchCase {
    I32(i32),
    U32(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrAtomicOp {
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// Declares an immutable local.
    Let { local: LocalId, init: Expr },
    /// Declares a mutable local; `None` is zero-initialised.
    Var { local: LocalId, init: Option<Expr> },
    Assign { dst: LValue, val: Expr },
    /// `binding[index] = val` on a storage binding.
    Store { binding: u32, index: Expr, val: Expr },
    /// `shared[index] = val` on a workgroup array.
    SharedStore { shared: u32, index: Expr, val: Expr },
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    Loop { body: Vec<Stmt>, continuing: Vec<Stmt>, break_if: Option<Expr> },
    /// `(value, body, fall_through)` per case.
    Switch { selector: Expr, cases: Vec<(SwitchCase, Vec<Stmt>, bool)>, default: Vec<Stmt> },
    Break,
    Continue,
    Return(Option<Expr>),
    Barrier(BarrierScope),
    AtomicStore { target: AtomicTarget, val: Expr },
    /// `dst` receives the previous value when present.
    AtomicRmw { dst: Option<LocalId>, op: IrAtomicOp, elem_ty: ScalarTy, target: AtomicTarget, val: Expr },
    /// `dst_old` receives the previous value, `dst_exchanged` whether the swap happened.
    AtomicCompareExchange { dst_old: LocalId, dst_exchanged: LocalId, elem_ty: ScalarTy, target: AtomicTarget, cmp: Expr, val: Expr },
    /// Evaluates the expression for its effects and drops the value (a call of a `void` function).
    Discard(Expr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrUnOp {
    Neg,
    Not,
    BitNot,
    Abs,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sign,
    Floor,
    Ceil,
    Trunc,
    Round,
    CountOneBits,
    FirstTrailingBit,
    IsNan,
    IsInf,
    /// Identity the compiler must not see through (`luma_o`): keeps error-free transformations
    /// from being simplified away.
    OpaqueGuard,
    // Beyond the original set: needed to lower arbitrary user WGSL.
    InverseSqrt,
    Exp2,
    Log2,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Log10,
    Cbrt,
    /// The error function.
    Erf,
    /// `1 - erf(x)`, accurate for large `x`.
    Erfc,
    /// `ln |gamma(x)|`.
    Lgamma,
    /// The gamma function.
    Tgamma,
    /// `ln(1 + x)`, accurate for small `x`.
    Log1p,
    /// `e^x - 1`, accurate for small `x`.
    Expm1,
    FirstLeadingBit,
    CountLeadingZeros,
    CountTrailingZeros,
    /// `all(vec<bool>)`.
    All,
    /// `any(vec<bool>)`.
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `f32` add/sub/mul that must be correctly rounded and never contracted into an `fma`.
    AddExact,
    SubExact,
    MulExact,
    Min,
    Max,
    Pow,
    Atan2,
    Copysign,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Rotl,
    /// High 32 bits of the 64-bit product of two `u32`.
    MulHiU32,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Logical (short-circuit free) `&&` / `||` on bools.
    And,
    Or,
}

impl IrBinOp {
    pub const fn is_comparison(self) -> bool {
        matches!(self, IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge)
    }
}

/// For `Select`: `a = cond`, `b = then_val`, `c = else_val`, i.e. `cond ? b : c`
/// (WGSL's `select(c, b, a)` has the operands the other way round).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrTernaryOp {
    Fma,
    Clamp,
    Select,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastKind {
    /// Value-preserving conversion to the destination scalar type.
    Convert,
    /// Reinterpret the bits.
    Bitcast,
    /// Float to integer: saturating, NaN to 0.
    FloatToIntSat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrSubgroupOp {
    /// `arg` is the predicate (`None` is `true`); yields `vec4<u32>` of ballot bits.
    Ballot,
    Add,
    Min,
    Max,
    ExclusiveAdd,
    /// Value of `arg` in lane `lane`; `lane == None` is the first active lane (broadcast-first).
    Shuffle,
}

/// A read of a texture binding ([`Expr::Texture`]); the texel is a `vec4<f32>` whose channels the
/// texture's format does not have read as 0, 0, 1 (red, green, blue) and 1 (alpha), and the
/// addressing is clamp-to-edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureOp {
    /// `textureLoad(t, coords, 0)`, `coords: vec2<i32>`: the texel at integer coordinates (clamped
    /// to the texture).
    Load,
    /// `textureGather(component, t, s, coords)`, `coords: vec2<f32>` normalized: the `component`
    /// (0 red .. 3 alpha) of the four texels `(i, j)`, `(i + 1, j)`, `(i + 1, j + 1)`, `(i, j + 1)` in
    /// that order (x, y, z, w) with `i = floor(coords.x * width - 0.5)` and `j` likewise,
    /// unfiltered and exact; texels past the edge are the clamped ones.
    Gather(u8),
    /// `textureSampleLevel(t, s, coords, 0.0)`, `coords: vec2<f32>` normalized: the texture's
    /// filter ([`crate::hal::TextureFilter`]) at the point.
    Sample,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    LitBool(bool),
    LitU32(u32),
    LitI32(i32),
    /// An `f16` literal, held as the `f32` of the same value.
    LitF16(f32),
    LitF32(f32),
    LitF64(f64),
    ZeroVal(KirType),
    Local(LocalId),
    FnParam(u32),
    Private(u32),
    /// Component 0, 1 or 2 of the launch grid size (`luma.size`).
    GridSize(u8),
    /// A read of a uniform binding at `path`, of type `ty`.
    UniformRead { binding: u32, path: Vec<AccessStep>, ty: KirType },
    /// `binding[index]` of an unsized array binding.
    Load { binding: u32, index: Box<Expr>, ty: KirType },
    /// A storage read that is not a plain `binding[index]`; the path starts at the binding value.
    StorageRead { binding: u32, path: Vec<AccessStep>, ty: KirType },
    SharedLoad { shared: u32, index: Box<Expr>, ty: KirType },
    SharedRead { shared: u32, path: Vec<AccessStep>, ty: KirType },
    Access { base: Box<Expr>, step: AccessStep, ty: KirType },
    Swizzle { base: Box<Expr>, indices: [u8; 4], len: u8, ty: KirType },
    /// Components are scalars, or smaller vectors whose lanes add up to the vector's.
    Compose { ty: KirType, comps: Vec<Expr> },
    Splat { elem: ScalarTy, lanes: u8, val: Box<Expr> },
    Unary { op: IrUnOp, arg: Box<Expr>, ty: KirType },
    Binary { op: IrBinOp, lhs: Box<Expr>, rhs: Box<Expr>, ty: KirType },
    Ternary { op: IrTernaryOp, a: Box<Expr>, b: Box<Expr>, c: Box<Expr>, ty: KirType },
    Cast { kind: CastKind, expr: Box<Expr>, src_ty: KirType, dst_ty: KirType },
    Call { func: FnId, args: Vec<Expr>, ty: Option<KirType> },
    AtomicLoad { target: AtomicTarget, ty: ScalarTy },
    Subgroup { op: IrSubgroupOp, arg: Option<Box<Expr>>, lane: Option<Box<Expr>>, ty: KirType },
    /// Element count of an unsized array binding (`arrayLength`); only backends that know the
    /// buffer length can honor it.
    ArrayLength { binding: u32 },
    /// A fetch from a [`BindingKind::Texture2d`] binding; a `vec4<f32>`.
    Texture { op: TextureOp, binding: u32, coords: Box<Expr> },
}

impl Expr {
    pub fn boxed(self) -> Box<Expr> {
        Box::new(self)
    }

    /// Calls `f` on each direct child expression.
    pub fn for_each_child(&self, f: &mut dyn FnMut(&Expr)) {
        fn path(p: &[AccessStep], f: &mut dyn FnMut(&Expr)) {
            for s in p {
                if let AccessStep::Index(e) = s {
                    f(e);
                }
            }
        }
        match self {
            Expr::UniformRead { path: p, .. } | Expr::StorageRead { path: p, .. } | Expr::SharedRead { path: p, .. } => path(p, f),
            Expr::Load { index, .. } | Expr::SharedLoad { index, .. } => f(index),
            Expr::Access { base, step, .. } => {
                f(base);
                if let AccessStep::Index(e) = step {
                    f(e);
                }
            }
            Expr::Swizzle { base, .. } => f(base),
            Expr::Compose { comps, .. } => comps.iter().for_each(f),
            Expr::Splat { val, .. } => f(val),
            Expr::Unary { arg, .. } => f(arg),
            Expr::Binary { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            Expr::Ternary { a, b, c, .. } => {
                f(a);
                f(b);
                f(c);
            }
            Expr::Cast { expr, .. } => f(expr),
            Expr::Texture { coords, .. } => f(coords),
            Expr::Call { args, .. } => args.iter().for_each(f),
            Expr::AtomicLoad { target, .. } => target.for_each_expr(f),
            Expr::Subgroup { arg, lane, .. } => {
                if let Some(a) = arg {
                    f(a);
                }
                if let Some(l) = lane {
                    f(l);
                }
            }
            Expr::LitBool(_)
            | Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF16(_)
            | Expr::LitF32(_)
            | Expr::LitF64(_)
            | Expr::ZeroVal(_)
            | Expr::Local(_)
            | Expr::FnParam(_)
            | Expr::Private(_)
            | Expr::GridSize(_)
            | Expr::ArrayLength { .. } => {}
        }
    }

    /// Calls `f` on each direct child expression, mutably.
    pub fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        fn path(p: &mut [AccessStep], f: &mut dyn FnMut(&mut Expr)) {
            for s in p {
                if let AccessStep::Index(e) = s {
                    f(e);
                }
            }
        }
        match self {
            Expr::UniformRead { path: p, .. } | Expr::StorageRead { path: p, .. } | Expr::SharedRead { path: p, .. } => path(p, f),
            Expr::Load { index, .. } | Expr::SharedLoad { index, .. } => f(index),
            Expr::Access { base, step, .. } => {
                f(base);
                if let AccessStep::Index(e) = step {
                    f(e);
                }
            }
            Expr::Swizzle { base, .. } => f(base),
            Expr::Compose { comps, .. } => comps.iter_mut().for_each(f),
            Expr::Splat { val, .. } => f(val),
            Expr::Unary { arg, .. } => f(arg),
            Expr::Binary { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            Expr::Ternary { a, b, c, .. } => {
                f(a);
                f(b);
                f(c);
            }
            Expr::Cast { expr, .. } => f(expr),
            Expr::Texture { coords, .. } => f(coords),
            Expr::Call { args, .. } => args.iter_mut().for_each(f),
            Expr::AtomicLoad { target, .. } => target.for_each_expr_mut(f),
            Expr::Subgroup { arg, lane, .. } => {
                if let Some(a) = arg {
                    f(a);
                }
                if let Some(l) = lane {
                    f(l);
                }
            }
            Expr::LitBool(_)
            | Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF16(_)
            | Expr::LitF32(_)
            | Expr::LitF64(_)
            | Expr::ZeroVal(_)
            | Expr::Local(_)
            | Expr::FnParam(_)
            | Expr::Private(_)
            | Expr::GridSize(_)
            | Expr::ArrayLength { .. } => {}
        }
    }

    /// Pre-order walk of this expression and all its descendants.
    pub fn walk(&self, f: &mut dyn FnMut(&Expr)) {
        f(self);
        self.for_each_child(&mut |c| c.walk(f));
    }

    /// Post-order mutable walk: children are rewritten before their parent.
    pub fn walk_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        self.for_each_child_mut(&mut |c| c.walk_mut(f));
        f(self);
    }
}

impl AtomicTarget {
    pub fn for_each_expr(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            AtomicTarget::Storage { index: Some(e), .. } | AtomicTarget::Shared { index: Some(e), .. } => f(e),
            _ => {}
        }
    }

    pub fn for_each_expr_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            AtomicTarget::Storage { index: Some(e), .. } | AtomicTarget::Shared { index: Some(e), .. } => f(e),
            _ => {}
        }
    }
}

impl LValue {
    fn for_each_expr(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            LValue::Local(_) | LValue::Private(_) => {}
            LValue::Storage { path, .. } | LValue::Shared { path, .. } => {
                for s in path {
                    if let AccessStep::Index(e) = s {
                        f(e);
                    }
                }
            }
            LValue::Index { base, index } => {
                base.for_each_expr(f);
                f(index);
            }
            LValue::Field { base, .. } => base.for_each_expr(f),
        }
    }

    fn for_each_expr_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            LValue::Local(_) | LValue::Private(_) => {}
            LValue::Storage { path, .. } | LValue::Shared { path, .. } => {
                for s in path {
                    if let AccessStep::Index(e) = s {
                        f(e);
                    }
                }
            }
            LValue::Index { base, index } => {
                base.for_each_expr_mut(f);
                f(index);
            }
            LValue::Field { base, .. } => base.for_each_expr_mut(f),
        }
    }

    /// The storage binding this place is rooted in, if any.
    pub fn storage_binding(&self) -> Option<u32> {
        match self {
            LValue::Storage { binding, .. } => Some(*binding),
            LValue::Index { base, .. } | LValue::Field { base, .. } => base.storage_binding(),
            _ => None,
        }
    }
}

impl Stmt {
    /// Calls `f` on each expression this statement holds directly (not those of nested blocks).
    pub fn for_each_expr(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            Stmt::Let { init, .. } => f(init),
            Stmt::Var { init, .. } => {
                if let Some(e) = init {
                    f(e);
                }
            }
            Stmt::Assign { dst, val } => {
                dst.for_each_expr(f);
                f(val);
            }
            Stmt::Store { index, val, .. } | Stmt::SharedStore { index, val, .. } => {
                f(index);
                f(val);
            }
            Stmt::If { cond, .. } => f(cond),
            Stmt::Loop { break_if, .. } => {
                if let Some(e) = break_if {
                    f(e);
                }
            }
            Stmt::Switch { selector, .. } => f(selector),
            Stmt::Return(Some(e)) | Stmt::Discard(e) => f(e),
            Stmt::AtomicStore { target, val } | Stmt::AtomicRmw { target, val, .. } => {
                target.for_each_expr(f);
                f(val);
            }
            Stmt::AtomicCompareExchange { target, cmp, val, .. } => {
                target.for_each_expr(f);
                f(cmp);
                f(val);
            }
            Stmt::Break | Stmt::Continue | Stmt::Return(None) | Stmt::Barrier(_) => {}
        }
    }

    pub fn for_each_expr_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            Stmt::Let { init, .. } => f(init),
            Stmt::Var { init, .. } => {
                if let Some(e) = init {
                    f(e);
                }
            }
            Stmt::Assign { dst, val } => {
                dst.for_each_expr_mut(f);
                f(val);
            }
            Stmt::Store { index, val, .. } | Stmt::SharedStore { index, val, .. } => {
                f(index);
                f(val);
            }
            Stmt::If { cond, .. } => f(cond),
            Stmt::Loop { break_if, .. } => {
                if let Some(e) = break_if {
                    f(e);
                }
            }
            Stmt::Switch { selector, .. } => f(selector),
            Stmt::Return(Some(e)) | Stmt::Discard(e) => f(e),
            Stmt::AtomicStore { target, val } | Stmt::AtomicRmw { target, val, .. } => {
                target.for_each_expr_mut(f);
                f(val);
            }
            Stmt::AtomicCompareExchange { target, cmp, val, .. } => {
                target.for_each_expr_mut(f);
                f(cmp);
                f(val);
            }
            Stmt::Break | Stmt::Continue | Stmt::Return(None) | Stmt::Barrier(_) => {}
        }
    }

    /// Calls `f` on each nested block.
    pub fn for_each_block(&self, f: &mut dyn FnMut(&[Stmt])) {
        match self {
            Stmt::If { then_body, else_body, .. } => {
                f(then_body);
                f(else_body);
            }
            Stmt::Loop { body, continuing, .. } => {
                f(body);
                f(continuing);
            }
            Stmt::Switch { cases, default, .. } => {
                for (_, body, _) in cases {
                    f(body);
                }
                f(default);
            }
            _ => {}
        }
    }

    pub fn for_each_block_mut(&mut self, f: &mut dyn FnMut(&mut Vec<Stmt>)) {
        match self {
            Stmt::If { then_body, else_body, .. } => {
                f(then_body);
                f(else_body);
            }
            Stmt::Loop { body, continuing, .. } => {
                f(body);
                f(continuing);
            }
            Stmt::Switch { cases, default, .. } => {
                for (_, body, _) in cases {
                    f(body);
                }
                f(default);
            }
            _ => {}
        }
    }
}

/// Pre-order walk of every statement of `block`, nested ones included.
pub fn walk_stmts(block: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in block {
        f(s);
        s.for_each_block(&mut |b| walk_stmts(b, f));
    }
}

/// Calls `f` post-order on every expression node reachable from `block`, mutably.
pub fn rewrite_exprs(block: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
    for s in block {
        s.for_each_expr_mut(&mut |e| e.walk_mut(f));
        s.for_each_block_mut(&mut |b| rewrite_exprs(b, f));
    }
}

impl KernelModule {
    /// Which of [`functions`](Self::functions) the entry point calls, directly or transitively.
    pub fn reachable_functions(&self) -> Vec<bool> {
        fn calls(block: &[Stmt], work: &mut Vec<usize>) {
            walk_stmts(block, &mut |s| {
                s.for_each_expr(&mut |e| {
                    e.walk(&mut |x| {
                        if let Expr::Call { func, .. } = x {
                            work.push(func.0 as usize);
                        }
                    })
                })
            });
        }
        let mut live = vec![false; self.functions.len()];
        let mut work = Vec::new();
        calls(&self.entry_body, &mut work);
        while let Some(f) = work.pop() {
            if let Some(slot) = live.get_mut(f)
                && !*slot
            {
                *slot = true;
                calls(&self.functions[f].body, &mut work);
            }
        }
        live
    }
}
