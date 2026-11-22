//! Fused kernels built directly as Kernel IR (see `luma_core::kir`), without printing source.
//!
//! The kernels are the ones [`codegen`](super::codegen) documents: the bindings are the outputs
//! (`out0`, `out1`, ...), the input buffers (`in0`, ... and `in{k}_{s}` for the segments of the
//! segmented variant), then, if the program reads scalars, the uniform `p`; a binding's index is
//! its position in this order, which is the position of the argument of the launch
//! (`launch::params`, `launch::pin_inputs`).
//!
//! [`Lower`] is the instruction layer: the elementwise operations, the casts, the views and the
//! shifts of a program as [`Val`]s of a [`FnBuilder`]. Everything that reads the uniform `p` (the
//! functions `view{n}`, `pview{id}`, `ld{k}`, `sub{k}`) is built natively as a helper function of
//! the module; the pure helpers that are easier to keep as WGSL (the double-float library, the
//! generators, the edge functions, a few numeric series) come from one [`Library`] per kernel,
//! lowered once per distinct text.

use std::cell::RefCell;
use std::sync::Arc;

use luma_core::kir::{
    AtomicTarget, BarrierScope, BuiltinVar, Expr, FnBuilder, FnId, IrAtomicOp, IrBinOp, IrSubgroupOp, IrUnOp, KernelModule,
    KirType, LocalId, ModuleBuilder, ScalarTy, Val,
};

use crate::builtin::{BinOp, Class, CmpOp, DF_MATH_LIBRARY, UnOp, df_library};
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu::lower::{Library, finish, library};
use crate::tuning::Tuning;

use super::codegen::{EDGE_HELPERS, Tile, direct, generates, generator_helpers, reads_scalars, stack_stages, views_of, views_of_inlined};
use super::inline::{Inlined, Lowered, Region};
use super::params::{self, DivRemSpec, StepSpec, ViewSpec};
use super::program::{AtomicOp, FusedKey, Instr, Ref, Tail, calls_df_library, calls_df_math};
use super::staging::Staging;
use super::stencil::WORKGROUP;
use super::{Edge, ElemTy, Gen, Op, OpKind, ShiftKind, ViewMap};

/// The quotient of an index below `2^23` by a divisor of the uniform: one `f32` product and a
/// correction of one, with the conversions between integer and `f32` made by adding `2^23`
/// (see `params::divisor_words`). The wide division is built natively, with `MulHiU32`.
const NARROW_DIV: &str = "fn luma_udiv(n: u32, d: u32, r: u32) -> u32 {
    let f = bitcast<f32>(n | 0x4B000000u) - 8388608.0;
    let q = bitcast<u32>(f * bitcast<f32>(r) + 8388608.0) & 0x7FFFFFu;
    return q - select(0u, 1u, i32(n - q * d) < 0);
}
";

/// The IR type of a register holding `ty`: a `bool` is a `u32` of 0 or 1 there.
pub(super) fn reg(ty: ElemTy) -> KirType {
    match ty {
        ElemTy::F32 => KirType::F32,
        ElemTy::U32 | ElemTy::Bool => KirType::U32,
        ElemTy::I32 => KirType::I32,
        ElemTy::F64 => KirType::Scalar(ScalarTy::F64),
        ElemTy::Df32 => KirType::vec(ScalarTy::F32, 2),
    }
}


/// The element type of a read-write storage buffer holding elements of `ty`: the bits of a
/// `bool` array are written by atomic operations, because the elements of a word are written by
/// different invocations.
pub(super) fn write_elem(ty: ElemTy) -> KirType {
    match ty {
        ElemTy::Bool => KirType::Atomic(ScalarTy::U32),
        other => reg(other),
    }
}

pub(super) fn lu(fb: &FnBuilder<'_>, v: u32) -> Val {
    fb.lit_u32(v)
}

pub(super) fn li(fb: &FnBuilder<'_>, v: i32) -> Val {
    fb.lit_i32(v)
}

fn lf(fb: &FnBuilder<'_>, v: f32) -> Val {
    fb.lit_f32(v)
}

fn storage_target(binding: u32, index: Val) -> AtomicTarget {
    AtomicTarget::Storage { binding, index: Some(Box::new(index.expr())) }
}

fn shared_target(shared: u32, index: Val) -> AtomicTarget {
    AtomicTarget::Shared { shared, index: Some(Box::new(index.expr())) }
}

/// The ballot word of the first 32 invocations of the subgroup whose `cond` holds.
fn ballot(fb: &FnBuilder<'_>, cond: Val) -> Val {
    let ty = KirType::vec(ScalarTy::U32, 4);
    let all = fb.val(Expr::Subgroup { op: IrSubgroupOp::Ballot, arg: Some(Box::new(cond.expr())), lane: None, ty: ty.clone() }, ty);
    fb.field(all, 0, KirType::U32)
}

/// `gid.x + gid.y * nwg.x * wg`: the element this invocation computes.
pub(super) fn element_index(fb: &mut FnBuilder<'_>, wg: u32) -> Val {
    let gid = fb.builtin(BuiltinVar::GlobalInvocationId);
    let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
    let (gx, gy, nx) = (fb.field(gid, 0, KirType::U32), fb.field(gid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32));
    let i = gx + gy * nx * lu(fb, wg);
    fb.let_val("i", i)
}

/// A value that can be named twice without being computed twice.
fn bind(fb: &mut FnBuilder<'_>, name: &str, v: Val) -> Val {
    if matches!(v.expr(), Expr::Local(_) | Expr::LitU32(_) | Expr::LitI32(_) | Expr::FnParam(_)) { v } else { fb.let_val(name, v) }
}

/// Sum of `terms`, 0 if there are none.
fn sum(fb: &FnBuilder<'_>, terms: Vec<Val>) -> Val {
    let mut terms = terms.into_iter();
    match terms.next() {
        Some(first) => terms.fold(first, |total, term| total + term),
        None => fb.lit_u32(0),
    }
}

/// Runs `body(b, e)` for each cell `e` assigned to invocation `lid` of a `WORKGROUP`-sized group.
fn for_cells(fb: &mut FnBuilder<'_>, lid: Val, cells: u32, body: impl FnOnce(&mut FnBuilder<'_>, Val)) {
    if cells == WORKGROUP {
        let e = fb.let_val("e", lid);
        body(fb, e);
    } else if cells < WORKGROUP {
        let ok = fb.lt(lid, lu(fb, cells));
        fb.if_then(ok, |b| {
            let e = b.let_val("e", lid);
            body(b, e);
        });
    } else {
        fb.for_range("e", lid, lu(fb, cells), lu(fb, WORKGROUP), body);
    }
}

/// The uniform `p` of a kernel: a struct of words, by field name.
pub(super) struct Uni {
    binding: u32,
    names: Vec<String>,
    tys: Vec<KirType>,
}

impl Uni {
    /// Declares `p` with `fields` as the next binding.
    pub(super) fn declare(mb: &mut ModuleBuilder, fields: Vec<(String, KirType)>) -> Uni {
        let layout: Vec<(&str, KirType)> = fields.iter().map(|(name, ty)| (name.as_str(), ty.clone())).collect();
        let params = mb.add_struct_layout("Params", &layout);
        let binding = mb.add_uniform("p", KirType::Struct(params));
        let (names, tys): (Vec<String>, Vec<KirType>) = fields.into_iter().unzip();
        Uni { binding, names, tys }
    }

    fn at(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|known| known == name).map(|at| at as u32)
    }

    fn field(&self, fb: &FnBuilder<'_>, at: u32) -> Val {
        fb.uniform_field(self.binding, at, self.tys[at as usize].clone())
    }
}

/// The values a program has computed so far: the inputs loaded (at context `at` of a lowered
/// program, 0 otherwise) and the results of its instructions.
#[derive(Clone, Default)]
pub(super) struct Frame {
    pub(super) a: FxMap<(u32, u32), Val>,
    pub(super) t: Vec<Option<Val>>,
    /// Per instruction that is a test (a comparison, a class test, a logic of those), its `bool` value.
    pub(super) b: Vec<Option<Val>>,
}

impl Frame {
    pub(super) fn with_len(len: usize) -> Frame {
        Frame { a: FxMap::default(), t: vec![None; len], b: vec![None; len] }
    }

    pub(super) fn t(&self, n: usize) -> Option<Val> {
        self.t.get(n).copied().flatten()
    }

    pub(super) fn b(&self, n: usize) -> Option<Val> {
        self.b.get(n).copied().flatten()
    }
}

/// The workgroup memory a lowered program reads staged sub-programs from.
pub(super) struct StagedCell<'a> {
    pub(super) staging: &'a Staging,
    /// The cell, relative to the tile origin (`i32` values).
    pub(super) lx: Val,
    pub(super) ly: Val,
    /// Per sub-program, the workgroup array that holds it.
    pub(super) shared: &'a [Option<u32>],
}

/// The instruction layer of the fused kernels (see the module's documentation): what a program
/// refers to beyond its own instructions.
pub(super) struct Lower<'a> {
    pub(super) lib: Option<Arc<Library>>,
    uni: Option<Uni>,
    /// Index of the field `s0` of the uniform.
    s0: u32,
    /// Per input, its binding (`None`: it has none, or only segments).
    pub(super) ins: Vec<Option<u32>>,
    /// Per input, the bindings of its segments (`in{k}_{s}`).
    pub(super) segs: Vec<Vec<u32>>,
    pub(super) outs: Vec<u32>,
    /// Pairs (input `k`, output `j`): input `k` is read from the output buffer `out{j}`.
    in_place: &'a [(u32, u32)],
    views: Vec<(ViewMap, Edge)>,
    view_fns: Vec<(FnId, Option<FnId>)>,
    pview_fns: Vec<(u16, FnId, Option<FnId>)>,
    sub_fns: Vec<(FnId, KirType)>,
    ld_fns: Vec<Option<(FnId, KirType)>>,
    /// The function `wrap(v, n)` of the tiled kernels.
    pub(super) wrap: Option<FnId>,
    failed: RefCell<Option<String>>,
}

impl<'a> Lower<'a> {
    pub(super) fn new(lib: Option<Arc<Library>>, in_place: &'a [(u32, u32)]) -> Lower<'a> {
        Lower {
            lib,
            uni: None,
            s0: 0,
            ins: Vec::new(),
            segs: Vec::new(),
            outs: Vec::new(),
            in_place,
            views: Vec::new(),
            view_fns: Vec::new(),
            pview_fns: Vec::new(),
            sub_fns: Vec::new(),
            ld_fns: Vec::new(),
            wrap: None,
            failed: RefCell::new(None),
        }
    }

    pub(super) fn set_uniform(&mut self, uni: Uni) {
        self.s0 = uni.at("s0").unwrap_or(0);
        self.uni = Some(uni);
    }

    /// Notes that the program cannot be lowered (what the source of a kernel would have had a
    /// shader compile error for) and stands in for the value.
    pub(super) fn fail(&self, fb: &FnBuilder<'_>, what: impl Into<String>) -> Val {
        let what = what.into();
        self.failed.borrow_mut().get_or_insert(what);
        fb.lit_u32(0)
    }

    /// The first failure noted.
    pub(super) fn error(&self) -> Option<String> {
        self.failed.borrow_mut().take()
    }

    /// A call of the helper `name` of the library.
    pub(super) fn call(&self, fb: &FnBuilder<'_>, name: &str, args: &[Val]) -> Val {
        match &self.lib {
            Some(lib) if lib.has(name) => lib.call(fb, name, args),
            _ => self.fail(fb, format!("the helper library of the kernel has no `{name}`")),
        }
    }

    /// Scalar operand `k`: field `s{k}` of the uniform.
    pub(super) fn scalar(&self, fb: &FnBuilder<'_>, k: u32) -> Val {
        match &self.uni {
            Some(uni) if ((self.s0 + k) as usize) < uni.tys.len() => uni.field(fb, self.s0 + k),
            _ => self.fail(fb, format!("scalar {k} is not in the uniform")),
        }
    }

    /// The field `name` of the uniform.
    pub(super) fn named(&self, fb: &FnBuilder<'_>, name: &str) -> Val {
        match self.uni.as_ref().and_then(|uni| uni.at(name).map(|at| (uni, at))) {
            Some((uni, at)) => uni.field(fb, at),
            None => self.fail(fb, format!("the uniform has no `{name}`")),
        }
    }

    /// An operand of an instruction, of type `ty`, computed at context `at` of a lowered program
    /// (0 for a plain one). The scalars of an `f64` or `Df32` operand are several `f32` slots.
    pub(super) fn operand(&self, fb: &FnBuilder<'_>, f: &Frame, reference: Ref, ty: ElemTy, at: u32) -> Val {
        match reference {
            Ref::Unused => fb.lit_u32(0),
            Ref::Input(k) => match f.a.get(&(k, at)) {
                Some(&v) => v,
                None => self.fail(fb, format!("input {k} is not loaded at context {at}")),
            },
            Ref::Scalar(k) => match ty {
                ElemTy::F64 => {
                    let wide = |j: u32| fb.convert(self.scalar(fb, k + j), KirType::Scalar(ScalarTy::F64));
                    (wide(0) + wide(1)) + wide(2)
                }
                ElemTy::Df32 => fb.compose(reg(ElemTy::Df32), &[self.scalar(fb, k), self.scalar(fb, k + 1)]),
                ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::Bool => self.scalar(fb, k),
            },
            Ref::Tmp(k) => match f.t(k as usize) {
                Some(v) => v,
                None => self.fail(fb, format!("t{k} is read before it is computed")),
            },
            Ref::Sub(k) => self.fail(fb, format!("sub-program {k} is not an operand")),
        }
    }

    // ---- divisions by a divisor of the uniform ----

    /// `n / d` for the divisor whose words start at uniform slot `slot`; `wide` for any `n`, else
    /// `n < 2^23` (see `params::divisor_words`).
    pub(super) fn udiv(&self, fb: &mut FnBuilder<'_>, n: Val, slot: u32, wide: bool) -> Val {
        if wide {
            let (m, s) = (self.scalar(fb, slot + 1), self.scalar(fb, slot + 2));
            let t = fb.mul_hi_u32(n, m);
            let t = fb.let_val("t", t);
            let one = lu(fb, 1);
            (t + ((n - t) >> one)) >> s
        } else {
            let (d, r) = (self.scalar(fb, slot), self.scalar(fb, slot + 3));
            self.call(fb, "luma_udiv", &[n, d, r])
        }
    }

    /// `n % d`, see [`Lower::udiv`].
    pub(super) fn urem(&self, fb: &mut FnBuilder<'_>, n: Val, slot: u32, wide: bool) -> Val {
        let n = bind(fb, "n", n);
        let q = self.udiv(fb, n, slot, wide);
        let d = self.scalar(fb, slot);
        n - q * d
    }

    // ---- operations ----

    fn unary(&self, fb: &mut FnBuilder<'_>, op: UnOp, ty: ElemTy, x: Val) -> Val {
        if ty == ElemTy::Df32 {
            let name = match op {
                UnOp::Neg => return fb.neg(x),
                UnOp::Id => return x,
                UnOp::Abs => "df_abs",
                UnOp::Sqrt => "df_sqrt",
                UnOp::Exp => "df_exp",
                UnOp::Ln => "df_ln",
                UnOp::Tanh => "df_tanh",
                UnOp::Log1p => "df_log1p",
                UnOp::Expm1 => "df_expm1",
                other => return self.fail(fb, format!("{other:?} of a Df32")),
            };
            return self.call(fb, name, &[x]);
        }
        if ty == ElemTy::Bool {
            return match op {
                UnOp::Id => x,
                UnOp::BitNot => {
                    let one = lu(fb, 1);
                    fb.bit_xor(x, one)
                }
                other => self.fail(fb, format!("{other:?} of a bool")),
            };
        }
        let float = |op: IrUnOp| fb.unary(op, x);
        match op {
            UnOp::Id => x,
            UnOp::BitNot => fb.bit_not(x),
            UnOp::Shl(0) | UnOp::Shr(0) | UnOp::Rotl(0) => x,
            UnOp::Shl(k) => fb.shl(x, lu(fb, u32::from(k))),
            UnOp::Shr(k) => fb.shr(x, lu(fb, u32::from(k))),
            UnOp::Rotl(k) => fb.bit_or(fb.shl(x, lu(fb, u32::from(k))), fb.shr(x, lu(fb, 32 - u32::from(k)))),
            UnOp::Log1p => self.call(fb, "luma_log1p", &[x]),
            UnOp::Expm1 => self.call(fb, "luma_expm1", &[x]),
            UnOp::Neg => float(IrUnOp::Neg),
            UnOp::Sqrt => float(IrUnOp::Sqrt),
            UnOp::Exp => float(IrUnOp::Exp),
            UnOp::Ln => float(IrUnOp::Ln),
            UnOp::Abs => float(IrUnOp::Abs),
            UnOp::Sin => float(IrUnOp::Sin),
            UnOp::Cos => float(IrUnOp::Cos),
            UnOp::Tan => float(IrUnOp::Tan),
            // The backends bound `tanh` where the device's overflows.
            UnOp::Tanh => float(IrUnOp::Tanh),
            UnOp::Sign => float(IrUnOp::Sign),
            UnOp::Floor => float(IrUnOp::Floor),
            UnOp::Ceil => float(IrUnOp::Ceil),
            // `f64` `round` and `trunc` are exact on every backend.
            UnOp::Trunc => float(IrUnOp::Trunc),
            UnOp::Round => float(IrUnOp::Round),
        }
    }

    fn binary(&self, fb: &mut FnBuilder<'_>, op: BinOp, ty: ElemTy, x: Val, y: Val) -> Val {
        if ty == ElemTy::Df32 {
            return match op {
                BinOp::Add => self.call(fb, "df_add", &[x, y]),
                BinOp::Sub => {
                    let negated = fb.neg(y);
                    self.call(fb, "df_add", &[x, negated])
                }
                BinOp::Mul => self.call(fb, "df_mul", &[x, y]),
                BinOp::Div => self.call(fb, "df_div", &[x, y]),
                BinOp::Min => self.call(fb, "df_min", &[x, y]),
                BinOp::Max => self.call(fb, "df_max", &[x, y]),
                other => self.fail(fb, format!("{other:?} of Df32 values")),
            };
        }
        // A shift or rotation amount is taken modulo 32.
        let amount = |fb: &FnBuilder<'_>| {
            let word = if ty == ElemTy::I32 { fb.bitcast(y, KirType::U32) } else { y };
            word & lu(fb, 31)
        };
        match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::Rem => x % y,
            BinOp::Min => fb.binary(IrBinOp::Min, x, y),
            BinOp::Max => fb.binary(IrBinOp::Max, x, y),
            BinOp::Atan2 => fb.binary(IrBinOp::Atan2, x, y),
            BinOp::Hypot => self.call(fb, "luma_hypot", &[x, y]),
            BinOp::Copysign => {
                let (xb, yb) = (fb.bitcast(x, KirType::U32), fb.bitcast(y, KirType::U32));
                let bits = fb.bit_or(fb.bit_and(xb, lu(fb, 0x7fff_ffff)), fb.bit_and(yb, lu(fb, 0x8000_0000)));
                fb.bitcast(bits, KirType::F32)
            }
            BinOp::BitAnd => x & y,
            BinOp::BitOr => x | y,
            BinOp::BitXor => x ^ y,
            BinOp::Shl => x << amount(fb),
            BinOp::Shr => x >> amount(fb),
            BinOp::Rotl => {
                let back = (lu(fb, 32) - y) & lu(fb, 31);
                fb.bit_or(x << amount(fb), x >> back)
            }
        }
    }

    /// The comparison as a `bool`.
    fn compare_cond(&self, fb: &mut FnBuilder<'_>, op: CmpOp, ty: ElemTy, x: Val, y: Val) -> Val {
        if ty == ElemTy::Df32 {
            let cond = match op {
                CmpOp::Lt => self.call(fb, "df_lt", &[x, y]),
                CmpOp::Le => self.call(fb, "df_le", &[x, y]),
                CmpOp::Gt => self.call(fb, "df_lt", &[y, x]),
                CmpOp::Ge => self.call(fb, "df_le", &[y, x]),
                CmpOp::Eq | CmpOp::Ne => self.call(fb, "df_eq", &[x, y]),
            };
            return if op == CmpOp::Ne { fb.unary(IrUnOp::Not, cond) } else { cond };
        }
        match op {
            CmpOp::Lt => fb.lt(x, y),
            CmpOp::Le => fb.le(x, y),
            CmpOp::Gt => fb.gt(x, y),
            CmpOp::Ge => fb.ge(x, y),
            CmpOp::Eq => fb.eq(x, y),
            CmpOp::Ne => fb.ne(x, y),
        }
    }

    /// The comparison as a `u32` of 0 or 1.
    fn compare(&self, fb: &mut FnBuilder<'_>, op: CmpOp, ty: ElemTy, x: Val, y: Val) -> Val {
        let cond = self.compare_cond(fb, op, ty, x, y);
        fb.select(cond, lu(fb, 1), lu(fb, 0))
    }

    /// A test of what kind of value `x` is, from the bits of its exponent. They come through an
    /// optimisation barrier (`luma_bits`): a compiler that assumes no `NaN` or infinity arises
    /// in the arithmetic of the kernel would otherwise fold a test of the bits of a computed
    /// value to its finite answer. The operand is an `f32` with the class of the value (see
    /// `builtin::class_operand`): the high word of a `Df32`; an `f64` scaled into the range of
    /// `f32`.
    fn classify_cond(&self, fb: &mut FnBuilder<'_>, class: Class, ty: ElemTy, x: Val) -> Val {
        let operand = match ty {
            ElemTy::Df32 => fb.field(x, 0, KirType::F32),
            ElemTy::F64 => {
                let scaled = x * fb.lit_f64(9.332636185032189e-302);
                fb.convert(scaled, KirType::F32)
            }
            _ => x,
        };
        let guarded = fb.opaque_guard(operand);
        let bits = fb.bitcast(guarded, KirType::U32);
        let magnitude = bits & lu(fb, 0x7fff_ffff);
        let infinity = lu(fb, 0x7f80_0000);
        match class {
            Class::Finite => fb.lt(magnitude, infinity),
            Class::Nan => fb.gt(magnitude, infinity),
            Class::Inf => fb.eq(magnitude, infinity),
        }
    }

    /// [`Lower::classify_cond`] as a `u32` of 0 or 1.
    fn classify(&self, fb: &mut FnBuilder<'_>, class: Class, ty: ElemTy, x: Val) -> Val {
        let cond = self.classify_cond(fb, class, ty, x);
        fb.select(cond, lu(fb, 1), lu(fb, 0))
    }

    /// The `bool` of a comparison or class test `op` on `x` and `y` (`None` for any other operation).
    pub(super) fn bool_condition(&self, fb: &mut FnBuilder<'_>, op: Op, x: Val, y: Val) -> Option<Val> {
        match op.kind {
            OpKind::Compare(cmp) => Some(self.compare_cond(fb, cmp, op.ty, x, y)),
            OpKind::Classify(class) => Some(self.classify_cond(fb, class, op.ty, x)),
            _ => None,
        }
    }

    /// `x` of type `from` converted to `to`. Every conversion that has a `Df32` end passes the
    /// value that a reassociating compiler could fold away through the optimisation barrier (see
    /// `builtin::df_library`), so that `x - to_df32(to_f32(x))` keeps the low word of `x`.
    fn cast(&self, fb: &mut FnBuilder<'_>, from: ElemTy, to: ElemTy, x: Val) -> Val {
        let f64t = KirType::Scalar(ScalarTy::F64);
        let df = reg(ElemTy::Df32);
        match (from, to) {
            (ElemTy::F32, ElemTy::F64) => fb.convert(x, f64t),
            (ElemTy::F64, ElemTy::F32) => fb.convert(x, KirType::F32),
            (ElemTy::F32, ElemTy::Df32) => fb.compose(df, &[fb.opaque_guard(x), lf(fb, 0.0)]),
            (ElemTy::Df32, ElemTy::F32) => {
                let sum = fb.field(x, 0, KirType::F32) + fb.opaque_guard(fb.field(x, 1, KirType::F32));
                fb.opaque_guard(sum)
            }
            (ElemTy::F64, ElemTy::Df32) => {
                let hi = fb.opaque_guard(fb.convert(x, KirType::F32));
                let lo = fb.convert(x - fb.convert(hi, f64t), KirType::F32);
                fb.compose(df, &[hi, lo])
            }
            (ElemTy::Df32, ElemTy::F64) => {
                let hi = fb.convert(fb.opaque_guard(fb.field(x, 0, KirType::F32)), f64t.clone());
                hi + fb.convert(fb.field(x, 1, KirType::F32), f64t)
            }
            (ElemTy::U32 | ElemTy::I32 | ElemTy::Bool, ElemTy::F32) => fb.convert(x, KirType::F32),
            (ElemTy::F32, ElemTy::U32) => {
                let nan = self.classify(fb, Class::Nan, ElemTy::F32, x);
                let clamped = fb.clamp(x, lf(fb, 0.0), lf(fb, 4294967040.0));
                let value = fb.float_to_int_sat(clamped, KirType::U32);
                fb.select(fb.ne(nan, lu(fb, 0)), lu(fb, 0), value)
            }
            (ElemTy::F32, ElemTy::I32) => {
                let nan = self.classify(fb, Class::Nan, ElemTy::F32, x);
                let clamped = fb.clamp(x, lf(fb, -2147483648.0), lf(fb, 2147483520.0));
                let value = fb.float_to_int_sat(clamped, KirType::I32);
                fb.select(fb.ne(nan, lu(fb, 0)), li(fb, 0), value)
            }
            (ElemTy::U32, ElemTy::I32) | (ElemTy::Bool, ElemTy::I32) => fb.bitcast(x, KirType::I32),
            (ElemTy::I32, ElemTy::U32) => fb.bitcast(x, KirType::U32),
            (ElemTy::Bool, ElemTy::U32) => x,
            (ElemTy::Bool, ElemTy::F64) => fb.convert(x, f64t),
            (ElemTy::Bool, ElemTy::Df32) => fb.compose(df, &[fb.convert(x, KirType::F32), lf(fb, 0.0)]),
            _ if from == to => x,
            _ => self.fail(fb, format!("cast {from:?} -> {to:?}")),
        }
    }

    /// The value a generator gives for the element index `coord`, as an element of type `ty`.
    fn shaped(&self, fb: &FnBuilder<'_>, ty: ElemTy, coord: Val) -> Val {
        match ty {
            ElemTy::U32 => coord,
            ElemTy::I32 => fb.bitcast(coord, KirType::I32),
            ElemTy::F32 | ElemTy::F64 | ElemTy::Df32 => fb.convert(coord, KirType::F32),
            ElemTy::Bool => self.fail(fb, "a generator of bools"),
        }
    }

    fn generate(&self, fb: &mut FnBuilder<'_>, kind: Gen, ty: ElemTy, index: Val, x: Val, y: Val) -> Val {
        let threefry = |fb: &FnBuilder<'_>| self.call(fb, "luma_threefry", &[index, x, y]);
        match kind {
            Gen::Bits => {
                let words = threefry(fb);
                fb.field(words, 0, KirType::U32)
            }
            Gen::Uniform => {
                let words = threefry(fb);
                let first = fb.field(words, 0, KirType::U32);
                self.call(fb, "luma_uniform", &[first])
            }
            Gen::Normal => {
                let words = threefry(fb);
                self.call(fb, "luma_normal", &[words])
            }
            Gen::Index => self.shaped(fb, ty, index),
            Gen::RangePow2 => {
                let coord = (index >> x) & y;
                self.shaped(fb, ty, coord)
            }
            Gen::RangeConst { stride, size, .. } => {
                let coord = if stride == 1 { index % lu(fb, size) } else { (index / lu(fb, stride)) % lu(fb, size) };
                self.shaped(fb, ty, coord)
            }
            Gen::ParamRange { kind, wide, stride, size, .. } => {
                let coord = match kind {
                    params::RangeKind::Zero => lu(fb, 0),
                    params::RangeKind::Mod => self.urem(fb, index, size, wide),
                    params::RangeKind::DivMod => {
                        let quotient = self.udiv(fb, index, stride, wide);
                        self.urem(fb, quotient, size, wide)
                    }
                };
                self.shaped(fb, ty, coord)
            }
            Gen::Linspace => x + fb.convert(index, KirType::F32) * y,
        }
    }

    /// The operation `op` (anything but a shift) on the operands `x`, `y` and `z`; `index` is the
    /// element it is computed at. `None` for a shift.
    pub(super) fn operation(&self, fb: &mut FnBuilder<'_>, op: Op, x: Val, y: Val, z: Val, index: Val) -> Option<Val> {
        let ty = op.ty;
        Some(match op.kind {
            OpKind::Unary(u) => self.unary(fb, u, ty, x),
            OpKind::Binary(b) => self.binary(fb, b, ty, x, y),
            OpKind::Compare(c) => self.compare(fb, c, ty, x, y),
            OpKind::Classify(class) => self.classify(fb, class, ty, x),
            OpKind::Powf if ty == ElemTy::Df32 => self.call(fb, "df_pow", &[x, y]),
            OpKind::Powf => fb.binary(IrBinOp::Pow, x, y),
            OpKind::Clamp if ty == ElemTy::Df32 => {
                let above = self.call(fb, "df_max", &[x, y]);
                self.call(fb, "df_min", &[above, z])
            }
            OpKind::Clamp => fb.clamp(x, y, z),
            OpKind::Select => {
                let cond = fb.ne(x, lu(fb, 0));
                fb.select(cond, y, z)
            }
            OpKind::Cast(to) => self.cast(fb, ty, to, x),
            OpKind::Generate(kind) => self.generate(fb, kind, op.out_ty(), index, x, y),
            OpKind::Shift(_) => return None,
        })
    }

    // ---- loads and stores ----

    /// Element `index` of the read-only buffer `binding`, as a register value of type `ty`: a
    /// `bool` is bit `index % 32` of word `index / 32`, every other type is stored as itself.
    pub(super) fn load_in(&self, fb: &mut FnBuilder<'_>, ty: ElemTy, binding: u32, index: Val) -> Val {
        match ty {
            ElemTy::Bool => {
                let j = bind(fb, "j", index);
                let word = fb.load(binding, j >> lu(fb, 5));
                (word >> (j & lu(fb, 31))) & lu(fb, 1)
            }
            _ => fb.load(binding, index),
        }
    }

    /// [`Lower::load_in`] from a read-write buffer: a kernel that overwrites an input reads it
    /// from the output buffer.
    fn load_out(&self, fb: &mut FnBuilder<'_>, ty: ElemTy, binding: u32, index: Val) -> Val {
        match ty {
            ElemTy::Bool => {
                let j = bind(fb, "j", index);
                let word = fb.atomic_load(storage_target(binding, j >> lu(fb, 5)), ScalarTy::U32);
                (word >> (j & lu(fb, 31))) & lu(fb, 1)
            }
            _ => fb.load(binding, index),
        }
    }

    /// Element `index` of input `k`, from where the kernel has the input: its own buffer, or,
    /// with `inplace`, for an input the kernel overwrites, the output buffer.
    pub(super) fn input_load(&self, fb: &mut FnBuilder<'_>, ty: ElemTy, k: u32, index: Val, inplace: bool) -> Val {
        if inplace && let Some(&(_, j)) = self.in_place.iter().find(|&&(input, _)| input == k) {
            return match self.outs.get(j as usize) {
                Some(&binding) => self.load_out(fb, ty, binding, index),
                None => self.fail(fb, format!("output {j} of an in-place input is not bound")),
            };
        }
        match self.ins.get(k as usize).copied().flatten() {
            Some(binding) => self.load_in(fb, ty, binding, index),
            None => self.fail(fb, format!("input {k} has no buffer")),
        }
    }

    /// Stores the register value `value` of type `ty` as the element `index` of the read-write
    /// buffer `binding`. A bool sets or clears its own bit atomically and leaves the others of the
    /// word alone: the buffer must hold zeros past the end of the array, and the kernels that
    /// write every element at once (a store of whole words) do it themselves.
    pub(super) fn store(&self, fb: &mut FnBuilder<'_>, ty: ElemTy, binding: u32, index: Val, value: Val) {
        match ty {
            ElemTy::Bool => {
                let j = fb.let_val("luma_j", index);
                let m = fb.lit_u32(1) << (j & lu(fb, 31));
                let m = fb.let_val("luma_m", m);
                let word = j >> lu(fb, 5);
                let set = fb.ne(value, lu(fb, 0));
                let target = move || storage_target(binding, word);
                fb.if_else(
                    set,
                    |b| b.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::U32, target(), m),
                    |b| b.atomic_rmw(None, IrAtomicOp::And, ScalarTy::U32, target(), !m),
                );
            }
            _ => fb.store(binding, index, value),
        }
    }

    // ---- shifts ----

    /// The index element `i` of a shift reads; `s` is the scalar shift of [`ShiftKind::Linear`],
    /// `total` the array length (scalar operand 2).
    pub(super) fn shift_index(&self, fb: &mut FnBuilder<'_>, kind: ShiftKind, s: Val, i: Val, total: Val) -> Val {
        match kind {
            ShiftKind::Linear => {
                let wrapped = i + (total - s);
                fb.select(fb.lt(i, s), wrapped, i - s)
            }
            ShiftKind::View { .. } | ShiftKind::ParamView { .. } => self.fail(fb, "views are indexed by their own functions"),
            ShiftKind::Take { len } => fb.binary(IrBinOp::Min, s, lu(fb, len.saturating_sub(1))),
            ShiftKind::ParamTake { max_slot } => {
                let top = self.scalar(fb, max_slot);
                fb.binary(IrBinOp::Min, s, top)
            }
            ShiftKind::ParamDims { axes } => {
                let mut terms = Vec::new();
                for axis in axes {
                    if !axis.used {
                        continue;
                    }
                    let [stride_slot, rem_slot, shift_slot] = axis.slots;
                    // The stride the coordinate is scaled back by is a shift for a power of two.
                    let (quotient, scale) = match axis.stride {
                        DivRemSpec::None => (i, None),
                        DivRemSpec::Pow2 => (i >> self.scalar(fb, stride_slot), Some((true, stride_slot))),
                        DivRemSpec::Runtime { wide } => (self.udiv(fb, i, stride_slot, wide), Some((false, stride_slot))),
                    };
                    let (coord, dim) = match (axis.rem, axis.shifted) {
                        (DivRemSpec::None, false) => (quotient, None),
                        (DivRemSpec::Pow2, _) => {
                            let mask = self.scalar(fb, rem_slot);
                            (quotient & mask, Some(mask + lu(fb, 1)))
                        }
                        (DivRemSpec::Runtime { wide }, _) => (self.urem(fb, quotient, rem_slot, wide), Some(self.scalar(fb, rem_slot))),
                        (DivRemSpec::None, true) => (quotient, Some(self.scalar(fb, rem_slot))),
                    };
                    let wrapped = if axis.shifted {
                        let shift = self.scalar(fb, shift_slot);
                        let coord = bind(fb, "c", coord);
                        let dim = dim.unwrap_or_else(|| lu(fb, 0));
                        fb.select(fb.lt(coord, shift), coord + (dim - shift), coord - shift)
                    } else {
                        coord
                    };
                    terms.push(match scale {
                        Some((true, slot)) => wrapped << self.scalar(fb, slot),
                        Some((false, slot)) => wrapped * self.scalar(fb, slot),
                        None => wrapped,
                    });
                }
                sum(fb, terms)
            }
            ShiftKind::Dims { dims, shifts } => {
                let [d0, d1, d2, _] = dims;
                let strides = [1, d0, d0 * d1, d0 * d1 * d2];
                let mut terms = Vec::new();
                for axis in 0..4 {
                    let (dim, shift, stride) = (dims[axis], shifts[axis], strides[axis]);
                    if dim == 1 {
                        continue;
                    }
                    let coord = match (stride, axis) {
                        (1, _) => i % lu(fb, dim),
                        (_, 3) => i / lu(fb, stride),
                        _ => (i / lu(fb, stride)) % lu(fb, dim),
                    };
                    let wrapped = if shift == 0 {
                        coord
                    } else {
                        let coord = bind(fb, "c", coord);
                        fb.select(fb.lt(coord, lu(fb, shift)), coord + lu(fb, dim - shift), coord - lu(fb, shift))
                    };
                    terms.push(if stride == 1 { wrapped } else { wrapped * lu(fb, stride) });
                }
                sum(fb, terms)
            }
        }
    }

    // ---- views ----

    fn view_index(&self, fb: &FnBuilder<'_>, n: Option<usize>, x: Val) -> Val {
        match n.and_then(|n| self.view_fns.get(n)) {
            Some(&(index, _)) => fb.call(index, &[x], KirType::U32),
            None => self.fail(fb, "a view of the program has no function"),
        }
    }

    fn view_inside(&self, fb: &FnBuilder<'_>, n: Option<usize>, x: Val) -> Val {
        match n.and_then(|n| self.view_fns.get(n)).and_then(|&(_, inside)| inside) {
            Some(inside) => fb.call(inside, &[x], KirType::BOOL),
            None => self.fail(fb, "a view of the program has no test of the source"),
        }
    }

    fn pview_fn(&self, id: u16) -> Option<(FnId, Option<FnId>)> {
        self.pview_fns.iter().find(|&&(known, ..)| known == id).map(|&(_, index, inside)| (index, inside))
    }

    fn pview_index(&self, fb: &FnBuilder<'_>, id: u16, x: Val) -> Val {
        match self.pview_fn(id) {
            Some((index, _)) => fb.call(index, &[x], KirType::U32),
            None => self.fail(fb, format!("parameterized view {id} has no function")),
        }
    }

    fn pview_inside(&self, fb: &FnBuilder<'_>, id: u16, x: Val) -> Val {
        match self.pview_fn(id).and_then(|(_, inside)| inside) {
            Some(inside) => fb.call(inside, &[x], KirType::BOOL),
            None => self.fail(fb, format!("parameterized view {id} has no test of the source")),
        }
    }

    /// The function of the view of a store through the view `map`.
    pub(super) fn mapped_view(&self, tail: Tail) -> Option<FnId> {
        match tail {
            Tail::StoreMapped(map) => {
                let n = self.views.iter().position(|&known| known == (map, Edge::None)).unwrap_or(0);
                self.view_fns.get(n).map(|&(index, _)| index)
            }
            Tail::ParamStoreMapped { id, .. } => self.pview_fn(id).map(|(index, _)| index),
            _ => None,
        }
    }

    /// The body of `view{n}(i)`: the flat source index view `map` reads for output element `i`,
    /// or, with `inside`, the body of `view{n}_in(i)`: whether it lies inside the source.
    fn view_body(&self, fb: &mut FnBuilder<'_>, map: ViewMap, edge: Edge, inside_test: bool) {
        let i = fb.param(0);
        let mut out_stride = [1u64; 4];
        let mut src_stride = [1u64; 4];
        for d in 1..4 {
            out_stride[d] = out_stride[d - 1] * u64::from(map.out[d - 1]);
            src_stride[d] = src_stride[d - 1] * u64::from(map.src[d - 1]);
        }
        // The unclipped source coordinate of axis `a`.
        let raw = |fb: &FnBuilder<'_>, a: usize| -> Val {
            let axis = map.axes[a];
            let from = axis.from as usize;
            if map.out[from] == 1 {
                return li(fb, axis.start);
            }
            let quotient = if out_stride[from] == 1 { i } else { i / lu(fb, out_stride[from] as u32) };
            let along = fb.convert(quotient % lu(fb, map.out[from]), KirType::I32);
            along * li(fb, axis.step) + li(fb, axis.start)
        };
        // Whether the unclipped coordinate of axis `a` is always inside the source.
        let inside = |a: usize| {
            let (first, last) = map.ends(a);
            let size = i64::from(map.src[a]);
            (0..size).contains(&first) && (0..size).contains(&last)
        };
        if inside_test {
            let mut all: Option<Val> = None;
            for a in (0..4).filter(|&a| !inside(a)) {
                let (low, high) = (raw(fb, a), raw(fb, a));
                let test = fb.and(fb.ge(low, li(fb, 0)), fb.lt(high, li(fb, map.src[a] as i32)));
                all = Some(match all {
                    Some(previous) => fb.and(previous, test),
                    None => test,
                });
            }
            let all = all.unwrap_or_else(|| fb.lit_bool(true));
            fb.ret(Some(all));
            return;
        }
        let mut terms = Vec::new();
        for a in (0..4).filter(|&a| map.src[a] > 1) {
            let size = map.src[a] as i32;
            let r = raw(fb, a);
            let coord = match edge {
                _ if inside(a) => r,
                Edge::None => r,
                Edge::Clamp | Edge::Fill => self.call(fb, "luma_clamp_i", &[r, li(fb, size)]),
                Edge::Mirror => self.call(fb, "luma_mirror_i", &[r, li(fb, size)]),
                Edge::Wrap => self.call(fb, "luma_wrap_i", &[r, li(fb, size)]),
            };
            let word = fb.bitcast(coord, KirType::U32);
            terms.push(if src_stride[a] == 1 { word } else { word * lu(fb, src_stride[a] as u32) });
        }
        let total = sum(fb, terms);
        fb.ret(Some(total));
    }

    /// The terms of the flat source index of the parameterized view `spec` for element `i`, and
    /// per axis the unclipped coordinate, the source extent where the view checks it, and the
    /// source coordinate inside `0..extent` of every active axis. Where `cache` has the source
    /// coordinates of context `parent` and that source has the dimensions of this view's output,
    /// a coordinate along the output is taken from them (`AxisSpec::from`) instead of from
    /// dividing `i`: `i` is their flat index.
    #[allow(clippy::type_complexity)]
    fn pview_axes(&self, fb: &mut FnBuilder<'_>, cache: &mut PviewCache, spec: ViewSpec, parent: u32, i: Val) -> (Vec<Val>, [Option<Val>; 4], [Option<Val>; 4], [Option<Val>; 4]) {
        let mut raws: [Option<Val>; 4] = [None; 4];
        let mut sizes: [Option<Val>; 4] = [None; 4];
        let mut coords: [Option<Val>; 4] = [None; 4];
        let mut terms = Vec::new();
        let known = cache.coords.iter().find(|&&(c, dims, _)| c == parent && dims == spec.out_dims).map(|&(.., coords)| coords);
        for a in 0..4 {
            let axis = spec.axes[a];
            if !(axis.active || (spec.edge == Edge::Fill && axis.checked)) {
                continue;
            }
            let slots = axis.slots;
            let mut scaled = None;
            if !axis.out_unit && axis.step != StepSpec::Zero {
                let along = match known {
                    // An axis of the parent's source without a coordinate has one element.
                    Some(known) => known[axis.from as usize].unwrap_or_else(|| li(fb, 0)),
                    None => {
                        let along = match (axis.out_div, axis.out_rem) {
                            (DivRemSpec::None, DivRemSpec::None) => i,
                            (DivRemSpec::None, rem) => self.pview_rem(fb, cache, parent, i, slots.rem, rem),
                            (div, DivRemSpec::None) => self.pview_div(fb, cache, parent, i, slots.div, div),
                            (div, rem) => {
                                let quotient = self.pview_div(fb, cache, parent, i, slots.div, div);
                                match rem {
                                    DivRemSpec::Pow2 => quotient & self.scalar(fb, slots.rem),
                                    DivRemSpec::Runtime { wide } => self.urem(fb, quotient, slots.rem, wide),
                                    DivRemSpec::None => quotient,
                                }
                            }
                        };
                        fb.convert(along, KirType::I32)
                    }
                };
                scaled = Some(match axis.step {
                    StepSpec::PosOne => along,
                    StepSpec::NegOne => fb.neg(along),
                    _ => along * fb.bitcast(self.scalar(fb, slots.step), KirType::I32),
                });
            }
            let start = fb.bitcast(self.scalar(fb, slots.start), KirType::I32);
            // The coordinate of the view the start word belongs to, then this view's offset from
            // it: neighbour reads share everything but the last addition.
            let base = match scaled {
                None => start,
                Some(scaled) => scaled + start,
            };
            let raw = if axis.start_delta == 0 { base } else { base + li(fb, i32::from(axis.start_delta)) };
            let size = axis.checked.then(|| fb.bitcast(self.scalar(fb, slots.size), KirType::I32));
            let stride = (axis.active && !axis.src_fastest).then(|| self.scalar(fb, slots.stride));
            if axis.active {
                let coord = match (size, spec.edge) {
                    (Some(size), Edge::Clamp | Edge::Fill) => self.call(fb, "luma_clamp_i", &[raw, size]),
                    (Some(size), Edge::Mirror) => self.call(fb, "luma_mirror_i", &[raw, size]),
                    (Some(size), Edge::Wrap) if axis.wrap_mask => {
                        let mask = fb.bitcast(size, KirType::U32);
                        self.call(fb, "luma_wrap_mask", &[raw, mask])
                    }
                    (Some(size), Edge::Wrap) => self.call(fb, "luma_wrap_i", &[raw, size]),
                    _ => raw,
                };
                let coord = fb.let_val(&format!("x{a}"), coord);
                coords[a] = Some(coord);
                let word = fb.bitcast(coord, KirType::U32);
                terms.push(match stride {
                    Some(stride) => word * stride,
                    None => word,
                });
            }
            raws[a] = Some(raw);
            sizes[a] = size;
        }
        (terms, raws, sizes, coords)
    }

    /// `i / d` of the view axis divisor at `slot` (a shift for `Pow2`), once per context `parent` and divisor.
    fn pview_div(&self, fb: &mut FnBuilder<'_>, cache: &mut PviewCache, parent: u32, i: Val, slot: u32, kind: DivRemSpec) -> Val {
        let pow2 = matches!(kind, DivRemSpec::Pow2);
        if let Some(&(.., q)) = cache.divs.iter().find(|&&(p, s, w, _)| p == parent && s == slot && w == pow2) {
            return q;
        }
        let q = match kind {
            DivRemSpec::Pow2 => i >> self.scalar(fb, slot),
            DivRemSpec::Runtime { wide } => self.udiv(fb, i, slot, wide),
            DivRemSpec::None => i,
        };
        let q = fb.let_val(&format!("q_{slot}"), q);
        cache.divs.push((parent, slot, pow2, q));
        q
    }

    /// `i % d` of the view axis modulus at `slot` (a mask for `Pow2`), once per context `parent` and modulus.
    fn pview_rem(&self, fb: &mut FnBuilder<'_>, cache: &mut PviewCache, parent: u32, i: Val, slot: u32, kind: DivRemSpec) -> Val {
        let pow2 = matches!(kind, DivRemSpec::Pow2);
        if let Some(&(.., r)) = cache.rems.iter().find(|&&(p, s, w, _)| p == parent && s == slot && w == pow2) {
            return r;
        }
        let r = match kind {
            DivRemSpec::Pow2 => i & self.scalar(fb, slot),
            DivRemSpec::Runtime { .. } => {
                let q = self.pview_div(fb, cache, parent, i, slot, kind);
                i - q * self.scalar(fb, slot)
            }
            DivRemSpec::None => i,
        };
        let r = fb.let_val(&format!("r_{slot}"), r);
        cache.rems.push((parent, slot, pow2, r));
        r
    }

    /// The flat source index of the view (or, with `inside_test`, whether it lies inside the source) from the result of [`Lower::pview_axes`].
    fn pview_result(&self, fb: &mut FnBuilder<'_>, terms: Vec<Val>, raws: [Option<Val>; 4], sizes: [Option<Val>; 4], inside_test: bool) -> Val {
        if !inside_test {
            return sum(fb, terms);
        }
        let mut all: Option<Val> = None;
        for a in 0..4 {
            if let (Some(raw), Some(size)) = (raws[a], sizes[a]) {
                let test = fb.and(fb.ge(raw, li(fb, 0)), fb.lt(raw, size));
                all = Some(match all {
                    Some(previous) => fb.and(previous, test),
                    None => test,
                });
            }
        }
        all.unwrap_or_else(|| fb.lit_bool(true))
    }

    /// The body of `pview{id}(i)`, or, with `inside_test`, of `pview{id}_in(i)`.
    fn pview_body(&self, fb: &mut FnBuilder<'_>, spec: ViewSpec, inside_test: bool) {
        let i = fb.param(0);
        let (terms, raws, sizes, _) = self.pview_axes(fb, &mut PviewCache::default(), spec, 0, i);
        let result = self.pview_result(fb, terms, raws, sizes, inside_test);
        fb.ret(Some(result));
    }

    /// Declares the functions of `views`, of the parameterized views `pviews`, the segment
    /// loaders of the segmented variant and, when `with_subs`, the sub-programs of `key`.
    pub(super) fn declare_functions(
        &mut self,
        mb: &mut ModuleBuilder,
        key: &FusedKey,
        views: &[(ViewMap, Edge)],
        pviews: &[(u16, ViewSpec)],
        segment_len: Option<u32>,
        with_subs: bool,
    ) {
        let one = [("i", KirType::U32)];
        self.views = views.to_vec();
        for (n, &(map, edge)) in views.iter().enumerate() {
            let index = mb.add_function(&format!("view{n}"), &one, Some(KirType::U32), |fb| self.view_body(fb, map, edge, false));
            let inside = (edge == Edge::Fill)
                .then(|| mb.add_function(&format!("view{n}_in"), &one, Some(KirType::BOOL), |fb| self.view_body(fb, map, edge, true)));
            self.view_fns.push((index, inside));
        }
        for &(id, spec) in pviews {
            let index = mb.add_function(&format!("pview{id}"), &one, Some(KirType::U32), |fb| self.pview_body(fb, spec, false));
            let inside = (spec.edge == Edge::Fill)
                .then(|| mb.add_function(&format!("pview{id}_in"), &one, Some(KirType::BOOL), |fb| self.pview_body(fb, spec, true)));
            self.pview_fns.push((id, index, inside));
        }
        if let Some(segment_len) = segment_len {
            for (k, &ty) in key.input_tys.iter().enumerate() {
                let m = self.segs.get(k).map_or(0, Vec::len);
                if m == 0 {
                    self.ld_fns.push(None);
                    continue;
                }
                // `segment_len` counts 4-byte words: an 8-byte element halves it.
                let per = segment_len / (ty.stride() as u32 / 4).max(1);
                let (log, mask) = (per.trailing_zeros(), per.wrapping_sub(1));
                let ret = reg(ty);
                let id = mb.add_function(&format!("ld{k}"), &[("g", KirType::U32)], Some(ret.clone()), |fb| {
                    let g = fb.param(0);
                    let c = fb.let_val("c", g >> lu(fb, log));
                    let o = fb.let_val("o", g & lu(fb, mask));
                    for s in 0..m - 1 {
                        let hit = fb.eq(c, self.named(fb, &format!("c{k}_{s}")));
                        let binding = self.segs[k][s];
                        fb.if_then(hit, |b| {
                            let v = self.load_in(b, ty, binding, o);
                            b.ret(Some(v));
                        });
                    }
                    let v = self.load_in(fb, ty, self.segs[k][m - 1], o);
                    fb.ret(Some(v));
                });
                self.ld_fns.push(Some((id, ret)));
            }
        }
        if with_subs {
            for (k, instrs) in key.subs.iter().enumerate() {
                let ret = reg(instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()));
                let last = instrs.len().saturating_sub(1);
                let id = mb.add_function(&format!("sub{k}"), &one, Some(ret.clone()), |fb| {
                    let i = fb.param(0);
                    let frame = self.body_with(fb, instrs, &key.input_tys, i, i, false, true, None, &|_, _, _| None);
                    let v = match frame.t(last) {
                        Some(v) => v,
                        None => self.fail(fb, "a sub-program without instructions"),
                    };
                    fb.ret(Some(v));
                });
                self.sub_fns.push((id, ret));
            }
        }
    }

    /// Declares `wrap(v, n)`, the index into a circular dimension of `n`. `small` (a tile and its
    /// halo fit in the dimension, so indices stay within one period) allows two selects instead
    /// of two divisions.
    pub(super) fn declare_wrap(&mut self, mb: &mut ModuleBuilder, small: bool) {
        let id = mb.add_function("wrap", &[("v", KirType::I32), ("n", KirType::U32)], Some(KirType::U32), |fb| {
            let (v, n) = (fb.param(0), fb.param(1));
            let m = fb.let_val("m", fb.convert(n, KirType::I32));
            let zero = li(fb, 0);
            let wrapped = if small {
                let inner = fb.select(fb.lt(v, zero), v + m, v);
                fb.select(fb.ge(v, m), v - m, inner)
            } else {
                ((v % m) + m) % m
            };
            let index = fb.bitcast(wrapped, KirType::U32);
            fb.ret(Some(index));
        });
        self.wrap = Some(id);
    }

    /// `wrap(v, n)` of the tiled kernels.
    pub(super) fn wrap(&self, fb: &FnBuilder<'_>, v: Val, n: Val) -> Val {
        match self.wrap {
            Some(id) => fb.call(id, &[v, n], KirType::U32),
            None => self.fail(fb, "the kernel has no wrap function"),
        }
    }

    // ---- programs ----

    /// The expression of `instr`. `index` is the element a shift is relative to; `segmented`
    /// makes shifts of inputs read through the accessors `ld{k}` (which take a flat index over
    /// all segments) instead of indexing `in{k}` directly; `inplace` reads the inputs the kernel
    /// overwrites from the output buffers.
    pub(super) fn expression(&self, fb: &mut FnBuilder<'_>, f: &Frame, instr: &Instr, index: Val, segmented: bool, inplace: bool) -> Val {
        let op = instr.op;
        let shift = matches!(op.kind, OpKind::Shift(_));
        let operand = |fb: &FnBuilder<'_>, n: usize| -> Val {
            if shift && n == 0 { lu(fb, 0) } else { self.operand(fb, f, instr.args[n], op.operand_ty(n), 0) }
        };
        let (x, y, z) = (operand(fb, 0), operand(fb, 1), operand(fb, 2));
        if matches!(op.kind, OpKind::Select)
            && let Ref::Tmp(m) = instr.args[0]
            && let Some(cond) = f.b(m as usize)
        {
            return fb.select(cond, y, z);
        }
        if let Some(value) = self.operation(fb, op, x, y, z, index) {
            return value;
        }
        let OpKind::Shift(kind) = op.kind else { return self.fail(fb, "an operation without a lowering") };
        let (at, inside) = match kind {
            ShiftKind::View { map, edge } => {
                let n = self.views.iter().position(|&known| known == (map, edge));
                let at = self.view_index(fb, n, index);
                let inside = (edge == Edge::Fill).then(|| self.view_inside(fb, n, index));
                (at, inside)
            }
            ShiftKind::ParamView { id, spec } => {
                let at = self.pview_index(fb, id, index);
                let inside = (spec.edge == Edge::Fill).then(|| self.pview_inside(fb, id, index));
                (at, inside)
            }
            _ => (self.shift_index(fb, kind, y, index, z), None),
        };
        let read = match instr.args[0] {
            Ref::Input(k) if segmented => match self.ld_fns.get(k as usize).and_then(Option::as_ref) {
                Some((id, ret)) => fb.call(*id, &[at], ret.clone()),
                None => self.fail(fb, format!("input {k} has no segment loader")),
            },
            Ref::Input(k) => self.input_load(fb, op.ty, k, at, inplace),
            Ref::Sub(k) => match self.sub_fns.get(k as usize) {
                Some((id, ret)) => fb.call(*id, &[at], ret.clone()),
                None => self.fail(fb, format!("sub-program {k} has no function")),
            },
            // A shift only reads an input buffer or a sub-program.
            other => self.fail(fb, format!("a shift of {other:?}")),
        };
        match inside {
            Some(inside) => fb.select(inside, read, y),
            None => read,
        }
    }

    /// The `bool` of `instr` if it is a comparison or class test.
    fn condition(&self, fb: &mut FnBuilder<'_>, f: &Frame, instr: &Instr) -> Option<Val> {
        let op = instr.op;
        if !matches!(op.kind, OpKind::Compare(_) | OpKind::Classify(_)) {
            return None;
        }
        let operand = |fb: &FnBuilder<'_>, n: usize| self.operand(fb, f, instr.args[n], op.operand_ty(n), 0);
        let (x, y) = (operand(fb, 0), operand(fb, 1));
        self.bool_condition(fb, op, x, y)
    }

    /// Binds instruction `k` of a plain program in `frame`, with the `bool` of a test where it has one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind_instr(&self, fb: &mut FnBuilder<'_>, frame: &mut Frame, k: usize, instr: &Instr, index: Val, segmented: bool, inplace: bool, staged: Option<Val>) {
        let value = match staged {
            Some(value) => value,
            None => match self.condition(fb, frame, instr) {
                Some(cond) => {
                    let cond = fb.let_val(&format!("b{k}"), cond);
                    frame.b[k] = Some(cond);
                    fb.select(cond, lu(fb, 1), lu(fb, 0))
                }
                None => self.expression(fb, frame, instr, index, segmented, inplace),
            },
        };
        let value = fb.let_val(&format!("t{k}"), value);
        frame.t[k] = Some(value);
    }

    /// The loads of the inputs at element `i` and the instructions, as `let` bindings. `index`,
    /// `segmented` and `inplace` are those of [`Lower::expression`]; `staged(k, instr)` may
    /// replace the value of instruction `k`, and only the instructions `keep` marks (all, if
    /// `None`) are written, with the inputs they load.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn body_with(
        &self,
        fb: &mut FnBuilder<'_>,
        instrs: &[Instr],
        input_tys: &[ElemTy],
        i: Val,
        index: Val,
        segmented: bool,
        inplace: bool,
        keep: Option<&[bool]>,
        staged: &dyn Fn(usize, &Instr, &FnBuilder<'_>) -> Option<Val>,
    ) -> Frame {
        let kept = |k: usize| keep.is_none_or(|keep| keep.get(k).copied().unwrap_or(false));
        let mut frame = Frame::with_len(instrs.len());
        // Inputs read only through shifts are not loaded at `i`.
        for (k, &ty) in input_tys.iter().enumerate() {
            if instrs.iter().enumerate().any(|(at, instr)| kept(at) && reads_directly(instr, k as u32)) {
                let value = self.input_load(fb, ty, k as u32, i, inplace);
                let value = fb.let_val(&format!("a{k}"), value);
                frame.a.insert((k as u32, 0), value);
            }
        }
        for (k, instr) in instrs.iter().enumerate().filter(|&(k, _)| kept(k)) {
            let replaced = staged(k, instr, fb);
            self.bind_instr(fb, &mut frame, k, instr, index, segmented, inplace, replaced);
        }
        frame
    }

    /// The body of a lowered key (see [`Inlined`]): the index of every context, the inputs loaded
    /// at them, then the instructions, each defined after everything it reads. A `Select` the
    /// lowering marks as a branch is an `if` that computes the instructions only its own operand
    /// needs. `i` is the element and `index` the one shifts are relative to; `cell` is set in a
    /// workgroup-staged kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lowered_body(
        &self,
        fb: &mut FnBuilder<'_>,
        key: &FusedKey,
        inlined: &Inlined,
        i: Val,
        index: Val,
        inplace: bool,
        cell: Option<&StagedCell<'_>>,
        keep: Option<&[bool]>,
        held: &[u32],
    ) -> Frame {
        let kept = |n: usize| keep.is_none_or(|keep| keep.get(n).copied().unwrap_or(false));
        let mut members: FxMap<Region, Vec<u32>> = FxMap::default();
        let mut groups: FxMap<u32, Vec<u32>> = FxMap::default();
        for (n, &region) in inlined.region.iter().enumerate().filter(|&(n, _)| kept(n)) {
            members.entry(region).or_default().push(n as u32);
            if inlined.branch.get(n).copied().unwrap_or(false) {
                groups.entry(inlined.group.get(n).copied().unwrap_or(n as u32)).or_default().push(n as u32);
            }
        }
        let (has_b, bool_only) = bool_masks(key, inlined, held);
        let mut emit = Emit {
            env: self,
            key,
            inlined,
            i,
            index,
            inplace,
            cell,
            frame: Frame::with_len(inlined.instrs.len()),
            defined: vec![None; inlined.ctxs.len()],
            staged_base: Vec::new(),
            pview: PviewCache::default(),
            has_b,
            bool_only,
            members,
            groups,
        };
        emit.region(fb, Region::Top);
        emit.frame
    }
}

/// Per lowered instruction, whether its value is a `bool` built in place (a comparison, a class
/// test, or `!`, `&`, `|` of those), and whether that is all that reads it (a `Select` condition or
/// such a logic), so that its `u32` is never made. `held` are instructions read from outside.
fn bool_masks(key: &FusedKey, inlined: &Inlined, held: &[u32]) -> (Vec<bool>, Vec<bool>) {
    let count = inlined.instrs.len();
    let mut has_b = vec![false; count];
    for (n, flat) in inlined.instrs.iter().enumerate() {
        let tmp = |reference: Ref| matches!(reference, Ref::Tmp(m) if (m as usize) < n && has_b[m as usize]);
        let op = flat.op;
        let value = match op.kind {
            OpKind::Compare(_) | OpKind::Classify(_) => true,
            OpKind::Unary(UnOp::BitNot) if op.ty == ElemTy::Bool => tmp(flat.args[0]),
            OpKind::Binary(BinOp::BitAnd | BinOp::BitOr) if op.ty == ElemTy::Bool => tmp(flat.args[0]) && tmp(flat.args[1]),
            _ => false,
        };
        has_b[n] = value;
    }
    let mut read = vec![false; count];
    let mut mark = |n: u32| {
        if let Some(slot) = read.get_mut(n as usize) {
            *slot = true;
        }
    };
    for &at in &key.outputs {
        if let Some(&lowered) = inlined.main.get(at as usize) {
            mark(lowered);
        }
    }
    held.iter().for_each(|&n| mark(n));
    for ctx in &inlined.ctxs {
        for reference in ctx.args {
            if let Ref::Tmp(n) = reference {
                mark(n);
            }
        }
    }
    for (u, flat) in inlined.instrs.iter().enumerate() {
        for (j, &reference) in flat.args.iter().enumerate() {
            let Ref::Tmp(n) = reference else { continue };
            let tolerated = match flat.op.kind {
                OpKind::Select => j == 0,
                OpKind::Unary(UnOp::BitNot) | OpKind::Binary(BinOp::BitAnd | BinOp::BitOr) if flat.op.ty == ElemTy::Bool => has_b[u],
                _ => false,
            };
            if !tolerated {
                mark(n);
            }
        }
    }
    let bool_only = (0..count).map(|n| has_b[n] && !read[n]).collect();
    (has_b, bool_only)
}

/// `x % d` for a literal `d`, a mask when it is a power of two.
fn u_rem(b: &FnBuilder<'_>, x: Val, d: u32) -> Val {
    if d == 1 {
        lu(b, 0)
    } else if d.is_power_of_two() {
        x & lu(b, d - 1)
    } else {
        x % lu(b, d)
    }
}

/// `x / d` for a literal `d`, a shift when it is a power of two.
fn u_div(b: &FnBuilder<'_>, x: Val, d: u32) -> Val {
    if d == 1 {
        x
    } else if d.is_power_of_two() {
        x >> lu(b, d.trailing_zeros())
    } else {
        x / lu(b, d)
    }
}

/// Whether `instr` loads input `k` at the element being computed.
fn reads_directly(instr: &Instr, k: u32) -> bool {
    let skip = usize::from(matches!(instr.op.kind, OpKind::Shift(_)));
    instr.args.iter().skip(skip).any(|arg| *arg == Ref::Input(k))
}

/// What the parameterized views of a lowered program have divided so far, per context and divisor
/// (the flag is whether it is a power of two), so that views of one context share their quotients,
/// and the source coordinates of the contexts that are parameterized views, which their own
/// views read instead of dividing the flat index.
#[derive(Clone, Default)]
struct PviewCache {
    divs: Vec<(u32, u32, bool, Val)>,
    rems: Vec<(u32, u32, bool, Val)>,
    /// Per context: the number of its source dimensions (`ViewSpec::src_dims`) and its coordinates.
    coords: Vec<(u32, u16, [Option<Val>; 4])>,
}

/// The state of writing a lowered program (see [`Lower::lowered_body`]).
struct Emit<'e, 'k> {
    env: &'e Lower<'e>,
    key: &'k FusedKey,
    inlined: &'k Inlined,
    i: Val,
    index: Val,
    inplace: bool,
    cell: Option<&'k StagedCell<'k>>,
    frame: Frame,
    /// The index of every context declared, in the block being written (a block that ends
    /// forgets what it declared).
    defined: Vec<Option<Val>>,
    /// The base `lx + ly * width` of the staged reads of each region width, in the block being written.
    staged_base: Vec<(u32, Val)>,
    /// The quotients and remainders of the views of the block being written.
    pview: PviewCache,
    /// Per instruction, whether it has a `bool` value (see [`bool_masks`]), and whether only that is read.
    has_b: Vec<bool>,
    bool_only: Vec<bool>,
    /// The instructions of every region not written yet, in order.
    members: FxMap<Region, Vec<u32>>,
    /// The selects computed as `if`s that are kept, by group (see [`Inlined::group`]): written
    /// together, at the last of them.
    groups: FxMap<u32, Vec<u32>>,
}

impl Emit<'_, '_> {
    /// The index of context `c`.
    fn index_of(&self, c: u32) -> Val {
        if c == 0 { self.index } else { self.defined.get(c as usize).copied().flatten().unwrap_or(self.index) }
    }

    /// The operand `reference` of an instruction computed at context `at`.
    fn operand(&self, fb: &FnBuilder<'_>, reference: Ref, ty: ElemTy, at: u32) -> Val {
        self.env.operand(fb, &self.frame, reference, ty, at)
    }

    /// Declares the index of context `c`, after its parents'.
    fn define(&mut self, fb: &mut FnBuilder<'_>, c: u32) {
        let inlined = self.inlined;
        let Some(ctx) = inlined.ctxs.get(c as usize) else { return };
        if c == 0 || self.defined[c as usize].is_some() {
            return;
        }
        self.define(fb, ctx.parent);
        let from = self.index_of(ctx.parent);
        let env = self.env;
        let value = match ctx.kind {
            ShiftKind::View { map, edge } => {
                let n = env.views.iter().position(|&known| known == (map, edge));
                env.view_index(fb, n, from)
            }
            ShiftKind::ParamView { spec, .. } => {
                let (index, coords) = self.inline_pview(fb, ctx.parent, from, spec, false);
                self.pview.coords.push((c, spec.src_dims, coords));
                index
            }
            kind => {
                if let Ref::Input(k) = ctx.args[0] {
                    self.load(fb, k, ctx.parent);
                }
                let [s, total] = ctx.args.map(|reference| self.operand(fb, reference, ElemTy::U32, ctx.parent));
                env.shift_index(fb, kind, s, from, total)
            }
        };
        let value = fb.let_val(&format!("i{c}"), value);
        self.defined[c as usize] = Some(value);
    }

    /// The flat source index of the parameterized view `spec` of the index `from` of context `parent`
    /// or, with `inside_test`, whether it lies inside the source, sharing the divisions or the
    /// coordinates of `from`; and the source coordinates.
    fn inline_pview(&mut self, fb: &mut FnBuilder<'_>, parent: u32, from: Val, spec: ViewSpec, inside_test: bool) -> (Val, [Option<Val>; 4]) {
        let env = self.env;
        let (terms, raws, sizes, coords) = env.pview_axes(fb, &mut self.pview, spec, parent, from);
        (env.pview_result(fb, terms, raws, sizes, inside_test), coords)
    }

    /// Declares input `k` read at the index of context `c`.
    fn load(&mut self, fb: &mut FnBuilder<'_>, k: u32, c: u32) {
        if self.frame.a.contains_key(&(k, c)) {
            return;
        }
        self.define(fb, c);
        let ty = self.key.input_tys.get(k as usize).copied().unwrap_or(ElemTy::F32);
        let (at, name) = if c == 0 { (self.i, format!("a{k}")) } else { (self.index_of(c), format!("a{k}_{c}")) };
        let value = self.env.input_load(fb, ty, k, at, self.inplace);
        let value = fb.let_val(&name, value);
        self.frame.a.insert((k, c), value);
    }

    /// Writes the instructions computed in `region`.
    fn region(&mut self, fb: &mut FnBuilder<'_>, region: Region) {
        for n in self.members.remove(&region).unwrap_or_default() {
            self.instruction(fb, n as usize);
        }
    }

    /// The offset of the cell instruction `n` (a read of a staged sub-program through `kind`,
    /// computed at context `at`) reads from the cell of its program: the starts of the
    /// translation views on the way and of the read itself, where the lowering holds them.
    fn staged_offset(&self, n: usize, kind: ShiftKind, at: u32) -> Option<[i32; 2]> {
        if let Some(read) = self.inlined.staged_reads.get(n) {
            return *read;
        }
        let ShiftKind::View { map, .. } = kind else {
            return None;
        };
        let base = self.inlined.translation(at);
        Some([map.axes[0].start + base[0], map.axes[1].start + base[1]])
    }

    /// Writes instruction `n`.
    fn instruction(&mut self, fb: &mut FnBuilder<'_>, n: usize) {
        let inlined = self.inlined;
        let env = self.env;
        let Some(flat) = inlined.instrs.get(n) else { return };
        // A read from workgroup memory needs no index: the cell is `(lx, ly)` plus offsets.
        let staged_read = match (flat.op.kind, flat.args[0]) {
            (OpKind::Shift(_), Ref::Sub(s)) => self.cell.is_some_and(|cell| cell.staging.find(s).is_some()),
            _ => false,
        };
        if !staged_read {
            self.define(fb, flat.at);
        }
        if inlined.branch.get(n).copied().unwrap_or(false) {
            let anchor = inlined.group.get(n).copied().unwrap_or(n as u32);
            return self.branch(fb, anchor, n);
        }
        let op = flat.op;
        if self.has_b.get(n).copied().unwrap_or(false) {
            for reference in flat.args.iter().take(op.kind.arity()) {
                if let Ref::Input(k) = *reference {
                    self.load(fb, k, flat.at);
                }
            }
            let flag = |this: &Self, j: usize| match flat.args[j] {
                Ref::Tmp(m) => this.frame.b(m as usize),
                _ => None,
            };
            let cond = match op.kind {
                OpKind::Compare(_) | OpKind::Classify(_) => {
                    let [x, y, _] = [0, 1, 2].map(|j| self.operand(fb, flat.args[j], op.operand_ty(j), flat.at));
                    env.bool_condition(fb, op, x, y)
                }
                OpKind::Unary(_) => flag(self, 0).map(|b0| fb.unary(IrUnOp::Not, b0)),
                OpKind::Binary(BinOp::BitAnd) => flag(self, 0).zip(flag(self, 1)).map(|(b0, b1)| fb.and(b0, b1)),
                OpKind::Binary(BinOp::BitOr) => flag(self, 0).zip(flag(self, 1)).map(|(b0, b1)| fb.or(b0, b1)),
                _ => None,
            };
            let Some(cond) = cond else {
                env.fail(fb, format!("the test t{n} is read before it is computed"));
                return;
            };
            let b = fb.let_val(&format!("b{n}"), cond);
            self.frame.b[n] = Some(b);
            let t = if self.bool_only[n] {
                lu(fb, 0)
            } else {
                let word = fb.select(b, lu(fb, 1), lu(fb, 0));
                fb.let_val(&format!("t{n}"), word)
            };
            self.frame.t[n] = Some(t);
            return;
        }
        let value = match op.kind {
            OpKind::Shift(kind) => {
                if !staged_read {
                    self.define(fb, flat.child);
                }
                let at = self.index_of(flat.child);
                let read = match flat.args[0] {
                    Ref::Input(k) => env.input_load(fb, op.ty, k, at, self.inplace),
                    Ref::Tmp(v) => match self.frame.t(v as usize) {
                        Some(value) => value,
                        None => env.fail(fb, format!("t{v} is read before it is computed")),
                    },
                    Ref::Sub(s) => match self.cell.zip(self.staged_offset(n, kind, flat.at)) {
                        // A read from workgroup memory: the region's cell `(lx, ly)` plus the
                        // offsets of the contexts the read is in and of the read itself.
                        Some((cell, [ox, oy])) => match (cell.staging.find(s), cell.shared.get(s as usize).copied().flatten()) {
                            (Some(entry), Some(shared)) => {
                                let (dx, dy) = (ox - entry.lo[0], oy - entry.lo[1]);
                                let index = if entry.lo[1] == 0 && entry.hi[1] == 0 {
                                    fb.bitcast(if dx == 0 { cell.lx } else { cell.lx + li(fb, dx) }, KirType::U32)
                                } else {
                                    let rw = entry.width();
                                    let base = match self.staged_base.iter().find(|(w, _)| *w == rw) {
                                        Some(&(_, base)) => base,
                                        None => {
                                            let base = fb.let_val(&format!("sb_{rw}"), cell.lx + cell.ly * li(fb, rw as i32));
                                            self.staged_base.push((rw, base));
                                            base
                                        }
                                    };
                                    let delta = dx + dy * (rw as i32);
                                    fb.bitcast(if delta == 0 { base } else { base + li(fb, delta) }, KirType::U32)
                                };
                                fb.shared_load(shared, index)
                            }
                            _ => env.fail(fb, format!("sub-program {s} is not staged")),
                        },
                        None => env.fail(fb, format!("sub-program {s} is read without staging")),
                    },
                    other => env.fail(fb, format!("a shift of {other:?}")),
                };
                match kind {
                    ShiftKind::View { map, edge: Edge::Fill } => {
                        let fill = self.operand(fb, flat.args[1], op.operand_ty(1), flat.at);
                        let view = env.views.iter().position(|&known| known == (map, Edge::Fill));
                        let inside = env.view_inside(fb, view, self.index_of(flat.at));
                        fb.select(inside, read, fill)
                    }
                    ShiftKind::ParamView { spec, .. } if spec.edge == Edge::Fill => {
                        let fill = self.operand(fb, flat.args[1], op.operand_ty(1), flat.at);
                        // An undefined context's index is the element's: no sharing with it.
                        let parent = if staged_read { u32::MAX } else { flat.at };
                        let from = self.index_of(flat.at);
                        let (inside, _) = self.inline_pview(fb, parent, from, spec, true);
                        fb.select(inside, read, fill)
                    }
                    _ => read,
                }
            }
            // A range that is a coordinate of the context (see `Gen::ParamRange`), read from the
            // coordinates of the context's view where the lowering has them.
            OpKind::Generate(Gen::ParamRange { axis: Some(a), .. })
                if let Some(coord) = self.pview.coords.iter().find(|&&(c, ..)| c == flat.at).and_then(|&(.., coords)| coords[a as usize]) =>
            {
                env.shaped(fb, op.ty, fb.bitcast(coord, KirType::U32))
            }
            _ => {
                for reference in flat.args.iter().take(op.kind.arity()) {
                    if let Ref::Input(k) = *reference {
                        self.load(fb, k, flat.at);
                    }
                }
                let [x, y, z] = [0, 1, 2].map(|j| self.operand(fb, flat.args[j], op.operand_ty(j), flat.at));
                let condition = match flat.args[0] {
                    Ref::Tmp(m) if matches!(op.kind, OpKind::Select) => self.frame.b(m as usize),
                    _ => None,
                };
                match condition {
                    Some(cond) => fb.select(cond, y, z),
                    None => match env.operation(fb, op, x, y, z, self.index_of(flat.at)) {
                        Some(value) => value,
                        None => env.fail(fb, "an operation without a lowering"),
                    },
                }
            }
        };
        let value = fb.let_val(&format!("t{n}"), value);
        self.frame.t[n] = Some(value);
    }

    /// Writes the group `anchor` of `Select`s as one `if`, each branch computing the
    /// instructions only its operands need before it takes them, if `n` is the last of its kept
    /// members (the rest are written there).
    fn branch(&mut self, fb: &mut FnBuilder<'_>, anchor: u32, n: usize) {
        let inlined = self.inlined;
        let Some(members) = self.groups.get(&anchor).cloned() else { return };
        if members.last().is_none_or(|&last| last as usize != n) {
            return;
        }
        let Some(flat) = inlined.instrs.get(n) else { return };
        let mask = self.operand(fb, flat.args[0], ElemTy::U32, flat.at);
        let locals: Vec<(usize, _)> = members.iter().map(|&m| (m as usize, fb.var_decl(&format!("t{m}"), reg(inlined.instrs[m as usize].op.out_ty()), None))).collect();
        let zero = lu(fb, 0);
        let taken = match flat.args[0] {
            Ref::Tmp(m) => self.frame.b(m as usize),
            _ => None,
        }
        .unwrap_or_else(|| fb.ne(mask, zero));
        let outer = (self.defined.clone(), self.frame.a.clone(), self.staged_base.clone(), self.pview.clone());
        let me = RefCell::new(&mut *self);
        let arm = |b: &mut FnBuilder<'_>, side: Region, slot: usize| {
            let mut guard = me.borrow_mut();
            let this: &mut Emit<'_, '_> = &mut guard;
            this.region(b, side);
            for &(m, local) in &locals {
                let flat = &inlined.instrs[m];
                if let Ref::Input(k) = flat.args[slot] {
                    this.load(b, k, flat.at);
                }
                let value = this.operand(b, flat.args[slot], flat.op.operand_ty(slot), flat.at);
                b.assign_var(local, value);
            }
            (this.defined, this.frame.a, this.staged_base, this.pview) = outer.clone();
        };
        fb.if_else(taken, |b| arm(b, Region::Taken(anchor), 1), |b| arm(b, Region::Skipped(anchor), 2));
        for (m, local) in locals {
            self.frame.t[m] = Some(fb.read_var(local));
        }
    }
}

/// The double-float library, the generators, the edge functions and the numeric series the
/// programs call, lowered once per distinct text (`None` if they call nothing). The guard of the
/// double-float library is the constant 0: `luma_o` becomes an optimisation barrier of the
/// backend.
pub(super) fn helper_library(programs: &[&[Instr]], tuning: &Tuning, edges: bool, narrow_div: bool) -> Result<Option<Arc<Library>>> {
    let instrs = || programs.iter().flat_map(|program| program.iter());
    let mut text = String::new();
    // The casts and the tests of the class of a value are built in place.
    if instrs().any(|instr| calls_df_library(instr) && !matches!(instr.op.kind, OpKind::Classify(_) | OpKind::Cast(_))) {
        text.push_str(&df_library("0u", tuning.df_fma, tuning.df_sloppy_add));
        if instrs().any(calls_df_math) {
            text.push_str(DF_MATH_LIBRARY);
        }
    }
    text.push_str(&generator_helpers(programs.iter().copied()));
    if edges {
        text.push_str(EDGE_HELPERS);
    }
    if narrow_div {
        text.push_str(NARROW_DIV);
    }
    if instrs().any(|instr| matches!(instr.op.kind, OpKind::Unary(UnOp::Log1p)) && instr.op.ty == ElemTy::F32) {
        text.push_str(&format!("fn luma_log1p(x: f32) -> f32 {{ return {}; }}\n", UnOp::Log1p.expr()));
    }
    if instrs().any(|instr| matches!(instr.op.kind, OpKind::Unary(UnOp::Expm1)) && instr.op.ty == ElemTy::F32) {
        text.push_str(&format!("fn luma_expm1(x: f32) -> f32 {{ return {}; }}\n", UnOp::Expm1.expr()));
    }
    if instrs().any(|instr| matches!(instr.op.kind, OpKind::Binary(BinOp::Hypot)) && instr.op.ty == ElemTy::F32) {
        text.push_str(&format!("fn luma_hypot(x: f32, y: f32) -> f32 {{ return {}; }}\n", BinOp::Hypot.expr()));
    }
    if text.is_empty() { Ok(None) } else { library(&text).map(Some) }
}

// ---------------------------------------------------------------------------------------------
// The store kernels
// ---------------------------------------------------------------------------------------------

/// What a store kernel is, by its tail and its key (the order is that of the precedence).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    /// [`Tail::ScatterAdd`] accumulating in workgroup memory.
    ScatterPrivate,
    /// Sub-programs staged in workgroup tiles with halos (see [`Tile`]).
    Tiled,
    /// Sub-programs staged in workgroup memory (see [`Staging`]).
    Staged,
    /// [`Tail::Stack`] writing through workgroup memory.
    StackStaged,
    /// A region write of `bool`s as words of runs.
    Runs,
    /// A store with `bool` outputs packed by the workgroup.
    Packed,
    Plain,
}

/// The program of a store kernel, computed: its instruction values.
struct Prog<'a> {
    frame: Frame,
    /// The lowered program the values come from, if the key is lowered.
    map: Option<&'a Inlined>,
}

struct Build<'a> {
    key: &'a FusedKey,
    tuning: &'a Tuning,
    lowered: Option<&'a Lowered>,
    inlined: Option<&'a Inlined>,
    env: Lower<'a>,
    /// The instructions whose values the kernel's output buffers hold.
    outputs: Vec<usize>,
    segmented: bool,
    based: bool,
    wg: u32,
    /// The workgroup arrays of the variant, in declaration order.
    shared: Vec<u32>,
    /// For a staged kernel, per sub-program, its workgroup array.
    shared_w: Vec<Option<u32>>,
}

/// Elements every invocation of the private scatter loads before it adds any of them.
///
/// Measured on an RTX 3090 (histogram of 256 bins of 10M `f32`, 0.0632 ms with one element): 2
/// elements 0.059, 4 0.058, 8 0.053 with the 1024 workgroups of the time, and 1 to 16 the same
/// 0.0502 at the workgroups of `launch::PRIVATE_GROUPS`; 16 was slower where all the elements
/// fall in one bin (0.083 against 0.077 ms).
const PRIVATE_UNROLL: u32 = 4;

/// What a private accumulator starts as for `op` on elements of type `ty`, as bits (the merge
/// skips the elements still holding it).
fn seed_bits(ty: ElemTy, op: AtomicOp) -> u32 {
    match (ty, op) {
        (ElemTy::F32, AtomicOp::Add) | (ElemTy::U32, AtomicOp::Add | AtomicOp::Max) => 0,
        (ElemTy::F32, AtomicOp::Min) => 0x7F80_0000,
        (ElemTy::F32, AtomicOp::Max) => 0xFF80_0000,
        (ElemTy::I32, AtomicOp::Add) => 0,
        (ElemTy::I32, AtomicOp::Min) => 0x7FFF_FFFF,
        (ElemTy::I32, AtomicOp::Max) => 0x8000_0000,
        _ => 0xFFFF_FFFF,
    }
}

/// The scalar the atomic accumulators of elements of type `ty` are made of: floats are native
/// atomics, integers their own kind.
fn atomic_scalar(ty: ElemTy) -> ScalarTy {
    match ty {
        ElemTy::F32 => ScalarTy::F32,
        ElemTy::I32 => ScalarTy::I32,
        _ => ScalarTy::U32,
    }
}

fn atomic_op(op: AtomicOp) -> IrAtomicOp {
    match op {
        AtomicOp::Add => IrAtomicOp::Add,
        AtomicOp::Min => IrAtomicOp::Min,
        AtomicOp::Max => IrAtomicOp::Max,
    }
}

/// The kernel of the pure tail of `key` (a store, a region write, a scatter, an atomic scatter
/// or a stack), with the workgroup shapes `tuning` gives. `segment_len` (a power of two) is the
/// elements per segment of the arrays the segmented variant (`key.spans` not empty) runs on.
/// `lowered` is the lowering of `key` (see `jit::inline`) if it [`inlines`](FusedKey::inlines):
/// the body then computes every sub-program at the indices it is read at, and no `sub{k}`
/// functions exist; for a [`Lowered::Staged`] key (`key.staging` set) the staged sub-programs are
/// computed into workgroup memory instead.
pub(super) fn store_module(key: &FusedKey, tuning: &Tuning, segment_len: u32, lowered: Option<&Lowered>) -> Result<KernelModule> {
    let inlined = match lowered {
        Some(Lowered::Plain(inlined)) => Some(inlined),
        Some(Lowered::Staged(programs)) => Some(&programs.main),
        None => None,
    };
    let wg = tuning.elementwise_workgroup;
    let segmented = !key.spans.is_empty();
    let slots = |k: usize| key.spans.get(k).copied().unwrap_or(0);
    // A product reads its operands at their own element indices: nothing is relative to a segment.
    let based = generates(key) && !segmented;
    let uniform = (!key.scalar_tys.is_empty() && reads_scalars(key)) || segmented || based || key.runs;
    let outputs: Vec<usize> = match key.tail {
        Tail::Store | Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } => key.outputs.iter().map(|&at| at as usize).collect(),
        Tail::Scatter { .. } | Tail::ParamScatter { .. } | Tail::ScatterAdd { .. } | Tail::Stack { .. } | Tail::ParamStack { .. } => {
            key.outputs.iter().take(1).map(|&at| at as usize).collect()
        }
        _ => return Err(LumaError::Gpu(format!("the tail {:?} is not a store kernel", key.tail))),
    };
    let out_elem = |at: usize| key.instrs.get(at).map_or(ElemTy::F32, |instr| instr.op.out_ty());

    let views = match lowered {
        Some(Lowered::Plain(inlined)) => views_of_inlined(key, &[inlined]),
        Some(Lowered::Staged(programs)) => views_of_inlined(key, &programs.bodies().collect::<Vec<_>>()),
        None => views_of(key),
    };
    let mut pviews = if key.params.is_some() {
        let bodies: Vec<&Inlined> = match lowered {
            Some(Lowered::Plain(inlined)) => vec![inlined],
            Some(Lowered::Staged(programs)) => programs.bodies().collect(),
            None => Vec::new(),
        };
        params::views_used(key, &bodies)
    } else {
        Vec::new()
    };
    let programs: Vec<&[Instr]> = std::iter::once(key.instrs.as_slice()).chain(key.subs.iter().map(Vec::as_slice)).collect();
    let lib = helper_library(&programs, tuning, !views.is_empty() || !pviews.is_empty(), key.params.is_some())?;
    // The views of a lowered program are written in place, but for the one a region write goes through.
    if lowered.is_some() {
        pviews.retain(|&(id, _)| matches!(key.tail, Tail::ParamStoreMapped { id: store, .. } if store == id));
    }

    let mut build = Build {
        key,
        tuning,
        lowered,
        inlined,
        env: Lower::new(lib.clone(), &key.in_place),
        outputs,
        segmented,
        based,
        wg,
        shared: Vec::new(),
        shared_w: Vec::new(),
    };
    let size = match build.variant() {
        Variant::Tiled | Variant::Staged => WORKGROUP,
        _ => wg,
    };
    let mut mb = ModuleBuilder::new(key.label(), "main", [size, 1, 1]);

    // The bindings, in the order of the arguments of the launch.
    for (j, &at) in build.outputs.iter().enumerate() {
        // An atomic scatter binds its target as atomics.
        let elem = match key.tail {
            Tail::ScatterAdd { .. } => KirType::Atomic(atomic_scalar(out_elem(at))),
            _ => write_elem(out_elem(at)),
        };
        build.env.outs.push(mb.add_storage_read_write(&format!("out{j}"), elem));
    }
    for (k, &ty) in key.input_tys.iter().enumerate() {
        build.env.ins.push(None);
        build.env.segs.push(Vec::new());
        if key.in_place.iter().any(|&(paired, _)| paired == k as u32) {
            continue;
        }
        if !segmented || direct(&key.instrs, k as u32) {
            build.env.ins[k] = Some(mb.add_storage_read(&format!("in{k}"), reg(ty)));
        }
        for s in 0..slots(k) {
            let binding = mb.add_storage_read(&format!("in{k}_{s}"), reg(ty));
            build.env.segs[k].push(binding);
        }
    }
    if uniform {
        let mut fields: Vec<(String, KirType)> = Vec::new();
        if segmented {
            fields.push(("base".to_owned(), KirType::U32));
            fields.push(("total".to_owned(), KirType::U32));
            for k in 0..key.input_tys.len() {
                for s in 0..slots(k) {
                    fields.push((format!("c{k}_{s}"), KirType::U32));
                }
            }
        } else if based {
            fields.push(("base".to_owned(), KirType::U32));
        }
        if key.runs {
            fields.push(("run".to_owned(), KirType::U32));
            fields.push(("words".to_owned(), KirType::U32));
        }
        for (k, &ty) in key.scalar_tys.iter().enumerate() {
            fields.push((format!("s{k}"), reg(ty)));
        }
        let uni = Uni::declare(&mut mb, fields);
        build.env.set_uniform(uni);
    }

    build.env.declare_functions(&mut mb, key, &views, &pviews, segmented.then_some(segment_len), key.tile.is_none() && lowered.is_none());
    if let Variant::Tiled = build.variant()
        && let Some(tile) = &key.tile
    {
        build.env.declare_wrap(&mut mb, tile.slots.map_or_else(|| tile.fits(), |slots| slots.small));
    }
    build.declare_shared(&mut mb);
    {
        let mut fb = mb.entry_builder();
        build.entry(&mut fb);
        fb.finish();
    }
    if let Some(message) = build.env.error() {
        return Err(LumaError::Gpu(format!("{}: {message}", key.label())));
    }
    Ok(finish(mb, lib.as_deref()))
}

impl<'a> Build<'a> {
    fn out_elem(&self, at: usize) -> ElemTy {
        self.key.instrs.get(at).map_or(ElemTy::F32, |instr| instr.op.out_ty())
    }

    fn variant(&self) -> Variant {
        let key = self.key;
        if let Tail::ScatterAdd { private: true, .. } = key.tail {
            return Variant::ScatterPrivate;
        }
        if key.tile.is_some() {
            return Variant::Tiled;
        }
        if let (Some(Lowered::Staged(_)), Some(_)) = (self.lowered, &key.staging) {
            return Variant::Staged;
        }
        if let Tail::Stack { .. } = key.tail
            && let Some(&first) = self.outputs.first()
            && stack_stages(key, self.wg, self.out_elem(first))
        {
            return Variant::StackStaged;
        }
        if key.runs {
            return Variant::Runs;
        }
        if key.tail == Tail::Store && key.writes_bits() {
            return Variant::Packed;
        }
        Variant::Plain
    }

    /// The numbers of the `bool` outputs.
    fn bit_outputs(&self) -> Vec<usize> {
        self.outputs.iter().enumerate().filter(|&(_, &at)| self.out_elem(at) == ElemTy::Bool).map(|(j, _)| j).collect()
    }

    /// Declares the workgroup memory of the variant.
    fn declare_shared(&mut self, mb: &mut ModuleBuilder) {
        let key = self.key;
        let ballot = self.tuning.subgroup32;
        let words = self.wg / 32;
        match self.variant() {
            Variant::ScatterPrivate => {
                let Tail::ScatterAdd { len, .. } = key.tail else { return };
                let bins = (len as usize).next_multiple_of(self.wg as usize) as u32;
                let ty = self.outputs.first().map_or(ElemTy::F32, |&at| self.out_elem(at));
                self.shared.push(mb.add_shared("acc", KirType::Atomic(atomic_scalar(ty)), bins));
            }
            Variant::Tiled => {
                let Some(tile) = &key.tile else { return };
                let (tx, ty) = tile.tile;
                for (k, instrs) in key.subs.iter().enumerate() {
                    let (lw, lh) = (tx + 2 * tile.need[k].0, ty + 2 * tile.need[k].1);
                    let elem = reg(instrs.last().map_or(ElemTy::F32, |instr| instr.op.out_ty()));
                    self.shared.push(mb.add_shared(&format!("w{k}"), elem, lw * lh));
                }
            }
            Variant::Staged => {
                let (Some(Lowered::Staged(programs)), Some(staging)) = (self.lowered, &key.staging) else { return };
                self.shared_w = vec![None; key.subs.len()];
                let level_of = |idx: usize| {
                    staging.levels.windows(2).position(|w| (w[0] as usize..w[1] as usize).contains(&idx)).unwrap_or(0) as u32
                };
                let main_level = staging.levels.len().saturating_sub(1) as u32;
                let items: Vec<(ElemTy, u32, u32, u32)> = staging
                    .subs
                    .iter()
                    .enumerate()
                    .map(|(idx, entry)| {
                        let ty = key.subs.get(entry.sub as usize).and_then(|program| program.last()).map_or(ElemTy::F32, |instr| instr.op.out_ty());
                        let prod = level_of(idx);
                        let mut last_use = prod;
                        if programs.main.instrs.iter().any(|flat| flat.args[0] == Ref::Sub(entry.sub)) {
                            last_use = main_level;
                        } else {
                            for group in &programs.groups {
                                if let Some(&first) = group.members.first()
                                    && group.body.instrs.iter().any(|flat| flat.args[0] == Ref::Sub(entry.sub))
                                {
                                    last_use = last_use.max(level_of(first));
                                }
                            }
                        }
                        (ty, entry.cells(), prod, last_use)
                    })
                    .collect();
                let (buffers, assignment) = super::staging::color_buffers(&items);
                let ids: Vec<u32> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, &(ty, cells))| mb.add_shared(&format!("w{i}"), reg(ty), cells))
                    .collect();
                for (entry, &buf) in staging.subs.iter().zip(&assignment) {
                    if let Some(slot) = self.shared_w.get_mut(entry.sub as usize) {
                        *slot = Some(ids[buf]);
                    }
                }
            }
            Variant::StackStaged => {
                let Tail::Stack { total, .. } = key.tail else { return };
                let ty = self.outputs.first().map_or(ElemTy::F32, |&at| self.out_elem(at));
                self.shared.push(mb.add_shared("luma_stage", reg(ty), self.wg * total));
            }
            Variant::Runs | Variant::Packed if !ballot => {
                if self.variant() == Variant::Runs {
                    self.shared.push(mb.add_shared("luma_valid", KirType::Atomic(ScalarTy::U32), words));
                }
                for j in self.bit_outputs() {
                    self.shared.push(mb.add_shared(&format!("luma_bits{j}"), KirType::Atomic(ScalarTy::U32), words));
                }
            }
            Variant::Runs | Variant::Packed | Variant::Plain => {}
        }
    }

    fn entry(&self, fb: &mut FnBuilder<'_>) {
        match self.variant() {
            Variant::ScatterPrivate => self.scatter_private(fb),
            Variant::Tiled => self.tiled(fb),
            Variant::Staged => self.staged(fb),
            Variant::StackStaged => self.stack_staged(fb),
            Variant::Runs => self.runs(fb),
            Variant::Packed => self.packed(fb),
            Variant::Plain => self.plain(fb),
        }
    }

    /// `gi = p.base + i`: the element's place in the whole array, for the kernels that run per
    /// segment or generate from the index.
    fn global(&self, fb: &mut FnBuilder<'_>, i: Val) -> Val {
        if self.segmented || self.based {
            let base = self.env.named(fb, "base");
            fb.let_val("gi", base + i)
        } else {
            i
        }
    }

    /// The program of the kernel at element `i`.
    fn program(&self, fb: &mut FnBuilder<'_>, i: Val, index: Val) -> Prog<'a> {
        match self.inlined {
            Some(inlined) => Prog { frame: self.env.lowered_body(fb, self.key, inlined, i, index, true, None, None, &[]), map: Some(inlined) },
            None => Prog {
                frame: self.env.body_with(fb, &self.key.instrs, &self.key.input_tys, i, index, self.segmented, true, None, &|_, _, _| None),
                map: None,
            },
        }
    }

    /// The value of main-program instruction `at`.
    fn value(&self, fb: &FnBuilder<'_>, prog: &Prog<'_>, at: usize) -> Val {
        let n = prog.map.and_then(|inlined| inlined.main.get(at)).map_or(at, |&lowered| lowered as usize);
        match prog.frame.t(n) {
            Some(value) => value,
            None => self.env.fail(fb, format!("the value of instruction {at} is not computed")),
        }
    }

    /// `if (i >= luma.size.x) { return; }`
    fn guard(&self, fb: &mut FnBuilder<'_>, i: Val) {
        let outside = fb.ge(i, fb.grid_size(0));
        fb.if_then(outside, |b| b.ret(None));
    }

    /// The stores of the tail of the kernel for element `i`.
    fn emit_stores(&self, fb: &mut FnBuilder<'_>, prog: &Prog<'_>, i: Val) {
        let key = self.key;
        let env = &self.env;
        let at0 = self.outputs.first().copied().unwrap_or(0);
        match key.tail {
            Tail::StoreMapped(_) | Tail::ParamStoreMapped { .. } => {
                let Some(view) = env.mapped_view(key.tail) else {
                    env.fail(fb, "a region write without its view");
                    return;
                };
                let o = fb.call(view, &[i], KirType::U32);
                let o = fb.let_val("o", o);
                for (j, &at) in self.outputs.iter().enumerate() {
                    let value = self.value(fb, prog, at);
                    env.store(fb, self.out_elem(at), env.outs[j], o, value);
                }
            }
            Tail::Scatter { .. } | Tail::ParamScatter { .. } | Tail::ScatterAdd { private: false, .. } => {
                let len = match key.tail {
                    Tail::ParamScatter { len_slot } => env.scalar(fb, len_slot),
                    Tail::Scatter { len } => lu(fb, len),
                    Tail::ScatterAdd { len, len_slot, .. } => match len_slot {
                        Some(slot) => env.scalar(fb, slot),
                        None => lu(fb, len),
                    },
                    _ => lu(fb, 0),
                };
                let (stored, index) = (key.outputs.first().copied().unwrap_or(0), key.outputs.get(1).copied().unwrap_or(0));
                let j = self.value(fb, prog, index as usize);
                let value = self.value(fb, prog, stored as usize);
                let j = fb.let_val("j", j);
                let ty = self.out_elem(at0);
                let binding = env.outs[0];
                let inside = fb.lt(j, len);
                if let Tail::ScatterAdd { op, .. } = key.tail {
                    fb.if_then(inside, |b| b.atomic_rmw(None, atomic_op(op), atomic_scalar(ty), storage_target(binding, j), value));
                } else {
                    fb.if_then(inside, |b| env.store(b, ty, binding, j, value));
                }
            }
            Tail::Stack { total, first, block } => {
                let o = if block == 1 {
                    i * lu(fb, total) + lu(fb, first)
                } else {
                    let q = fb.let_val("q", i / lu(fb, block));
                    q * lu(fb, (u64::from(total) * u64::from(block)) as u32) + (i - q * lu(fb, block)) + lu(fb, (u64::from(first) * u64::from(block)) as u32)
                };
                let o = fb.let_val("o", o);
                for (j, &at) in key.outputs.iter().enumerate() {
                    let at_j = o + lu(fb, (j as u64 * u64::from(block)) as u32);
                    let value = self.value(fb, prog, at as usize);
                    env.store(fb, self.out_elem(at0), env.outs[0], at_j, value);
                }
            }
            Tail::ParamStack { total, first, block_slot, div, div_slot } => {
                let b = env.scalar(fb, block_slot);
                let b = fb.let_val("b", b);
                let quotient = match div {
                    DivRemSpec::None => i,
                    DivRemSpec::Pow2 => i >> env.scalar(fb, div_slot),
                    DivRemSpec::Runtime { wide } => env.udiv(fb, i, div_slot, wide),
                };
                let q = fb.let_val("q", quotient);
                let o = q * (lu(fb, total) * b) + (i - q * b) + (lu(fb, first) * b);
                let o = fb.let_val("o", o);
                for (j, &at) in key.outputs.iter().enumerate() {
                    let at_j = o + lu(fb, j as u32) * b;
                    let value = self.value(fb, prog, at as usize);
                    env.store(fb, self.out_elem(at0), env.outs[0], at_j, value);
                }
            }
            _ => {
                for (j, &at) in self.outputs.iter().enumerate() {
                    let value = self.value(fb, prog, at);
                    env.store(fb, self.out_elem(at), env.outs[j], i, value);
                }
            }
        }
    }

    /// The plain store kernel: one element per invocation, or `key.coarsen` elements spaced by
    /// the number of invocations of the launch (see `jit::coarsen`): the elements of one warp stay
    /// contiguous, and the loads of all of them are issued before the first store.
    fn plain(&self, fb: &mut FnBuilder<'_>) {
        let i = element_index(fb, self.wg);
        self.guard(fb, i);
        let index = self.global(fb, i);
        let prog = self.program(fb, i, index);
        let c = self.key.coarsen;
        if c <= 1 {
            self.emit_stores(fb, &prog, i);
            return;
        }
        debug_assert!(super::coarsen::coarsenable(self.key), "a coarsened kernel of a variant that cannot be");
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let (nx, ny) = (fb.field(nwg, 0, KirType::U32), fb.field(nwg, 1, KirType::U32));
        let stride = fb.let_val("stride", nx * ny * lu(fb, self.wg));
        let last = fb.grid_size(0) - lu(fb, 1);
        let mut rest = Vec::with_capacity(c as usize - 1);
        for k in 1..u32::from(c) {
            let e = fb.let_val("e", i + stride * lu(fb, k));
            let inside = fb.let_val("ok", fb.lt(e, fb.grid_size(0)));
            // Elements past the end read the last one, so that the loads need no branch; their
            // stores are skipped.
            let at = fb.let_val("ec", fb.select(inside, e, last));
            let index = self.global(fb, at);
            rest.push((e, inside, self.program(fb, at, index)));
        }
        self.emit_stores(fb, &prog, i);
        for (e, inside, prog) in &rest {
            fb.if_then(*inside, |b| self.emit_stores(b, prog, *e));
        }
    }

    /// An atomic scatter whose workgroups first accumulate into workgroup memory (the target is
    /// small next to the number of elements) and merge it into the buffer once.
    fn scatter_private(&self, fb: &mut FnBuilder<'_>) {
        let key = self.key;
        let (wg, acc) = (self.wg, self.shared.first().copied().unwrap_or(0));
        let Tail::ScatterAdd { len, op, .. } = key.tail else { return };
        let (stored, slot) = (key.outputs.first().copied().unwrap_or(0), key.outputs.get(1).copied().unwrap_or(0));
        let ty = self.out_elem(self.outputs.first().copied().unwrap_or(0));
        let bins = (len as usize).next_multiple_of(wg as usize) as u32;
        let seed = seed_bits(ty, op);
        let scalar = atomic_scalar(ty);
        let literal = move |fb: &FnBuilder<'_>, bits: u32| match scalar {
            ScalarTy::F32 => lf(fb, f32::from_bits(bits)),
            ScalarTy::I32 => li(fb, bits as i32),
            _ => lu(fb, bits),
        };
        let lid = fb.builtin(BuiltinVar::LocalInvocationId);
        let wid = fb.builtin(BuiltinVar::WorkgroupId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let lx = fb.field(lid, 0, KirType::U32);
        let (wx, wy, nx, ny) = (fb.field(wid, 0, KirType::U32), fb.field(wid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32), fb.field(nwg, 1, KirType::U32));
        let group = fb.let_val("group", wx + wy * nx);
        fb.for_range("b", lx, lu(fb, bins), lu(fb, wg), |b, bin| {
            let seed = literal(b, seed);
            b.atomic_store(shared_target(acc, bin), seed);
        });
        fb.barrier(BarrierScope::Workgroup);
        let (start, size, step) = (group * lu(fb, wg) + lx, fb.grid_size(0), nx * ny * lu(fb, wg));
        let last = size - lu(fb, 1);
        fb.for_range("i", start, size, step * lu(fb, PRIVATE_UNROLL), |b, i| {
            // The loads of all the elements come before the first atomic; elements past the end
            // read the last one (no branch around the loads) and add nothing.
            let mut elements = Vec::with_capacity(PRIVATE_UNROLL as usize);
            for k in 0..PRIVATE_UNROLL {
                let e = if k == 0 { i } else { b.let_val("e", i + step * lu(b, k)) };
                let at = if k == 0 { e } else { b.let_val("ec", b.select(b.lt(e, size), e, last)) };
                let index = self.global(b, at);
                let prog = self.program(b, at, index);
                let j = self.value(b, &prog, slot as usize);
                let value = self.value(b, &prog, stored as usize);
                let j = b.let_val("j", j);
                let value = b.let_val("v", value);
                elements.push((e, j, value));
            }
            for (e, j, value) in elements {
                let inside = b.and(b.lt(e, size), b.lt(j, lu(b, len)));
                b.if_then(inside, |b| b.atomic_rmw(None, atomic_op(op), scalar, shared_target(acc, j), value));
            }
        });
        fb.barrier(BarrierScope::Workgroup);
        let target = self.env.outs.first().copied().unwrap_or(0);
        fb.for_range("b", lx, lu(fb, len), lu(fb, wg), |b, bin| {
            let a = b.atomic_load(shared_target(acc, bin), scalar);
            let a = b.let_val("a", a);
            let bits = if scalar == ScalarTy::U32 { a } else { b.bitcast(a, KirType::U32) };
            let touched = b.ne(bits, lu(b, seed));
            b.if_then(touched, |b| b.atomic_rmw(None, atomic_op(op), scalar, storage_target(target, bin), a));
        });
    }

    /// A stack that writes through workgroup memory: each invocation computes its cell's `total`
    /// values into workgroup memory, then the workgroup writes its contiguous stretch of the
    /// result, coalesced.
    fn stack_staged(&self, fb: &mut FnBuilder<'_>) {
        let key = self.key;
        let Tail::Stack { total, .. } = key.tail else { return };
        let (wg, stage) = (self.wg, self.shared.first().copied().unwrap_or(0));
        let gid = fb.builtin(BuiltinVar::GlobalInvocationId);
        let lid = fb.builtin(BuiltinVar::LocalInvocationId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let (gx, gy, lx, nx) = (fb.field(gid, 0, KirType::U32), fb.field(gid, 1, KirType::U32), fb.field(lid, 0, KirType::U32), fb.field(nwg, 0, KirType::U32));
        let base = fb.let_val("base", gx - lx + gy * nx * lu(fb, wg));
        let i = fb.let_val("i", base + lx);
        let inside = fb.lt(i, fb.grid_size(0));
        fb.if_then(inside, |b| {
            let index = self.global(b, i);
            let prog = self.program(b, i, index);
            for (j, &at) in key.outputs.iter().enumerate() {
                let value = self.value(b, &prog, at as usize);
                b.shared_store(stage, lx * lu(b, total) + lu(b, j as u32), value);
            }
        });
        fb.barrier(BarrierScope::Workgroup);
        let cells = fb.binary(IrBinOp::Min, lu(fb, wg), fb.grid_size(0) - base);
        let cells = fb.let_val("cells", cells);
        let out = self.env.outs.first().copied().unwrap_or(0);
        let end = cells * lu(fb, total);
        fb.for_range("e", lx, end, lu(fb, wg), |b, e| {
            let value = b.shared_load(stage, e);
            b.store(out, base * lu(b, total) + e, value);
        });
    }

    /// A store kernel with `bool` outputs, which are `bits` (the numbers of the outputs): a
    /// workgroup of `wg` invocations (a multiple of 32) owns `wg` consecutive elements, which are
    /// `wg / 32` whole words of every bool output, so it packs the bits it computes and writes
    /// the words itself. The invocations past the end take part (the packing is a barrier, or a
    /// ballot) and contribute false, so the unused bits of the last word are 0. With `ballot`
    /// (the subgroups have exactly 32 invocations, which are consecutive in the workgroup) one
    /// ballot gives a word and its first invocation writes it; otherwise the invocations OR their
    /// bits into workgroup memory and the first `wg / 32` of them write the words after a barrier.
    fn packed(&self, fb: &mut FnBuilder<'_>) {
        debug_assert!(self.wg.is_multiple_of(32));
        let (wg, ballot_words) = (self.wg, self.tuning.subgroup32);
        let words = wg / 32;
        let bits = self.bit_outputs();
        let gid = fb.builtin(BuiltinVar::GlobalInvocationId);
        let lid = fb.builtin(BuiltinVar::LocalInvocationId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let lane = ballot_words.then(|| fb.builtin(BuiltinVar::SubgroupInvocationId));
        let (gx, gy, lx, nx) = (fb.field(gid, 0, KirType::U32), fb.field(gid, 1, KirType::U32), fb.field(lid, 0, KirType::U32), fb.field(nwg, 0, KirType::U32));
        let base = fb.let_val("base", gx - lx + gy * nx * lu(fb, wg));
        let i = fb.let_val("i", base + lx);
        let size = fb.grid_size(0);
        if !ballot_words {
            for &shared in self.shared.iter().take(bits.len()) {
                let first = fb.lt(lx, lu(fb, words));
                fb.if_then(first, |b| b.atomic_store(shared_target(shared, lx), lu(b, 0)));
            }
            fb.barrier(BarrierScope::Workgroup);
        }
        let vars: Vec<(usize, LocalId)> = bits.iter().map(|&j| (j, fb.var_decl(&format!("b{j}"), KirType::U32, Some(lu(fb, 0))))).collect();
        let inside = fb.lt(i, size);
        fb.if_then(inside, |b| {
            let index = self.global(b, i);
            let prog = self.program(b, i, index);
            for (j, &at) in self.outputs.iter().enumerate() {
                let value = self.value(b, &prog, at);
                match vars.iter().find(|&&(bit, _)| bit == j) {
                    Some(&(_, local)) => b.assign_var(local, value),
                    None => self.env.store(b, self.out_elem(at), self.env.outs[j], i, value),
                }
            }
        });
        for (n, &(j, local)) in vars.iter().enumerate() {
            let out = self.env.outs[j];
            let bit = fb.read_var(local);
            if let Some(lane) = lane {
                let word = ballot(fb, fb.ne(bit, lu(fb, 0)));
                let word = fb.let_val("m", word);
                let writes = fb.and(fb.eq(lane, lu(fb, 0)), fb.lt(i, size));
                fb.if_then(writes, |b| b.atomic_store(storage_target(out, i >> lu(b, 5)), word));
            } else {
                let shared = self.shared[n];
                let shifted = bit << (lx & lu(fb, 31));
                fb.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::U32, shared_target(shared, lx >> lu(fb, 5)), shifted);
            }
        }
        if !ballot_words {
            fb.barrier(BarrierScope::Workgroup);
            for (n, &(j, _)) in vars.iter().enumerate() {
                let (out, shared) = (self.env.outs[j], self.shared[n]);
                let writes = fb.and(fb.lt(lx, lu(fb, words)), fb.lt(base + lx * lu(fb, 32), size));
                fb.if_then(writes, |b| {
                    let word = b.atomic_load(shared_target(shared, lx), ScalarTy::U32);
                    b.atomic_store(storage_target(out, (base >> lu(b, 5)) + lx), word);
                });
            }
        }
    }

    /// The write of the word `v` of the output `j` into the base word `wd`, `m` being the bits of
    /// the run: a whole word is stored as it is, at the ends of a run an atomic AND clears the
    /// bits the run writes and an atomic OR sets those that are true.
    fn store_word(&self, fb: &mut FnBuilder<'_>, j: usize, v: Val, wd: Val, m: Val) {
        let out = self.env.outs[j];
        let target = move || storage_target(out, wd);
        let full = fb.eq(m, lu(fb, 0xFFFF_FFFF));
        fb.if_else(
            full,
            |b| b.atomic_store(target(), v),
            |b| {
                b.atomic_rmw(None, IrAtomicOp::And, ScalarTy::U32, target(), !m);
                let any = b.ne(v, lu(b, 0));
                b.if_then(any, |b| b.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::U32, target(), v));
            },
        );
    }

    /// The kernel of a region write ([`Tail::StoreMapped`]) with `bool` outputs when the region is
    /// made of runs of consecutive elements of the base (`p.run` elements; the first element of
    /// run `r` is `view(r * p.run)`; a run spans at most `p.words` words of the base). The
    /// invocations are the 32 bits of every word a run spans, 32 neighbours to a word: the bit of
    /// base element `at` computes the element `i` of the region that goes there (if the run holds
    /// `at`: otherwise the invocation only takes part in packing, as a bit nobody writes), so the
    /// 32 neighbours build the whole word in registers (a ballot, or an OR in workgroup memory),
    /// as [`Build::packed`] does, and the word is stored as it is if all 32 bits belong to the
    /// run. The words at the ends of a run, shared with elements that the write leaves alone (and
    /// with the neighbouring runs), only change the bits of the run.
    fn runs(&self, fb: &mut FnBuilder<'_>) {
        debug_assert!(self.wg.is_multiple_of(32));
        let key = self.key;
        let (wg, ballot_words) = (self.wg, self.tuning.subgroup32);
        let words = wg / 32;
        let Some(view) = self.env.mapped_view(key.tail) else {
            self.env.fail(fb, "runs are for region writes");
            return;
        };
        let bits = self.bit_outputs();
        let gid = fb.builtin(BuiltinVar::GlobalInvocationId);
        let lid = fb.builtin(BuiltinVar::LocalInvocationId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let lane = ballot_words.then(|| fb.builtin(BuiltinVar::SubgroupInvocationId));
        let (gx, gy, lx, nx) = (fb.field(gid, 0, KirType::U32), fb.field(gid, 1, KirType::U32), fb.field(lid, 0, KirType::U32), fb.field(nwg, 0, KirType::U32));
        let g = fb.let_val("g", gx + gy * nx * lu(fb, wg));
        let slot = fb.let_val("slot", g >> lu(fb, 5));
        let bit = fb.let_val("bit", g & lu(fb, 31));
        let (run_len, run_words) = (self.env.named(fb, "run"), self.env.named(fb, "words"));
        let run = fb.let_val("run", slot / run_words);
        let start = fb.call(view, &[run * run_len], KirType::U32);
        let rb = fb.let_val("rb", start);
        let wd = fb.let_val("wd", (rb >> lu(fb, 5)) + (slot - run * run_words));
        let element = fb.let_val("element", (wd << lu(fb, 5)) + bit);
        let ok = fb.and(fb.and(fb.lt(g, fb.grid_size(0)), fb.ge(element, rb)), fb.lt(element - rb, run_len));
        let ok = fb.let_val("ok", ok);
        let i = fb.let_val("i", run * run_len + (element - rb));
        // The shared arrays: `luma_valid`, then the words of the `bool` outputs.
        let valid = (!ballot_words).then(|| self.shared[0]);
        if let Some(valid) = valid {
            let first = fb.lt(lx, lu(fb, words));
            fb.if_then(first, |b| b.atomic_store(shared_target(valid, lx), lu(b, 0)));
            for n in 0..bits.len() {
                let shared = self.shared[1 + n];
                let first = fb.lt(lx, lu(fb, words));
                fb.if_then(first, |b| b.atomic_store(shared_target(shared, lx), lu(b, 0)));
            }
            fb.barrier(BarrierScope::Workgroup);
        }
        let vars: Vec<(usize, LocalId)> = bits.iter().map(|&j| (j, fb.var_decl(&format!("b{j}"), KirType::U32, Some(lu(fb, 0))))).collect();
        fb.if_then(ok, |b| {
            let index = self.global(b, i);
            let prog = self.program(b, i, index);
            for (j, &at) in self.outputs.iter().enumerate() {
                let value = self.value(b, &prog, at);
                match vars.iter().find(|&&(bit, _)| bit == j) {
                    Some(&(_, local)) => b.assign_var(local, value),
                    None => {
                        let o = b.call(view, &[i], KirType::U32);
                        self.env.store(b, self.out_elem(at), self.env.outs[j], o, value);
                    }
                }
            }
        });
        if let Some(lane) = lane {
            let m = ballot(fb, ok);
            let m = fb.let_val("m", m);
            for &(j, local) in &vars {
                let bit_set = fb.ne(fb.read_var(local), lu(fb, 0));
                let word = ballot(fb, bit_set);
                let word = fb.let_val(&format!("v{j}"), word);
                let writes = fb.and(fb.eq(lane, lu(fb, 0)), fb.ne(m, lu(fb, 0)));
                fb.if_then(writes, |b| self.store_word(b, j, word, wd, m));
            }
        } else if let Some(valid) = valid {
            for (n, &(_, local)) in vars.iter().enumerate() {
                let shared = self.shared[1 + n];
                let shifted = fb.read_var(local) << bit;
                fb.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::U32, shared_target(shared, lx >> lu(fb, 5)), shifted);
            }
            let mine = fb.select(ok, lu(fb, 1) << bit, lu(fb, 0));
            fb.atomic_rmw(None, IrAtomicOp::Or, ScalarTy::U32, shared_target(valid, lx >> lu(fb, 5)), mine);
            fb.barrier(BarrierScope::Workgroup);
            let first = fb.lt(lx, lu(fb, words));
            fb.if_then(first, |b| {
                let m = b.atomic_load(shared_target(valid, lx), ScalarTy::U32);
                let m = b.let_val("m", m);
                let any = b.ne(m, lu(b, 0));
                b.if_then(any, |b| {
                    let s = b.let_val("s", ((g - lx) >> lu(b, 5)) + lx);
                    let r = b.let_val("r", s / run_words);
                    let first_word = b.call(view, &[r * run_len], KirType::U32);
                    let wd = b.let_val("wd", (first_word >> lu(b, 5)) + (s - r * run_words));
                    for (n, &(j, _)) in vars.iter().enumerate() {
                        let v = b.atomic_load(shared_target(self.shared[1 + n], lx), ScalarTy::U32);
                        let v = b.let_val("v", v);
                        self.store_word(b, j, v, wd, m);
                    }
                });
            });
        }
    }

    /// The kernel of a key with its sub-programs staged as `tile` says, running one workgroup per
    /// tile of the array.
    fn tiled(&self, fb: &mut FnBuilder<'_>) {
        let key = self.key;
        let Some(tile) = &key.tile else { return };
        let env = &self.env;
        let (tx, ty) = tile.tile;
        let region = |j: usize| (tx + 2 * tile.need[j].0, ty + 2 * tile.need[j].1);
        // The extents are literals, or words of the uniform.
        let (one_d, w, h, tiles_x) = match tile.slots {
            Some(slots) => (slots.one_d, env.scalar(fb, slots.base), env.scalar(fb, slots.base + 1), env.scalar(fb, slots.base + 2)),
            None => (tile.dims.1 == 1, lu(fb, tile.dims.0), lu(fb, tile.dims.1), lu(fb, tile.dims.0.div_ceil(tx))),
        };
        let lid = fb.builtin(BuiltinVar::LocalInvocationIndex);
        let wid = fb.builtin(BuiltinVar::WorkgroupId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let (wx, wy, nx) = (fb.field(wid, 0, KirType::U32), fb.field(wid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32));
        let g = fb.let_val("g", wx + wy * nx);
        let tile_x = fb.let_val("tile_x", (g % tiles_x) * lu(fb, tx));
        let tile_y = fb.let_val("tile_y", (g / tiles_x) * lu(fb, ty));
        for (k, instrs) in key.subs.iter().enumerate() {
            let ((nx, ny), (lw, lh)) = (tile.need[k], region(k));
            let last = instrs.len().saturating_sub(1);
            let shared = self.shared[k];
            for_cells(fb, lid, lw * lh, |b, e| {
                let lx = b.let_val("lx", b.convert(u_rem(b, e, lw), KirType::I32) - li(b, nx as i32));
                let ly = b.let_val("ly", b.convert(u_div(b, e, lw), KirType::I32) - li(b, ny as i32));
                // Cell (lx, ly) relative to the tile origin, as a global element.
                let across = env.wrap(b, b.convert(tile_x, KirType::I32) + lx, w);
                let element = if one_d { across } else { across + env.wrap(b, b.convert(tile_y, KirType::I32) + ly, h) * w };
                let i = b.let_val("i", element);
                let index = self.global(b, i);
                let staged = |at: usize, instr: &Instr, fb: &FnBuilder<'_>| self.tile_read(fb, tile, k + 1, at, instr, lx, ly, one_d);
                let frame = env.body_with(b, instrs, &key.input_tys, i, index, false, false, None, &staged);
                let value = match frame.t(last) {
                    Some(value) => value,
                    None => env.fail(b, "a staged sub-program without instructions"),
                };
                b.shared_store(shared, e, value);
            });
            fb.barrier(BarrierScope::Workgroup);
        }
        for_cells(fb, lid, tx * ty, |b, e| {
            let lx_word = b.let_val("lx_word", u_rem(b, e, tx));
            let lx = b.let_val("lx", b.convert(lx_word, KirType::I32));
            let ly_word = b.let_val("ly_word", u_div(b, e, tx));
            let ly = b.let_val("ly", b.convert(ly_word, KirType::I32));
            let x = b.let_val("x", tile_x + lx_word);
            let y = b.let_val("y", tile_y + ly_word);
            let inside = b.and(b.lt(x, w), b.lt(y, h));
            b.if_then(inside, |b| {
                let i = b.let_val("i", x + y * w);
                let index = self.global(b, i);
                let staged = |at: usize, instr: &Instr, fb: &FnBuilder<'_>| self.tile_read(fb, tile, 0, at, instr, lx, ly, one_d);
                let frame = env.body_with(b, &key.instrs, &key.input_tys, i, index, false, true, None, &staged);
                self.emit_stores(b, &Prog { frame, map: None }, i);
            });
        });
    }

    /// The value of instruction `at` of program `p` (0 = main, `1 + k` = `subs[k]`) of a tiled
    /// kernel if it is the read of a staged sub-program: the cell of its region at `(lx, ly)`
    /// plus the baked offset.
    #[allow(clippy::too_many_arguments)]
    fn tile_read(&self, fb: &FnBuilder<'_>, tile: &Tile, p: usize, at: usize, instr: &Instr, lx: Val, ly: Val, one_d: bool) -> Option<Val> {
        let (ox, oy) = tile.offsets.get(p)?.get(at).copied().flatten()?;
        let Ref::Sub(j) = instr.args[0] else { return None };
        let j = j as usize;
        let (nx, ny) = tile.need[j];
        let lw = tile.tile.0 + 2 * nx;
        let delta = (ox + nx as i32) + if one_d { 0 } else { (oy + ny as i32) * (lw as i32) };
        let base = if one_d { lx } else { lx + ly * li(fb, lw as i32) };
        let index = fb.bitcast(if delta == 0 { base } else { base + li(fb, delta) }, KirType::U32);
        Some(fb.shared_load(*self.shared.get(j)?, index))
    }

    /// The kernel of a key with its sub-programs staged as `staging` says: one workgroup per tile
    /// of output cells. Every level of staged sub-programs fills its workgroup-memory arrays
    /// (each cell of its region, strided over the invocations; `programs.groups` are the lowered
    /// bodies) before a barrier, then the main program reads them.
    fn staged(&self, fb: &mut FnBuilder<'_>) {
        let key = self.key;
        let (Some(Lowered::Staged(programs)), Some(staging)) = (self.lowered, &key.staging) else { return };
        let env = &self.env;
        let [tx, ty] = staging.tile;
        // The extents are literals, or words of the uniform.
        let (w, h, tiles_x) = match staging.slots {
            Some(slots) => (env.scalar(fb, slots.base), env.scalar(fb, slots.base + 1), env.scalar(fb, slots.base + 2)),
            None => (lu(fb, staging.dims[0]), lu(fb, staging.dims[1]), lu(fb, staging.dims[0].div_ceil(tx))),
        };
        let lid = fb.builtin(BuiltinVar::LocalInvocationIndex);
        let wid = fb.builtin(BuiltinVar::WorkgroupId);
        let nwg = fb.builtin(BuiltinVar::NumWorkgroups);
        let (wx, wy, nx) = (fb.field(wid, 0, KirType::U32), fb.field(wid, 1, KirType::U32), fb.field(nwg, 0, KirType::U32));
        let g = fb.let_val("g", wx + wy * nx);
        let tile_x = fb.let_val("tile_x", (g % tiles_x) * lu(fb, tx));
        let tile_y = fb.let_val("tile_y", (g / tiles_x) * lu(fb, ty));
        for level in staging.levels.windows(2) {
            for group in programs.groups.iter().filter(|group| {
                group.members.first().is_some_and(|&first| (level[0] as usize..level[1] as usize).contains(&first))
            }) {
                let Some(&first) = group.members.first() else { continue };
                let Some(entry) = staging.subs.get(first) else { continue };
                let (dw_i, dh_i, dw_u) = match staging.slots {
                    Some(slots) => {
                        let at = slots.sub(first);
                        let width = env.scalar(fb, at);
                        (fb.bitcast(width, KirType::I32), fb.bitcast(env.scalar(fb, at + 1), KirType::I32), width)
                    }
                    None => (li(fb, entry.dims[0] as i32), li(fb, entry.dims[1] as i32), lu(fb, entry.dims[0])),
                };
                let rw = entry.width();
                for_cells(fb, lid, entry.cells(), |b, e| {
                    let lx = b.let_val("lx", b.convert(u_rem(b, e, rw), KirType::I32) + li(b, entry.lo[0]));
                    let ly = b.let_val("ly", b.convert(u_div(b, e, rw), KirType::I32) + li(b, entry.lo[1]));
                    let x = b.let_val("x", b.convert(tile_x, KirType::I32) + lx);
                    let y = b.let_val("y", b.convert(tile_y, KirType::I32) + ly);
                    let zero = li(b, 0);
                    let inside = b.and(b.and(b.ge(x, zero), b.lt(x, dw_i)), b.and(b.ge(y, zero), b.lt(y, dh_i)));
                    b.if_then(inside, |b| {
                        let i = b.let_val("i", b.bitcast(x, KirType::U32) + b.bitcast(y, KirType::U32) * dw_u);
                        let index = self.global(b, i);
                        let cell = StagedCell { staging, lx, ly, shared: &self.shared_w };
                        let frame = env.lowered_body(b, key, &group.body, i, index, false, Some(&cell), None, &group.values);
                        for (&member, &last) in group.members.iter().zip(&group.values) {
                            let Some(sub) = staging.subs.get(member) else { continue };
                            let Some(shared) = self.shared_w.get(sub.sub as usize).copied().flatten() else { continue };
                            let value = match frame.t(last as usize) {
                                Some(value) => value,
                                None => env.fail(b, "a staged value is not computed"),
                            };
                            b.shared_store(shared, e, value);
                        }
                    });
                });
            }
            fb.barrier(BarrierScope::Workgroup);
        }
        for_cells(fb, lid, tx * ty, |b, e| {
            let lx_word = b.let_val("lx_word", u_rem(b, e, tx));
            let lx = b.let_val("lx", b.convert(lx_word, KirType::I32));
            let ly_word = b.let_val("ly_word", u_div(b, e, tx));
            let ly = b.let_val("ly", b.convert(ly_word, KirType::I32));
            let x = b.let_val("x", tile_x + lx_word);
            let y = b.let_val("y", tile_y + ly_word);
            let inside = b.and(b.lt(x, w), b.lt(y, h));
            b.if_then(inside, |b| {
                let i = b.let_val("i", x + y * w);
                let index = self.global(b, i);
                let cell = StagedCell { staging, lx, ly, shared: &self.shared_w };
                let frame = env.lowered_body(b, key, &programs.main, i, index, true, Some(&cell), None, &[]);
                self.emit_stores(b, &Prog { frame, map: Some(&programs.main) }, i);
            });
        });
    }
}

// ---------------------------------------------------------------------------------------------
// The register loop
// ---------------------------------------------------------------------------------------------

/// A kernel running `p.steps` iterations of the store program `key` per element, with the state
/// in registers. `states[k]` is the state that input `k` reads (`None`: a loop-invariant
/// input). Bindings: one output and one initial value per state, the invariant inputs, `p`.
pub(super) fn loop_module(key: &FusedKey, states: &[Option<usize>], tuning: &Tuning) -> Result<KernelModule> {
    let wg = tuning.elementwise_workgroup;
    let count = key.outputs.len();
    // The type of the states (`run_loop_fused` checked that they agree).
    let state_elem = key.outputs.first().and_then(|&at| key.instrs.get(at as usize)).map_or(ElemTy::F32, |instr| instr.op.out_ty());
    // The tests of the class of a value come with the double-float library.
    let lib = helper_library(&[key.instrs.as_slice()], tuning, false, false)?;
    let label = format!("Loop{}", key.label().trim_start_matches("Fused"));
    let mut mb = ModuleBuilder::new(label, "main", [wg, 1, 1]);
    let mut env = Lower::new(lib.clone(), &[]);
    let inits: Vec<u32>;
    {
        for j in 0..count {
            env.outs.push(mb.add_storage_read_write(&format!("out{j}"), write_elem(state_elem)));
        }
        inits = (0..count).map(|j| mb.add_storage_read(&format!("init{j}"), reg(state_elem))).collect();
        for (k, &ty) in key.input_tys.iter().enumerate() {
            let binding = states.get(k).copied().flatten().is_none().then(|| mb.add_storage_read(&format!("in{k}"), reg(ty)));
            env.ins.push(binding);
        }
        let mut fields: Vec<(String, KirType)> = vec![("steps".to_owned(), KirType::U32)];
        for (k, &ty) in key.scalar_tys.iter().enumerate() {
            fields.push((format!("s{k}"), reg(ty)));
        }
        let uni = Uni::declare(&mut mb, fields);
        env.set_uniform(uni);
    }
    {
        let mut fb = mb.entry_builder();
        let i = element_index(&mut fb, wg);
        let size = fb.grid_size(0);
        let outside = fb.ge(i, size);
        fb.if_then(outside, |b| b.ret(None));
        let mut locals = Vec::with_capacity(count);
        for (j, &init) in inits.iter().enumerate() {
            let value = env.load_in(&mut fb, state_elem, init, i);
            locals.push(fb.var_decl(&format!("s{j}"), reg(state_elem), Some(value)));
        }
        let mut frame = Frame::with_len(key.instrs.len());
        for (k, &ty) in key.input_tys.iter().enumerate() {
            if states.get(k).copied().flatten().is_none() && direct(&key.instrs, k as u32) {
                let value = env.input_load(&mut fb, ty, k as u32, i, true);
                let value = fb.let_val(&format!("a{k}"), value);
                frame.a.insert((k as u32, 0), value);
            }
        }
        let steps = env.named(&fb, "steps");
        let (zero, one) = (lu(&fb, 0), lu(&fb, 1));
        let env_ref = &env;
        fb.for_range("step", zero, steps, one, |b, _| {
            let mut frame = frame;
            for (k, state) in states.iter().enumerate() {
                if let Some(j) = state {
                    let value = b.read_var(locals[*j]);
                    let value = b.let_val(&format!("a{k}"), value);
                    frame.a.insert((k as u32, 0), value);
                }
            }
            for (k, instr) in key.instrs.iter().enumerate() {
                env_ref.bind_instr(b, &mut frame, k, instr, i, false, true, None);
            }
            for (j, &at) in key.outputs.iter().enumerate() {
                if let Some(value) = frame.t(at as usize) {
                    b.assign_var(locals[j], value);
                }
            }
        });
        for (j, &local) in locals.iter().enumerate() {
            let value = fb.read_var(local);
            env.store(&mut fb, state_elem, env.outs[j], i, value);
        }
        fb.finish();
    }
    if let Some(message) = env.error() {
        return Err(LumaError::Gpu(format!("loop kernel: {message}")));
    }
    Ok(finish(mb, lib.as_deref()))
}
