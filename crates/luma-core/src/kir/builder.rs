//! Direct construction of a [`KernelModule`] without formatting source text.
//!
//! [`Val`] is a `Copy` handle to an expression held in a thread-local arena that lives while any
//! [`FnBuilder`] of the thread does, so `a + b`, `a * b`, `!a`, `-a` work on plain values.
//! Expressions are trees: combining two `Val`s copies their operands, so bind a shared
//! sub-expression with [`FnBuilder::let_val`] instead of building it twice.

use std::cell::{Cell, RefCell};
use crate::hal::BindingKind;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use super::ir::*;

thread_local! {
    static ARENA: RefCell<Vec<(Expr, KirType)>> = const { RefCell::new(Vec::new()) };
    static LIVE: Cell<u32> = const { Cell::new(0) };
}

/// A value (an expression and its type) of the function being built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Val(pub u32);

fn push(expr: Expr, ty: KirType) -> Val {
    ARENA.with(|a| {
        let mut a = a.borrow_mut();
        a.push((expr, ty));
        Val(a.len() as u32 - 1)
    })
}

impl Val {
    /// The expression this value stands for.
    pub fn expr(self) -> Expr {
        ARENA.with(|a| a.borrow()[self.0 as usize].0.clone())
    }

    pub fn ty(self) -> KirType {
        ARENA.with(|a| a.borrow()[self.0 as usize].1.clone())
    }
}

fn bool_like(t: &KirType) -> KirType {
    match t {
        KirType::Vec { lanes, .. } => KirType::vec(ScalarTy::Bool, *lanes),
        _ => KirType::BOOL,
    }
}

fn binary_ty(op: IrBinOp, l: &KirType, r: &KirType) -> KirType {
    if op.is_comparison() {
        return bool_like(if matches!(l, KirType::Vec { .. }) { l } else { r });
    }
    match op {
        IrBinOp::And | IrBinOp::Or => KirType::BOOL,
        IrBinOp::MulHiU32 => KirType::U32,
        IrBinOp::Shl | IrBinOp::Shr | IrBinOp::Rotl => l.clone(),
        _ => {
            if matches!(l, KirType::Vec { .. }) || !matches!(r, KirType::Vec { .. }) {
                l.clone()
            } else {
                r.clone()
            }
        }
    }
}

fn unary_ty(op: IrUnOp, t: &KirType) -> KirType {
    match op {
        IrUnOp::IsNan | IrUnOp::IsInf => bool_like(t),
        IrUnOp::All | IrUnOp::Any => KirType::BOOL,
        _ => t.clone(),
    }
}

fn bin(op: IrBinOp, a: Val, b: Val) -> Val {
    let (ta, tb) = (a.ty(), b.ty());
    let ty = binary_ty(op, &ta, &tb);
    push(Expr::Binary { op, lhs: a.expr().boxed(), rhs: b.expr().boxed(), ty: ty.clone() }, ty)
}

fn un(op: IrUnOp, a: Val) -> Val {
    let ty = unary_ty(op, &a.ty());
    push(Expr::Unary { op, arg: a.expr().boxed(), ty: ty.clone() }, ty)
}

macro_rules! val_op {
    ($tr:ident, $m:ident, $op:expr) => {
        impl $tr for Val {
            type Output = Val;
            fn $m(self, rhs: Val) -> Val {
                bin($op, self, rhs)
            }
        }
    };
}

val_op!(Add, add, IrBinOp::Add);
val_op!(Sub, sub, IrBinOp::Sub);
val_op!(Mul, mul, IrBinOp::Mul);
val_op!(Div, div, IrBinOp::Div);
val_op!(Rem, rem, IrBinOp::Rem);
val_op!(BitAnd, bitand, IrBinOp::BitAnd);
val_op!(BitOr, bitor, IrBinOp::BitOr);
val_op!(BitXor, bitxor, IrBinOp::BitXor);
val_op!(Shl, shl, IrBinOp::Shl);
val_op!(Shr, shr, IrBinOp::Shr);

impl Neg for Val {
    type Output = Val;
    fn neg(self) -> Val {
        un(IrUnOp::Neg, self)
    }
}

impl Not for Val {
    type Output = Val;
    /// Logical not on bools, bitwise not on integers.
    fn not(self) -> Val {
        let op = if self.ty().elem_scalar() == Some(ScalarTy::Bool) { IrUnOp::Not } else { IrUnOp::BitNot };
        un(op, self)
    }
}

/// Declarations of a module; entry and helper bodies are built with [`FnBuilder`].
#[derive(Debug)]
pub struct ModuleBuilder {
    m: KernelModule,
}

impl ModuleBuilder {
    pub fn new(label: impl Into<String>, entry: impl Into<String>, workgroup_size: [u32; 3]) -> Self {
        ModuleBuilder {
            m: KernelModule {
                label: label.into(),
                entry: entry.into(),
                workgroup_size,
                uses_exact_f32: false,
                structs: Vec::new(),
                bindings: Vec::new(),
                shared: Vec::new(),
                privates: Vec::new(),
                functions: Vec::new(),
                entry_builtins: Vec::new(),
                entry_locals: Vec::new(),
                entry_body: Vec::new(),
            },
        }
    }

    pub fn set_uses_exact_f32(&mut self, exact: bool) -> &mut Self {
        self.m.uses_exact_f32 = exact;
        self
    }

    /// Declares a struct whose fields are given with their offsets, size and alignment.
    pub fn add_struct(&mut self, decl: StructDecl) -> u32 {
        self.m.structs.push(decl);
        self.m.structs.len() as u32 - 1
    }

    /// Declares a struct laid out by WGSL's rules (what `encase` writes for uniforms).
    pub fn add_struct_layout(&mut self, name: &str, fields: &[(&str, KirType)]) -> u32 {
        let mut offset = 0u32;
        let mut align = 1u32;
        let mut out = Vec::with_capacity(fields.len());
        for (fname, ty) in fields {
            let (size, a) = ty.layout(&self.m.structs);
            offset = offset.next_multiple_of(a);
            out.push(StructField { name: (*fname).to_owned(), ty: ty.clone(), offset });
            offset += size;
            align = align.max(a);
        }
        self.add_struct(StructDecl { name: name.to_owned(), fields: out, size: offset.next_multiple_of(align), align })
    }

    /// A struct of `u32` words, the shape of luma's own uniform parameters: `names[i]` at byte
    /// `4 * i`, size `4 * n`, align 4.
    pub fn add_uniform_u32_struct(&mut self, name: &str, names: &[&str]) -> u32 {
        let fields: Vec<(&str, KirType)> = names.iter().map(|n| (*n, KirType::U32)).collect();
        self.add_struct_layout(name, &fields)
    }

    fn add_storage(&mut self, name: &str, kind: BindingKind, elem_ty: KirType) -> u32 {
        let (size, align) = elem_ty.layout(&self.m.structs);
        self.m.bindings.push(BindingDecl {
            name: name.to_owned(),
            kind,
            elem_ty,
            is_unsized_array: true,
            element_stride: Some(size.next_multiple_of(align)),
            min_size: None,
        });
        self.m.bindings.len() as u32 - 1
    }

    /// `var<storage, read> name: array<elem_ty>`; returns the binding number.
    pub fn add_storage_read(&mut self, name: &str, elem_ty: KirType) -> u32 {
        self.add_storage(name, BindingKind::StorageRead, elem_ty)
    }

    /// `var<storage, read_write> name: array<elem_ty>`.
    pub fn add_storage_read_write(&mut self, name: &str, elem_ty: KirType) -> u32 {
        self.add_storage(name, BindingKind::StorageReadWrite, elem_ty)
    }

    /// `var<uniform> name: ty`.
    pub fn add_uniform(&mut self, name: &str, ty: KirType) -> u32 {
        let size = ty.layout(&self.m.structs).0;
        self.m.bindings.push(BindingDecl {
            name: name.to_owned(),
            kind: BindingKind::Uniform,
            elem_ty: ty,
            is_unsized_array: false,
            element_stride: None,
            min_size: Some(size),
        });
        self.m.bindings.len() as u32 - 1
    }

    /// `var<workgroup> name: array<elem_ty, len>`.
    pub fn add_shared(&mut self, name: &str, elem_ty: KirType, len: u32) -> u32 {
        self.m.shared.push(SharedDecl { name: name.to_owned(), elem_ty, len: Some(len) });
        self.m.shared.len() as u32 - 1
    }

    /// `var<workgroup> name: ty`.
    pub fn add_shared_scalar(&mut self, name: &str, ty: KirType) -> u32 {
        self.m.shared.push(SharedDecl { name: name.to_owned(), elem_ty: ty, len: None });
        self.m.shared.len() as u32 - 1
    }

    pub fn add_private(&mut self, name: &str, ty: KirType, init: Option<Expr>) -> u32 {
        self.m.privates.push(PrivateDecl { name: name.to_owned(), ty, init });
        self.m.privates.len() as u32 - 1
    }

    /// Builds a helper function with `body`.
    pub fn add_function(
        &mut self,
        name: &str,
        params: &[(&str, KirType)],
        ret_ty: Option<KirType>,
        body: impl FnOnce(&mut FnBuilder<'_>),
    ) -> FnId {
        let mut b = FnBuilder::new(&mut self.m, Some((name, params, ret_ty)));
        body(&mut b);
        b.finish().expect("helper function")
    }

    /// The builder of the entry point body; call [`FnBuilder::finish`] to install it.
    pub fn entry_builder(&mut self) -> FnBuilder<'_> {
        FnBuilder::new(&mut self.m, None)
    }

    pub fn finish(self) -> KernelModule {
        self.m
    }
}

struct Header {
    name: String,
    params: Vec<(String, KirType)>,
    ret_ty: Option<KirType>,
}

/// Builds the body of the entry point or of a helper function.
pub struct FnBuilder<'m> {
    m: &'m mut KernelModule,
    header: Option<Header>,
    locals: Vec<(String, KirType, bool)>,
    blocks: Vec<Vec<Stmt>>,
    builtins: Vec<(BuiltinVar, LocalId)>,
}

impl Drop for FnBuilder<'_> {
    fn drop(&mut self) {
        LIVE.with(|l| {
            let n = l.get() - 1;
            l.set(n);
            if n == 0 {
                ARENA.with(|a| a.borrow_mut().clear());
            }
        });
    }
}

impl<'m> FnBuilder<'m> {
    #[allow(clippy::type_complexity)]
    fn new(m: &'m mut KernelModule, header: Option<(&str, &[(&str, KirType)], Option<KirType>)>) -> Self {
        LIVE.with(|l| l.set(l.get() + 1));
        let header = header.map(|(name, params, ret_ty)| Header {
            name: name.to_owned(),
            params: params.iter().map(|(n, t)| ((*n).to_owned(), t.clone())).collect(),
            ret_ty,
        });
        FnBuilder { m, header, locals: Vec::new(), blocks: vec![Vec::new()], builtins: Vec::new() }
    }

    /// Installs the body: the entry point (returns `None`) or a helper function.
    pub fn finish(mut self) -> Option<FnId> {
        let body = std::mem::take(&mut self.blocks[0]);
        let locals = std::mem::take(&mut self.locals);
        match self.header.take() {
            Some(h) => {
                self.m.functions.push(KirFunction { name: h.name, params: h.params, ret_ty: h.ret_ty, locals, body });
                Some(FnId(self.m.functions.len() as u32 - 1))
            }
            None => {
                self.m.entry_builtins = std::mem::take(&mut self.builtins);
                self.m.entry_locals = locals;
                self.m.entry_body = body;
                None
            }
        }
    }

    /// The expression of `v`.
    pub fn expr(&self, v: Val) -> Expr {
        v.expr()
    }

    /// The type of `v`.
    pub fn ty(&self, v: Val) -> KirType {
        v.ty()
    }

    /// Imports an expression of known type.
    pub fn val(&self, expr: Expr, ty: KirType) -> Val {
        push(expr, ty)
    }

    /// Appends a raw statement.
    pub fn push_stmt(&mut self, s: Stmt) {
        self.blocks.last_mut().expect("block").push(s);
    }

    fn new_local(&mut self, name: &str, ty: KirType, mutable: bool) -> LocalId {
        self.locals.push((name.to_owned(), ty, mutable));
        LocalId(self.locals.len() as u32 - 1)
    }

    // Literals.
    pub fn lit_u32(&self, v: u32) -> Val {
        push(Expr::LitU32(v), KirType::U32)
    }
    pub fn lit_i32(&self, v: i32) -> Val {
        push(Expr::LitI32(v), KirType::I32)
    }
    pub fn lit_f32(&self, v: f32) -> Val {
        push(Expr::LitF32(v), KirType::F32)
    }
    pub fn lit_f64(&self, v: f64) -> Val {
        push(Expr::LitF64(v), KirType::Scalar(ScalarTy::F64))
    }
    pub fn lit_bool(&self, v: bool) -> Val {
        push(Expr::LitBool(v), KirType::BOOL)
    }
    pub fn zero(&self, ty: KirType) -> Val {
        push(Expr::ZeroVal(ty.clone()), ty)
    }

    /// Component `axis` (0..3) of the launch grid size.
    pub fn grid_size(&self, axis: u8) -> Val {
        push(Expr::GridSize(axis), KirType::U32)
    }

    /// A builtin input of the entry point, declared on first use.
    pub fn builtin(&mut self, var: BuiltinVar) -> Val {
        let local = match self.builtins.iter().find(|(v, _)| *v == var) {
            Some((_, l)) => *l,
            None => {
                let l = self.new_local(&format!("{var:?}"), var.ty(), false);
                self.builtins.push((var, l));
                l
            }
        };
        push(Expr::Local(local), var.ty())
    }

    /// Parameter `idx` of a helper function.
    pub fn param(&self, idx: u32) -> Val {
        let ty = self.header.as_ref().expect("param outside a helper function").params[idx as usize].1.clone();
        push(Expr::FnParam(idx), ty)
    }

    pub fn private(&self, idx: u32) -> Val {
        push(Expr::Private(idx), self.m.privates[idx as usize].ty.clone())
    }

    // Operators.
    pub fn unary(&self, op: IrUnOp, x: Val) -> Val {
        un(op, x)
    }
    pub fn binary(&self, op: IrBinOp, x: Val, y: Val) -> Val {
        bin(op, x, y)
    }
    pub fn add(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Add, x, y)
    }
    pub fn sub(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Sub, x, y)
    }
    pub fn mul(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Mul, x, y)
    }
    pub fn div(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Div, x, y)
    }
    pub fn rem(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Rem, x, y)
    }
    pub fn bit_and(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::BitAnd, x, y)
    }
    pub fn bit_or(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::BitOr, x, y)
    }
    pub fn bit_xor(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::BitXor, x, y)
    }
    pub fn shl(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Shl, x, y)
    }
    pub fn shr(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Shr, x, y)
    }
    pub fn mul_hi_u32(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::MulHiU32, x, y)
    }
    pub fn eq(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Eq, x, y)
    }
    pub fn ne(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Ne, x, y)
    }
    pub fn lt(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Lt, x, y)
    }
    pub fn le(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Le, x, y)
    }
    pub fn gt(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Gt, x, y)
    }
    pub fn ge(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Ge, x, y)
    }
    pub fn and(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::And, x, y)
    }
    pub fn or(&self, x: Val, y: Val) -> Val {
        bin(IrBinOp::Or, x, y)
    }
    pub fn neg(&self, x: Val) -> Val {
        un(IrUnOp::Neg, x)
    }
    /// Logical not.
    pub fn not(&self, x: Val) -> Val {
        un(IrUnOp::Not, x)
    }
    pub fn bit_not(&self, x: Val) -> Val {
        un(IrUnOp::BitNot, x)
    }
    /// `x` through an optimisation barrier (`luma_o`).
    pub fn opaque_guard(&self, x: Val) -> Val {
        un(IrUnOp::OpaqueGuard, x)
    }

    fn ternary(&self, op: IrTernaryOp, a: Val, b: Val, c: Val, ty: KirType) -> Val {
        push(Expr::Ternary { op, a: a.expr().boxed(), b: b.expr().boxed(), c: c.expr().boxed(), ty: ty.clone() }, ty)
    }
    pub fn fma(&self, a: Val, b: Val, c: Val) -> Val {
        self.ternary(IrTernaryOp::Fma, a, b, c, a.ty())
    }
    pub fn clamp(&self, x: Val, lo: Val, hi: Val) -> Val {
        self.ternary(IrTernaryOp::Clamp, x, lo, hi, x.ty())
    }
    /// `cond ? then_v : else_v`.
    pub fn select(&self, cond: Val, then_v: Val, else_v: Val) -> Val {
        self.ternary(IrTernaryOp::Select, cond, then_v, else_v, then_v.ty())
    }

    fn cast(&self, kind: CastKind, x: Val, dst_ty: KirType) -> Val {
        push(Expr::Cast { kind, expr: x.expr().boxed(), src_ty: x.ty(), dst_ty: dst_ty.clone() }, dst_ty)
    }
    pub fn convert(&self, x: Val, dst_ty: KirType) -> Val {
        self.cast(CastKind::Convert, x, dst_ty)
    }
    pub fn bitcast(&self, x: Val, dst_ty: KirType) -> Val {
        self.cast(CastKind::Bitcast, x, dst_ty)
    }
    pub fn float_to_int_sat(&self, x: Val, dst_ty: KirType) -> Val {
        self.cast(CastKind::FloatToIntSat, x, dst_ty)
    }

    // Memory.
    /// `path` of a uniform binding.
    pub fn uniform_read(&self, binding: u32, path: Vec<AccessStep>, ty: KirType) -> Val {
        push(Expr::UniformRead { binding, path, ty: ty.clone() }, ty)
    }
    /// Field `field_idx` of a struct uniform.
    pub fn uniform_field(&self, binding: u32, field_idx: u32, ty: KirType) -> Val {
        self.uniform_read(binding, vec![AccessStep::Field(field_idx)], ty)
    }
    /// `binding[idx]` of an unsized array binding.
    pub fn load(&self, binding: u32, idx: Val) -> Val {
        let ty = self.m.bindings[binding as usize].elem_ty.clone();
        push(Expr::Load { binding, index: idx.expr().boxed(), ty: ty.clone() }, ty)
    }
    pub fn store(&mut self, binding: u32, idx: Val, val: Val) {
        self.push_stmt(Stmt::Store { binding, index: idx.expr(), val: val.expr() });
    }
    pub fn shared_load(&self, shared: u32, idx: Val) -> Val {
        let ty = self.m.shared[shared as usize].elem_ty.clone();
        push(Expr::SharedLoad { shared, index: idx.expr().boxed(), ty: ty.clone() }, ty)
    }
    pub fn shared_store(&mut self, shared: u32, idx: Val, val: Val) {
        self.push_stmt(Stmt::SharedStore { shared, index: idx.expr(), val: val.expr() });
    }

    // Locals.
    /// `let name = val;`
    pub fn let_val(&mut self, name: &str, val: Val) -> Val {
        let ty = val.ty();
        let local = self.new_local(name, ty.clone(), false);
        self.push_stmt(Stmt::Let { local, init: val.expr() });
        push(Expr::Local(local), ty)
    }
    /// `var name: ty = init;`
    pub fn var_decl(&mut self, name: &str, ty: KirType, init: Option<Val>) -> LocalId {
        let local = self.new_local(name, ty, true);
        self.push_stmt(Stmt::Var { local, init: init.map(Val::expr) });
        local
    }
    pub fn read_var(&self, local: LocalId) -> Val {
        push(Expr::Local(local), self.locals[local.0 as usize].1.clone())
    }
    pub fn assign_var(&mut self, local: LocalId, val: Val) {
        self.push_stmt(Stmt::Assign { dst: LValue::Local(local), val: val.expr() });
    }
    pub fn assign(&mut self, dst: LValue, val: Val) {
        self.push_stmt(Stmt::Assign { dst, val: val.expr() });
    }

    // Aggregates.
    /// Struct field (or vector lane) `idx` of `v`.
    pub fn field(&self, v: Val, idx: u32, ty: KirType) -> Val {
        push(Expr::Access { base: v.expr().boxed(), step: AccessStep::Field(idx), ty: ty.clone() }, ty)
    }
    pub fn index(&self, v: Val, idx: Val, ty: KirType) -> Val {
        push(Expr::Access { base: v.expr().boxed(), step: AccessStep::Index(idx.expr().boxed()), ty: ty.clone() }, ty)
    }
    pub fn swizzle(&self, v: Val, indices: [u8; 4], len: u8, ty: KirType) -> Val {
        push(Expr::Swizzle { base: v.expr().boxed(), indices, len, ty: ty.clone() }, ty)
    }
    pub fn compose(&self, ty: KirType, comps: &[Val]) -> Val {
        push(Expr::Compose { ty: ty.clone(), comps: comps.iter().map(|c| c.expr()).collect() }, ty)
    }
    pub fn splat(&self, elem: ScalarTy, lanes: u8, val: Val) -> Val {
        push(Expr::Splat { elem, lanes, val: val.expr().boxed() }, KirType::vec(elem, lanes))
    }

    // Calls.
    pub fn call(&self, func: FnId, args: &[Val], ret_ty: KirType) -> Val {
        let call = Expr::Call { func, args: args.iter().map(|a| a.expr()).collect(), ty: Some(ret_ty.clone()) };
        push(call, ret_ty)
    }
    pub fn call_void(&mut self, func: FnId, args: &[Val]) {
        let call = Expr::Call { func, args: args.iter().map(|a| a.expr()).collect(), ty: None };
        self.push_stmt(Stmt::Discard(call));
    }

    // Atomics, barriers.
    pub fn atomic_load(&self, target: AtomicTarget, ty: ScalarTy) -> Val {
        push(Expr::AtomicLoad { target, ty }, KirType::Scalar(ty))
    }
    pub fn atomic_store(&mut self, target: AtomicTarget, val: Val) {
        self.push_stmt(Stmt::AtomicStore { target, val: val.expr() });
    }
    /// Read-modify-write; the previous value lands in `dst`.
    pub fn atomic_rmw(&mut self, dst: Option<LocalId>, op: IrAtomicOp, ty: ScalarTy, target: AtomicTarget, val: Val) {
        self.push_stmt(Stmt::AtomicRmw { dst, op, elem_ty: ty, target, val: val.expr() });
    }
    pub fn barrier(&mut self, scope: BarrierScope) {
        self.push_stmt(Stmt::Barrier(scope));
    }

    // Control flow.
    fn block(&mut self, f: impl FnOnce(&mut Self)) -> Vec<Stmt> {
        self.blocks.push(Vec::new());
        f(self);
        self.blocks.pop().expect("block")
    }

    pub fn if_then(&mut self, cond: Val, then_f: impl FnOnce(&mut Self)) {
        let then_body = self.block(then_f);
        self.push_stmt(Stmt::If { cond: cond.expr(), then_body, else_body: Vec::new() });
    }

    pub fn if_else(&mut self, cond: Val, then_f: impl FnOnce(&mut Self), else_f: impl FnOnce(&mut Self)) {
        let then_body = self.block(then_f);
        let else_body = self.block(else_f);
        self.push_stmt(Stmt::If { cond: cond.expr(), then_body, else_body });
    }

    /// `loop { body; continuing { cont; break if break_if } }`.
    pub fn loop_stmt(&mut self, body_f: impl FnOnce(&mut Self), cont_f: impl FnOnce(&mut Self), break_if: Option<Val>) {
        let body = self.block(body_f);
        let continuing = self.block(cont_f);
        self.push_stmt(Stmt::Loop { body, continuing, break_if: break_if.map(Val::expr) });
    }

    /// `for (var i = start; i < end; i += step) { body(i) }`; `i` has the type of `start`.
    pub fn for_range(&mut self, name: &str, start: Val, end: Val, step: Val, body_f: impl FnOnce(&mut Self, Val)) {
        let i = self.var_decl(name, start.ty(), Some(start));
        let body = self.block(|b| {
            let iv = b.read_var(i);
            let done = b.ge(iv, end);
            b.if_then(done, |b| b.brk());
            body_f(b, iv);
        });
        let continuing = self.block(|b| {
            let next = b.add(b.read_var(i), step);
            b.assign_var(i, next);
        });
        self.push_stmt(Stmt::Loop { body, continuing, break_if: None });
    }

    pub fn ret(&mut self, val: Option<Val>) {
        self.push_stmt(Stmt::Return(val.map(Val::expr)));
    }
    pub fn brk(&mut self) {
        self.push_stmt(Stmt::Break);
    }
    pub fn cont(&mut self) {
        self.push_stmt(Stmt::Continue);
    }
}
