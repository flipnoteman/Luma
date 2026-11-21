//! KIR -> NVVM IR (LLVM 7 dialect text), compiled to PTX by the driver's NVVM library.
//!
//! The emitted module is self-contained apart from `libdevice`, which the compiler links in:
//! math functions are calls to `__nv_*`. Every value is either SSA lanes (scalar or vector) or a
//! byte-addressed temporary (struct / array); places are `i8*` plus byte offsets, which a load or
//! store turns into one typed `getelementptr` of the pointer they came from (see `fb`). Scalar
//! locals that are declared once and never reassigned are SSA values ([`ssa_locals`]); the others
//! are `alloca [N x i8]` slots, which LLVM promotes to registers. The IR is kept small on purpose:
//! NVVM parses and optimizes it for every kernel the first time it is built, and the instructions
//! nothing uses are removed before it is handed over.
//!
//! Kernel ABI of `luma_main` (the order `launch_kernel` must pass the parameters in):
//! `%LumaGrid* byval align 4` (the grid), `i8*` `luma_indirect` (not `noalias`: a gate folded into a launch rewrites the arguments it is dispatched from) (null for direct
//! dispatches), one parameter per *active* binding in ascending original binding order (see
//! [`CudaArgAbi`]), then one `i32` per entry of [`CudaTranslated::array_lengths`].

mod conv;
mod fb;
mod ops;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use luma_core::BindingKind;
use luma_core::kir::*;

use fb::{Fb, Op, Val, align_of, llty, size_of};

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
    /// NVVM IR text.
    pub source: String,
    /// One entry per active binding, in ascending original binding order.
    pub param_abis: Vec<CudaArgAbi>,
    /// Bindings whose element count is appended, after the binding arguments, as
    /// `i32 %_luma_len{binding}` parameters (ascending). The count is the bound byte range
    /// divided by the binding's element stride.
    pub array_lengths: Vec<u32>,
}

const PARAM_BYTES: usize = 4096;
/// Largest uniform passed by value, whatever else is left of the parameter area.
const MAX_INLINE_UNIFORM: usize = 3584;

const TRIPLE: &str = "target triple = \"nvptx64-nvidia-cuda\"\n";
// `p3:32`: shared-memory pointers are 32-bit, as NVRTC emits them, so shared addressing is 32-bit
// arithmetic with constant offsets folded into the access (no `mul.wide`/`add.s64` per access).
const DATALAYOUT: &str = "target datalayout = \"e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n";
const TBAA: &str = "!0 = !{!\"luma tbaa\"}
!1 = !{!\"omnipotent char\", !0, i64 0}
!2 = !{!\"int\", !1, i64 0}
!3 = !{!\"float\", !1, i64 0}
!4 = !{!\"double\", !1, i64 0}
!5 = !{!\"short\", !1, i64 0}
!6 = !{!\"bool\", !1, i64 0}
!7 = !{!2, !2, i64 0}
!8 = !{!3, !3, i64 0}
!9 = !{!4, !4, i64 0}
!10 = !{!5, !5, i64 0}
!11 = !{!6, !6, i64 0}
";

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

/// The locals that can be SSA values instead of stack slots: scalars that are declared by exactly
/// one `let` (or by the kernel's prologue, `predefined`), never written again, and used only where
/// that `let` dominates the use (in its own block after it, or in blocks nested in the rest of that
/// block). Everything else keeps a stack slot, which NVVM promotes itself. Vectors stay in slots:
/// NVVM compiled the kernels that hold vectors as SSA lanes (and slots typed after their contents,
/// likewise) into different, branchier code.
fn ssa_locals(decls: &[(String, KirType, bool)], body: &[Stmt], predefined: &[LocalId]) -> Vec<bool> {
    struct Scope {
        cand: Vec<bool>,
    }
    impl Scope {
        fn uses(&mut self, s: &Stmt, vis: &[bool]) {
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| {
                    if let Expr::Local(l) = x
                        && !vis[l.0 as usize]
                    {
                        self.cand[l.0 as usize] = false;
                    }
                })
            });
        }
        fn block(&mut self, b: &[Stmt], vis: &mut [bool]) {
            for s in b {
                self.stmt(s, vis);
            }
        }
        fn nested(&mut self, b: &[Stmt], vis: &[bool]) {
            let mut v = vis.to_vec();
            self.block(b, &mut v);
        }
        fn stmt(&mut self, s: &Stmt, vis: &mut [bool]) {
            match s {
                Stmt::Let { local, .. } => {
                    self.uses(s, vis);
                    vis[local.0 as usize] = true;
                }
                Stmt::If { then_body, else_body, .. } => {
                    self.uses(s, vis);
                    self.nested(then_body, vis);
                    self.nested(else_body, vis);
                }
                Stmt::Loop { body, continuing, break_if } => {
                    let mut v = vis.to_vec();
                    self.block(body, &mut v);
                    // The `continuing` block is also reached by a `continue`, which skips the
                    // rest of the body.
                    let mut skips = false;
                    walk_stmts(body, &mut |x| skips |= matches!(x, Stmt::Continue));
                    let mut vc = if skips { vis.to_vec() } else { v };
                    self.block(continuing, &mut vc);
                    if let Some(e) = break_if {
                        e.walk(&mut |x| {
                            if let Expr::Local(l) = x
                                && !vc[l.0 as usize]
                            {
                                self.cand[l.0 as usize] = false;
                            }
                        });
                    }
                }
                Stmt::Switch { cases, default, .. } => {
                    self.uses(s, vis);
                    for (_, b, _) in cases {
                        self.nested(b, vis);
                    }
                    self.nested(default, vis);
                }
                _ => self.uses(s, vis),
            }
        }
    }
    let n = decls.len();
    let mut cand: Vec<bool> = decls.iter().map(|(_, t, _)| matches!(t, KirType::Scalar(_))).collect();
    let mut lets = vec![0u32; n];
    for l in predefined {
        lets[l.0 as usize] += 1;
    }
    fn root(lv: &LValue) -> Option<LocalId> {
        match lv {
            LValue::Local(l) => Some(*l),
            LValue::Index { base, .. } | LValue::Field { base, .. } => root(base),
            _ => None,
        }
    }
    walk_stmts(body, &mut |s| match s {
        Stmt::Let { local, .. } => lets[local.0 as usize] += 1,
        Stmt::Var { local, .. } => cand[local.0 as usize] = false,
        Stmt::Assign { dst, .. } => {
            if let Some(l) = root(dst) {
                cand[l.0 as usize] = false;
            }
        }
        Stmt::AtomicRmw { dst: Some(d), .. } => cand[d.0 as usize] = false,
        Stmt::AtomicCompareExchange { dst_old, dst_exchanged, .. } => {
            cand[dst_old.0 as usize] = false;
            cand[dst_exchanged.0 as usize] = false;
        }
        _ => {}
    });
    for (c, &l) in cand.iter_mut().zip(&lets) {
        *c &= l == 1;
    }
    let mut vis = vec![false; n];
    for l in predefined {
        vis[l.0 as usize] = true;
    }
    let mut sc = Scope { cand };
    sc.block(body, &mut vis);
    sc.cand
}

/// Everything an expression needs to name: the function's locals and parameters and the
/// context values (grid, bindings, shared, privates, lengths) it was handed.
struct FnCx<'b> {
    local_decls: &'b [(String, KirType, bool)],
    param_decls: &'b [(String, KirType)],
    /// `i8*` of each local's stack slot (empty for the locals that are SSA values).
    local_ptrs: Vec<Op>,
    /// Locals that live in SSA values, not stack slots (see [`ssa_locals`]).
    ssa: Vec<bool>,
    ssa_vals: RefCell<Vec<Option<Val>>>,
    param_vals: Vec<Val>,
    grid: Option<Op>,
    binds: BTreeMap<u32, Op>,
    shared: BTreeMap<u32, Op>,
    privs: BTreeMap<u32, Op>,
    lens: BTreeMap<u32, Op>,
    /// Hidden return pointer and type of a function returning a vector or aggregate.
    ret: Option<(Op, KirType)>,
}

impl<'b> FnCx<'b> {
    fn new(local_decls: &'b [(String, KirType, bool)], param_decls: &'b [(String, KirType)]) -> FnCx<'b> {
        FnCx {
            local_decls,
            param_decls,
            local_ptrs: Vec::new(),
            ssa: vec![false; local_decls.len()],
            ssa_vals: RefCell::new(vec![None; local_decls.len()]),
            param_vals: Vec::new(),
            grid: None,
            binds: BTreeMap::new(),
            shared: BTreeMap::new(),
            privs: BTreeMap::new(),
            lens: BTreeMap::new(),
            ret: None,
        }
    }
}

/// A leaf of an aggregate as [`Em::leaves`] visits it.
#[derive(Clone, Copy)]
enum Leaf {
    Scalar(ScalarTy),
    Vec(ScalarTy, u32),
}

/// Aggregates up to this size are copied and zeroed leaf by leaf, larger ones with
/// `llvm.memcpy`/`llvm.memset`.
const LEAF_COPY_BYTES: u32 = 1024;

/// One step of a place computation, the index already evaluated (an `i32`).
enum Step {
    Field(u32),
    Index(Op),
}

struct Jump {
    brk: String,
    cont: Option<String>,
}

struct Em<'a> {
    m: &'a KernelModule,
    active: Vec<bool>,
    abi: Vec<Option<CudaArgAbi>>,
    res: Vec<Res>,
    init_shared: bool,
    decls: BTreeSet<String>,
    helpers: BTreeMap<String, String>,
    jumps: Vec<Jump>,
    /// The subgroup operations every lane of the warp reaches (see [`conv`]).
    converged: HashSet<*const Expr>,
}

// ----- small value helpers ---------------------------------------------------------------

fn invariant(msg: impl std::fmt::Display) -> ! {
    panic!("luma-cuda codegen invariant: {msg}")
}

fn lanes_of(v: &Val) -> (ScalarTy, &[Op]) {
    match v {
        Val::Lanes { elem, ops } => (*elem, ops.as_slice()),
        Val::Mem { ty, .. } => invariant(format!("expected scalar or vector lanes, found aggregate {ty:?}")),
    }
}

fn one(v: &Val) -> (ScalarTy, Op) {
    match v {
        Val::Lanes { elem, ops } if ops.len() == 1 => (*elem, ops[0].clone()),
        other => invariant(format!("expected a scalar, found {other:?}")),
    }
}

fn pick(ops: &[Op], k: usize) -> &Op {
    if ops.len() == 1 { &ops[0] } else { &ops[k] }
}

fn scalar_val(elem: ScalarTy, op: Op) -> Val {
    Val::Lanes { elem, ops: vec![op] }
}

/// Whether a value of `ty` is returned in a register (scalar) rather than through `%ret`.
fn returns_direct(ty: &KirType) -> bool {
    matches!(ty, KirType::Scalar(_) | KirType::Atomic(_))
}

/// `min(idx, last)` of two `i32`s, unsigned.
fn umin_const(fb: &mut Fb, idx: &Op, last: u32) -> Op {
    let c = fb.def(&format!("icmp ult i32 {idx}, {last}"));
    fb.def(&format!("select i1 {c}, i32 {idx}, i32 {last}"))
}

/// Byte address `ptr + zext(idx) * stride`.
fn index_ptr(fb: &mut Fb, ptr: &Op, idx: &Op, stride: u32) -> Op {
    fb.gep_elem(ptr, idx, stride)
}

fn sreg(fb: &mut Fb, kind: &str, k: usize) -> Op {
    let axis = ["x", "y", "z"][k];
    fb.declare(&format!("declare i32 @llvm.nvvm.read.ptx.sreg.{kind}.{axis}()"));
    fb.def(&format!("call i32 @llvm.nvvm.read.ptx.sreg.{kind}.{axis}()"))
}

impl<'a> Em<'a> {
    /// Gives the builtin local `l` its lanes (`u32`s): as an SSA value, or stored in its slot.
    fn bind_local(&self, fb: &mut Fb, cx: &FnCx, l: LocalId, vals: Vec<Op>) {
        let li = l.0 as usize;
        if cx.ssa[li] {
            // Only scalars are SSA values: one lane.
            let elem = cx.local_decls[li].1.elem_scalar().unwrap_or(ScalarTy::U32);
            cx.ssa_vals.borrow_mut()[li] = Some(Val::Lanes { elem, ops: vals.into_iter().take(1).collect() });
        } else {
            for (k, v) in vals.iter().enumerate() {
                fb.store(&cx.local_ptrs[li], 4 * k as u32, ScalarTy::U32, v, true);
            }
        }
    }

    fn sz(&self, ty: &KirType) -> u32 {
        size_of(ty, &self.m.structs)
    }

    fn al(&self, ty: &KirType) -> u32 {
        align_of(ty, &self.m.structs)
    }

    fn expr_ty(&self, e: &Expr, cx: &FnCx) -> KirType {
        match e {
            Expr::LitBool(_) => KirType::BOOL,
            Expr::LitU32(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => KirType::U32,
            Expr::LitI32(_) => KirType::I32,
            Expr::LitF16(_) => KirType::Scalar(ScalarTy::F16),
            Expr::LitF32(_) => KirType::F32,
            Expr::LitF64(_) => KirType::Scalar(ScalarTy::F64),
            Expr::ZeroVal(t) => t.clone(),
            Expr::Local(l) => cx.local_decls[l.0 as usize].1.clone(),
            Expr::FnParam(i) => cx.param_decls[*i as usize].1.clone(),
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

    /// Type reached from `base` by a field (`Some(i)`) or an index (`None`) step.
    fn step_ty(&self, base: &KirType, field: Option<u32>) -> KirType {
        match (base, field) {
            (KirType::Struct(s), Some(i)) => self.m.structs[*s as usize].fields[i as usize].ty.clone(),
            (KirType::Vec { elem, .. }, _) => KirType::Scalar(*elem),
            (KirType::Array { elem, .. }, None) => (**elem).clone(),
            _ => base.clone(),
        }
    }

    // ----- memory ---------------------------------------------------------------------------

    /// Reads the value of type `ty` at `ptr`. Aggregates are copied into a fresh temporary.
    fn read_place(&self, fb: &mut Fb, ptr: &Op, ty: &KirType, tbaa: bool) -> Val {
        match ty {
            KirType::Scalar(s) | KirType::Atomic(s) => scalar_val(*s, fb.load(ptr, 0, *s, tbaa)),
            KirType::Vec { elem, lanes } if Fb::vector_access(*elem, u32::from(*lanes)) => {
                Val::Lanes { elem: *elem, ops: fb.load_vec(ptr, *elem, u32::from(*lanes), tbaa) }
            }
            KirType::Vec { elem, lanes } => {
                let ops = (0..u32::from(*lanes)).map(|k| fb.load(ptr, k * elem.size_bytes(), *elem, tbaa)).collect();
                Val::Lanes { elem: *elem, ops }
            }
            KirType::Array { .. } | KirType::Struct(_) => {
                let (size, align) = (self.sz(ty), self.al(ty));
                let t = fb.alloca(size, align);
                self.copy(fb, &t, ptr, ty);
                Val::Mem { ptr: t, ty: ty.clone() }
            }
        }
    }

    /// Stores `v` into the place of type `ty` at `ptr`.
    fn write_place(&self, fb: &mut Fb, ptr: &Op, ty: &KirType, v: &Val, tbaa: bool) {
        match v {
            Val::Lanes { elem, ops } if matches!(ty, KirType::Vec { .. }) && Fb::vector_access(*elem, ops.len() as u32) => {
                fb.store_vec(ptr, *elem, ops, tbaa);
            }
            Val::Lanes { elem, ops } => {
                for (k, op) in ops.iter().enumerate() {
                    fb.store(ptr, k as u32 * elem.size_bytes(), *elem, op, tbaa);
                }
            }
            Val::Mem { ptr: src, .. } => self.copy(fb, ptr, src, ty),
        }
    }

    fn zero_val(&self, fb: &mut Fb, ty: &KirType) -> Val {
        match ty {
            KirType::Scalar(s) | KirType::Atomic(s) => scalar_val(*s, ops::zero(*s)),
            KirType::Vec { elem, lanes } => Val::Lanes { elem: *elem, ops: vec![ops::zero(*elem); *lanes as usize] },
            KirType::Array { .. } | KirType::Struct(_) => {
                let (size, align) = (self.sz(ty), self.al(ty));
                let t = fb.alloca(size, align);
                self.zero(fb, &t, ty);
                Val::Mem { ptr: t, ty: ty.clone() }
            }
        }
    }

    /// Copies the aggregate of type `ty` from `src` to `dst`. Small aggregates are copied leaf by
    /// leaf with typed accesses: NVVM's SROA does not split an `llvm.memcpy` between slots that
    /// are also accessed as vectors, which leaves matrices in local memory.
    fn copy(&self, fb: &mut Fb, dst: &Op, src: &Op, ty: &KirType) {
        if dst == src {
            return;
        }
        if self.sz(ty) > LEAF_COPY_BYTES {
            fb.memcpy(dst, src, self.sz(ty), self.al(ty));
            return;
        }
        self.leaves(ty, 0, &mut |fb: &mut Fb, off, leaf| {
            let (s, p, q) = (leaf, fb.gep_const(src, u64::from(off)), fb.gep_const(dst, u64::from(off)));
            match s {
                Leaf::Scalar(s) => {
                    let v = fb.load(&p, 0, s, true);
                    fb.store(&q, 0, s, &v, true);
                }
                Leaf::Vec(s, n) => {
                    let v = fb.load_vec(&p, s, n, true);
                    fb.store_vec(&q, s, &v, true);
                }
            }
        }, fb);
    }

    /// Zeroes the aggregate of type `ty` at `dst`; see [`Self::copy`].
    fn zero(&self, fb: &mut Fb, dst: &Op, ty: &KirType) {
        if self.sz(ty) > LEAF_COPY_BYTES {
            fb.memzero(dst, self.sz(ty), self.al(ty));
            return;
        }
        self.leaves(ty, 0, &mut |fb: &mut Fb, off, leaf| {
            let q = fb.gep_const(dst, u64::from(off));
            match leaf {
                Leaf::Scalar(s) => fb.store(&q, 0, s, &ops::zero(s), true),
                Leaf::Vec(s, n) => fb.store_vec(&q, s, &vec![ops::zero(s); n as usize], true),
            }
        }, fb);
    }

    /// Calls `f` for every scalar or whole-vector leaf of `ty`, at its byte offset from `off`.
    fn leaves(&self, ty: &KirType, off: u32, f: &mut dyn FnMut(&mut Fb, u32, Leaf), fb: &mut Fb) {
        match ty {
            KirType::Scalar(s) | KirType::Atomic(s) => f(fb, off, Leaf::Scalar(*s)),
            KirType::Vec { elem, lanes } => {
                let n = u32::from(*lanes);
                if Fb::vector_access(*elem, n) && off.is_multiple_of(n * elem.size_bytes()) {
                    f(fb, off, Leaf::Vec(*elem, n));
                } else {
                    for k in 0..n {
                        f(fb, off + k * elem.size_bytes(), Leaf::Scalar(*elem));
                    }
                }
            }
            KirType::Struct(s) => {
                for field in &self.m.structs[*s as usize].fields {
                    self.leaves(&field.ty, off + field.offset, f, fb);
                }
            }
            KirType::Array { elem, len, stride } => {
                for k in 0..*len {
                    self.leaves(elem, off + k * stride, f, fb);
                }
            }
        }
    }

    fn step_place(&self, fb: &mut Fb, ptr: &Op, ty: &KirType, step: Step) -> (Op, KirType) {
        match (ty, step) {
            (KirType::Struct(s), Step::Field(i)) => {
                let off = self.m.structs[*s as usize].fields[i as usize].offset;
                (fb.gep_const(ptr, u64::from(off)), self.step_ty(ty, Some(i)))
            }
            (KirType::Vec { elem, .. }, Step::Field(i)) => (fb.gep_const(ptr, u64::from(i * elem.size_bytes())), KirType::Scalar(*elem)),
            (KirType::Vec { elem, lanes }, Step::Index(idx)) => {
                let j = umin_const(fb, &idx, u32::from(*lanes) - 1);
                (index_ptr(fb, ptr, &j, elem.size_bytes()), KirType::Scalar(*elem))
            }
            (KirType::Array { len, stride, .. }, Step::Index(idx)) => {
                // `luma_arr`: out-of-range indices clamp to the last element.
                let c = fb.def(&format!("icmp ult i32 {idx}, {len}"));
                let last = if *len == 0 { 0 } else { len - 1 };
                let j = fb.def(&format!("select i1 {c}, i32 {idx}, i32 {last}"));
                (index_ptr(fb, ptr, &j, *stride), self.step_ty(ty, None))
            }
            (_, Step::Field(i)) => (ptr.clone(), self.step_ty(ty, Some(i))),
            (_, Step::Index(_)) => (ptr.clone(), ty.clone()),
        }
    }

    fn eval_step(&mut self, fb: &mut Fb, s: &AccessStep, cx: &FnCx) -> Step {
        match s {
            AccessStep::Field(i) => Step::Field(*i),
            AccessStep::Index(e) => Step::Index(self.index(fb, e, cx)),
        }
    }

    /// An index, a scalar `i32`.
    fn index(&mut self, fb: &mut Fb, e: &Expr, cx: &FnCx) -> Op {
        one(&self.expr(fb, e, cx)).1
    }

    /// Applies an access path to the place `ptr: ty`.
    fn path(&mut self, fb: &mut Fb, mut ptr: Op, mut ty: KirType, steps: &[AccessStep], cx: &FnCx) -> (Op, KirType) {
        for step in steps {
            let st = self.eval_step(fb, step, cx);
            (ptr, ty) = self.step_place(fb, &ptr, &ty, st);
        }
        (ptr, ty)
    }

    fn bind_ptr(&self, binding: u32, cx: &FnCx) -> Op {
        cx.binds.get(&binding).cloned().unwrap_or_else(|| invariant(format!("binding {binding} is not available in this function")))
    }

    /// Address of element `idx` of binding `binding`.
    fn bind_elem(&self, fb: &mut Fb, binding: u32, idx: &Op, cx: &FnCx) -> Op {
        let d = &self.m.bindings[binding as usize];
        let stride = d.element_stride.unwrap_or(self.sz(&d.elem_ty));
        let base = self.bind_ptr(binding, cx);
        index_ptr(fb, &base, idx, stride)
    }

    fn storage_path(&mut self, fb: &mut Fb, binding: u32, steps: &[AccessStep], cx: &FnCx) -> (Op, KirType) {
        let m = self.m;
        let d = &m.bindings[binding as usize];
        if d.is_unsized_array {
            match steps.split_first() {
                Some((AccessStep::Index(i), rest)) => {
                    let iv = self.index(fb, i, cx);
                    let p = self.bind_elem(fb, binding, &iv, cx);
                    self.path(fb, p, d.elem_ty.clone(), rest, cx)
                }
                _ => invariant(format!("an unsized storage array binding `{}` must be indexed first", d.name)),
            }
        } else {
            let base = self.bind_ptr(binding, cx);
            self.path(fb, base, d.elem_ty.clone(), steps, cx)
        }
    }

    fn shared_ptr(&self, shared: u32, cx: &FnCx) -> Op {
        cx.shared.get(&shared).cloned().unwrap_or_else(|| invariant(format!("shared variable {shared} is not available in this function")))
    }

    fn shared_stride(&self, shared: u32) -> u32 {
        let e = &self.m.shared[shared as usize].elem_ty;
        self.sz(e).next_multiple_of(self.al(e))
    }

    /// Address of `shared[idx]`; the variable itself when it is not an array.
    fn shared_elem(&self, fb: &mut Fb, shared: u32, idx: &Op, cx: &FnCx) -> Op {
        let base = self.shared_ptr(shared, cx);
        match self.m.shared[shared as usize].len {
            // Untrusted kernels keep WGSL's clamped indexing; trusted ones index directly.
            Some(n) => {
                let j = if self.init_shared { umin_const(fb, idx, n.wrapping_sub(1)) } else { idx.clone() };
                index_ptr(fb, &base, &j, self.shared_stride(shared))
            }
            None => base,
        }
    }

    fn shared_root_ty(&self, shared: u32) -> KirType {
        let d = &self.m.shared[shared as usize];
        match d.len {
            Some(len) => KirType::Array { elem: Box::new(d.elem_ty.clone()), len, stride: self.shared_stride(shared) },
            None => d.elem_ty.clone(),
        }
    }

    fn shared_path(&mut self, fb: &mut Fb, shared: u32, steps: &[AccessStep], cx: &FnCx) -> (Op, KirType) {
        let m = self.m;
        let elem = m.shared[shared as usize].elem_ty.clone();
        let base = self.shared_ptr(shared, cx);
        if m.shared[shared as usize].len.is_some() {
            match steps.split_first() {
                Some((AccessStep::Index(i), rest)) => {
                    let iv = self.index(fb, i, cx);
                    let p = self.shared_elem(fb, shared, &iv, cx);
                    self.path(fb, p, elem, rest, cx)
                }
                _ => (base, self.shared_root_ty(shared)),
            }
        } else {
            self.path(fb, base, elem, steps, cx)
        }
    }

    fn lvalue(&mut self, fb: &mut Fb, lv: &LValue, cx: &FnCx) -> (Op, KirType) {
        match lv {
            LValue::Local(l) => (cx.local_ptrs[l.0 as usize].clone(), cx.local_decls[l.0 as usize].1.clone()),
            LValue::Private(p) => {
                let ptr = cx.privs.get(p).cloned().unwrap_or_else(|| invariant(format!("private {p} is not available in this function")));
                (ptr, self.m.privates[*p as usize].ty.clone())
            }
            LValue::Storage { binding, path } => self.storage_path(fb, *binding, path, cx),
            LValue::Shared { shared, path } => self.shared_path(fb, *shared, path, cx),
            LValue::Index { base, index } => {
                let (b, bty) = self.lvalue(fb, base, cx);
                let iv = self.index(fb, index, cx);
                self.step_place(fb, &b, &bty, Step::Index(iv))
            }
            LValue::Field { base, field } => {
                let (b, bty) = self.lvalue(fb, base, cx);
                self.step_place(fb, &b, &bty, Step::Field(*field))
            }
        }
    }

    fn atomic_ptr(&mut self, fb: &mut Fb, t: &AtomicTarget, cx: &FnCx) -> Op {
        match t {
            AtomicTarget::Storage { binding, index: Some(i) } => {
                let iv = self.expr(fb, i, cx);
                self.bind_elem(fb, *binding, &one(&iv).1, cx)
            }
            AtomicTarget::Storage { binding, index: None } => self.bind_ptr(*binding, cx),
            AtomicTarget::Shared { shared, index: Some(i) } => {
                let iv = self.expr(fb, i, cx);
                self.shared_elem(fb, *shared, &one(&iv).1, cx)
            }
            AtomicTarget::Shared { shared, index: None } => self.shared_ptr(*shared, cx),
        }
    }

    // ----- expressions ----------------------------------------------------------------------

    /// The memory place an aggregate-typed expression denotes, without reading it; `None` (and
    /// nothing emitted) when `e` is not a place.
    fn place(&mut self, fb: &mut Fb, e: &Expr, cx: &FnCx) -> Option<(Op, KirType)> {
        let m = self.m;
        Some(match e {
            Expr::Local(l) => (cx.local_ptrs[l.0 as usize].clone(), cx.local_decls[l.0 as usize].1.clone()),
            Expr::Private(p) => {
                let ptr = cx.privs.get(p).cloned().unwrap_or_else(|| invariant(format!("private {p} is not available in this function")));
                (ptr, m.privates[*p as usize].ty.clone())
            }
            Expr::UniformRead { binding, path, .. } => {
                let base = self.bind_ptr(*binding, cx);
                self.path(fb, base, m.bindings[*binding as usize].elem_ty.clone(), path, cx)
            }
            Expr::Load { binding, index, .. } => {
                let iv = self.index(fb, index, cx);
                (self.bind_elem(fb, *binding, &iv, cx), m.bindings[*binding as usize].elem_ty.clone())
            }
            Expr::StorageRead { binding, path, .. } => self.storage_path(fb, *binding, path, cx),
            Expr::SharedLoad { shared, index, .. } => {
                let iv = self.index(fb, index, cx);
                (self.shared_elem(fb, *shared, &iv, cx), m.shared[*shared as usize].elem_ty.clone())
            }
            Expr::SharedRead { shared, path, .. } => self.shared_path(fb, *shared, path, cx),
            Expr::Access { base, step, .. } if Self::is_place(base) && matches!(self.expr_ty(base, cx), KirType::Array { .. } | KirType::Struct(_)) => {
                let (ptr, ty) = self.place(fb, base, cx)?;
                let st = self.eval_step(fb, step, cx);
                self.step_place(fb, &ptr, &ty, st)
            }
            _ => return None,
        })
    }

    fn is_place(e: &Expr) -> bool {
        match e {
            Expr::Local(_) | Expr::Private(_) | Expr::UniformRead { .. } | Expr::Load { .. } | Expr::StorageRead { .. } | Expr::SharedLoad { .. } | Expr::SharedRead { .. } => true,
            Expr::Access { base, .. } => Self::is_place(base),
            _ => false,
        }
    }

    fn expr(&mut self, fb: &mut Fb, e: &Expr, cx: &FnCx) -> Val {
        let m = self.m;
        match e {
            Expr::LitBool(b) => scalar_val(ScalarTy::Bool, b.to_string()),
            Expr::LitU32(v) => scalar_val(ScalarTy::U32, v.to_string()),
            Expr::LitI32(v) => scalar_val(ScalarTy::I32, v.to_string()),
            Expr::LitF16(v) => {
                fb.declare("declare i16 @llvm.convert.to.fp16.f32(float)");
                let o = fb.def(&format!("call i16 @llvm.convert.to.fp16.f32(float {})", ops::f32_lit(*v)));
                scalar_val(ScalarTy::F16, o)
            }
            Expr::LitF32(v) => scalar_val(ScalarTy::F32, ops::f32_lit(*v)),
            Expr::LitF64(v) => scalar_val(ScalarTy::F64, ops::f64_lit(*v)),
            Expr::ZeroVal(t) => self.zero_val(fb, t),
            Expr::Local(l) if cx.ssa[l.0 as usize] => {
                cx.ssa_vals.borrow()[l.0 as usize].clone().unwrap_or_else(|| invariant(format!("local {} is used before it is declared", l.0)))
            }
            Expr::Local(l) => {
                let ty = &cx.local_decls[l.0 as usize].1;
                self.read_place(fb, &cx.local_ptrs[l.0 as usize], ty, true)
            }
            Expr::FnParam(i) => cx.param_vals[*i as usize].clone(),
            Expr::Private(p) => {
                let ptr = cx.privs.get(p).cloned().unwrap_or_else(|| invariant(format!("private {p} is not available in this function")));
                self.read_place(fb, &ptr, &m.privates[*p as usize].ty, true)
            }
            Expr::GridSize(a) => {
                let g = cx.grid.clone().unwrap_or_else(|| invariant("the launch grid is not available in this function"));
                scalar_val(ScalarTy::U32, fb.load(&g, 4 * u32::from(*a), ScalarTy::U32, true))
            }
            Expr::UniformRead { binding, path, .. } => {
                let base = self.bind_ptr(*binding, cx);
                let (p, t) = self.path(fb, base, m.bindings[*binding as usize].elem_ty.clone(), path, cx);
                self.read_place(fb, &p, &t, true)
            }
            Expr::Load { binding, index, .. } => {
                let iv = self.index(fb, index, cx);
                let p = self.bind_elem(fb, *binding, &iv, cx);
                self.read_place(fb, &p, &m.bindings[*binding as usize].elem_ty, true)
            }
            Expr::StorageRead { binding, path, .. } => {
                let (p, t) = self.storage_path(fb, *binding, path, cx);
                self.read_place(fb, &p, &t, true)
            }
            Expr::SharedLoad { shared, index, .. } => {
                let iv = self.index(fb, index, cx);
                let p = self.shared_elem(fb, *shared, &iv, cx);
                self.read_place(fb, &p, &m.shared[*shared as usize].elem_ty, true)
            }
            Expr::SharedRead { shared, path, .. } => {
                let (p, t) = self.shared_path(fb, *shared, path, cx);
                self.read_place(fb, &p, &t, true)
            }
            Expr::Access { base, step, .. } => {
                // An aggregate place is stepped into and only the leaf is read: copying the whole
                // aggregate first keeps arrays indexed dynamically out of registers.
                if matches!(self.expr_ty(base, cx), KirType::Array { .. } | KirType::Struct(_))
                    && let Some((ptr, ty)) = self.place(fb, base, cx)
                {
                    let st = self.eval_step(fb, step, cx);
                    let (p, t) = self.step_place(fb, &ptr, &ty, st);
                    return self.read_place(fb, &p, &t, true);
                }
                let bv = self.expr(fb, base, cx);
                match bv {
                    Val::Lanes { elem, ops } => match step {
                        AccessStep::Field(k) => scalar_val(elem, ops[*k as usize].clone()),
                        AccessStep::Index(ie) => {
                            let iv = self.expr(fb, ie, cx);
                            let idx = one(&iv).1;
                            let t = llty(elem);
                            let mut r = ops[ops.len() - 1].clone();
                            for l in (0..ops.len() - 1).rev() {
                                let c = fb.def(&format!("icmp eq i32 {idx}, {l}"));
                                r = fb.def(&format!("select i1 {c}, {t} {}, {t} {r}", ops[l]));
                            }
                            scalar_val(elem, r)
                        }
                    },
                    Val::Mem { ptr, ty } => {
                        let st = self.eval_step(fb, step, cx);
                        let (p, t) = self.step_place(fb, &ptr, &ty, st);
                        self.read_place(fb, &p, &t, true)
                    }
                }
            }
            Expr::Swizzle { base, indices, len, .. } => {
                let bv = self.expr(fb, base, cx);
                let (elem, o) = lanes_of(&bv);
                let picked = indices[..*len as usize].iter().map(|i| o[*i as usize].clone()).collect();
                Val::Lanes { elem, ops: picked }
            }
            Expr::Compose { ty, comps } => self.compose(fb, ty, comps, cx),
            Expr::Splat { elem, lanes, val } => {
                let v = self.expr(fb, val, cx);
                let o = one(&v).1;
                Val::Lanes { elem: *elem, ops: vec![o; *lanes as usize] }
            }
            Expr::Unary { op, arg, .. } => {
                let a = self.expr(fb, arg, cx);
                let (s, o) = lanes_of(&a);
                if matches!(op, IrUnOp::All | IrUnOp::Any) {
                    let j = if *op == IrUnOp::All { "and" } else { "or" };
                    let mut acc = o[0].clone();
                    for x in &o[1..] {
                        acc = fb.def(&format!("{j} i1 {acc}, {x}"));
                    }
                    return scalar_val(ScalarTy::Bool, acc);
                }
                let mut rt = s;
                let mut out = Vec::with_capacity(o.len());
                for x in o {
                    let (t, r) = ops::unary(fb, *op, s, x);
                    rt = t;
                    out.push(r);
                }
                Val::Lanes { elem: rt, ops: out }
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let a = self.expr(fb, lhs, cx);
                let b = self.expr(fb, rhs, cx);
                let (ls, ao) = lanes_of(&a);
                let (rs, bo) = lanes_of(&b);
                let n = ao.len().max(bo.len());
                let mut rt = ls;
                let mut out = Vec::with_capacity(n);
                for k in 0..n {
                    let (t, r) = ops::binary(fb, *op, ls, rs, pick(ao, k), pick(bo, k));
                    rt = t;
                    out.push(r);
                }
                Val::Lanes { elem: rt, ops: out }
            }
            Expr::Ternary { op: IrTernaryOp::Select, a, b, c, .. } if self.expr_ty(a, cx).lanes() == 1 => {
                // A scalar condition only evaluates the chosen operand, as the C++ `?:` did: KIR
                // guards out-of-range binding loads this way, and the operands of a solver's
                // selects are often whole expressions (an eager select measured 6-9 % slower on
                // bowshock's steps). The result is a `phi`, which LLVM if-converts when both sides
                // are cheap. Not a stack slot: NVVM (driver 615) miscompiled slot round-trips here.
                let cond = one(&self.expr(fb, a, cx)).1;
                let (then_l, else_l, end_l) = (fb.label(), fb.label(), fb.label());
                fb.cond_br(&cond, &then_l, &else_l);
                fb.start(&then_l);
                let bv = self.expr(fb, b, cx);
                let then_end = fb.cur.clone();
                fb.br(&end_l);
                fb.start(&else_l);
                let cv = self.expr(fb, c, cx);
                let else_end = fb.cur.clone();
                fb.br(&end_l);
                fb.start(&end_l);
                match (bv, cv) {
                    (Val::Mem { ptr: pb, ty }, Val::Mem { ptr: pc, .. }) => {
                        // Both are temporaries nothing else writes.
                        Val::Mem { ptr: fb.def(&format!("phi i8* [ {pb}, %{then_end} ], [ {pc}, %{else_end} ]")), ty }
                    }
                    (Val::Lanes { elem, ops: bo }, Val::Lanes { ops: co, .. }) => {
                        let t = llty(elem);
                        let n = bo.len().max(co.len());
                        let ops = (0..n).map(|k| fb.def(&format!("phi {t} [ {}, %{then_end} ], [ {}, %{else_end} ]", pick(&bo, k), pick(&co, k)))).collect();
                        Val::Lanes { elem, ops }
                    }
                    _ => invariant("select operands of different shapes"),
                }
            }
            Expr::Ternary { op, a, b, c, .. } => {
                let av = self.expr(fb, a, cx);
                let bv = self.expr(fb, b, cx);
                let cv = self.expr(fb, c, cx);
                match op {
                    IrTernaryOp::Select => match (&bv, &cv) {
                        (Val::Mem { ptr: pb, ty }, Val::Mem { ptr: pc, .. }) => {
                            let cond = one(&av).1;
                            let sel = fb.def(&format!("select i1 {cond}, i8* {pb}, i8* {pc}"));
                            let (size, align) = (self.sz(ty), self.al(ty));
                            let t = fb.alloca(size, align);
                            self.copy(fb, &t, &sel, ty);
                            Val::Mem { ptr: t, ty: ty.clone() }
                        }
                        _ => {
                            let (cs, co) = lanes_of(&av);
                            debug_assert_eq!(cs, ScalarTy::Bool);
                            let (elem, bo) = lanes_of(&bv);
                            let (_, co2) = lanes_of(&cv);
                            let n = bo.len().max(co2.len()).max(co.len());
                            let t = llty(elem);
                            let out = (0..n)
                                .map(|k| fb.def(&format!("select i1 {}, {t} {}, {t} {}", pick(co, k), pick(bo, k), pick(co2, k))))
                                .collect();
                            Val::Lanes { elem, ops: out }
                        }
                    },
                    IrTernaryOp::Fma | IrTernaryOp::Clamp => {
                        let (s, ao) = lanes_of(&av);
                        let (_, bo) = lanes_of(&bv);
                        let (_, co) = lanes_of(&cv);
                        let n = ao.len().max(bo.len()).max(co.len());
                        let out = (0..n)
                            .map(|k| {
                                if *op == IrTernaryOp::Fma {
                                    ops::fma(fb, s, pick(ao, k), pick(bo, k), pick(co, k))
                                } else {
                                    ops::clamp(fb, s, pick(ao, k), pick(bo, k), pick(co, k))
                                }
                            })
                            .collect();
                        Val::Lanes { elem: s, ops: out }
                    }
                }
            }
            Expr::Cast { kind, expr, src_ty, dst_ty } => {
                let v = self.expr(fb, expr, cx);
                self.cast(fb, *kind, v, src_ty, dst_ty)
            }
            Expr::Call { func, args, .. } => self.call(fb, func.0 as usize, args, cx),
            Expr::AtomicLoad { target, ty } => {
                let p = self.atomic_ptr(fb, target, cx);
                scalar_val(*ty, ops::atomic_load(fb, *ty, &p))
            }
            Expr::Subgroup { op, arg, lane, .. } => {
                let converged = self.converged.contains(&(e as *const Expr));
                let a = arg.as_ref().map(|a| self.expr(fb, a, cx));
                let l = lane.as_ref().map(|l| one(&self.expr(fb, l, cx)).1);
                if *op == IrSubgroupOp::Ballot {
                    let pred = a.as_ref().map(|a| one(a).1);
                    let b = ops::ballot(fb, pred.as_ref(), converged);
                    return Val::Lanes { elem: ScalarTy::U32, ops: vec![b, "0".into(), "0".into(), "0".into()] };
                }
                let a = a.unwrap_or_else(|| invariant("a subgroup reduction needs an argument"));
                let (s, o) = lanes_of(&a);
                let lane = if *op == IrSubgroupOp::Shuffle { l } else { None };
                let out = o.iter().map(|x| ops::subgroup(fb, *op, s, x, lane.as_ref(), converged)).collect();
                Val::Lanes { elem: s, ops: out }
            }
            Expr::ArrayLength { binding } => {
                let l = cx.lens.get(binding).cloned().unwrap_or_else(|| invariant(format!("length of binding {binding} is not available")));
                scalar_val(ScalarTy::U32, l)
            }
        }
    }

    fn compose(&mut self, fb: &mut Fb, ty: &KirType, comps: &[Expr], cx: &FnCx) -> Val {
        let vals: Vec<Val> = comps.iter().map(|c| self.expr(fb, c, cx)).collect();
        match ty {
            KirType::Vec { elem, .. } => {
                let mut out = Vec::new();
                for v in &vals {
                    out.extend(lanes_of(v).1.iter().cloned());
                }
                Val::Lanes { elem: *elem, ops: out }
            }
            KirType::Array { elem, stride, .. } => {
                let z = self.zero_val(fb, ty);
                let Val::Mem { ptr, .. } = &z else { invariant("array compose") };
                for (k, v) in vals.iter().enumerate() {
                    let p = fb.gep_const(ptr, u64::from(k as u32 * stride));
                    self.write_place(fb, &p, elem, v, true);
                }
                z
            }
            KirType::Struct(si) => {
                let d = &self.m.structs[*si as usize];
                assert_eq!(d.fields.len(), vals.len(), "luma-cuda codegen invariant: struct compose arity");
                let z = self.zero_val(fb, ty);
                let Val::Mem { ptr, .. } = &z else { invariant("struct compose") };
                for (f, v) in d.fields.iter().zip(&vals) {
                    let p = fb.gep_const(ptr, u64::from(f.offset));
                    self.write_place(fb, &p, &f.ty, v, true);
                }
                z
            }
            other => invariant(format!("compose of {other:?}")),
        }
    }

    fn cast(&mut self, fb: &mut Fb, kind: CastKind, v: Val, src: &KirType, dst: &KirType) -> Val {
        let ss = src.elem_scalar().unwrap_or(ScalarTy::F32);
        let ds = dst.elem_scalar().unwrap_or(ScalarTy::F32);
        match kind {
            CastKind::Bitcast => {
                if let Val::Lanes { ops: o, .. } = &v
                    && o.len() == dst.lanes() as usize
                    && let Some(first) = ops::bitcast_scalar(fb, ss, ds, &o[0])
                {
                    let mut out = vec![first];
                    for x in &o[1..] {
                        out.push(ops::bitcast_scalar(fb, ss, ds, x).unwrap_or_else(|| invariant("bitcast lanes disagree")));
                    }
                    return Val::Lanes { elem: ds, ops: out };
                }
                let size = self.sz(src).max(self.sz(dst));
                let align = self.al(src).max(self.al(dst));
                let tmp = fb.alloca(size, align);
                self.write_place(fb, &tmp, src, &v, false);
                self.read_place(fb, &tmp, dst, false)
            }
            CastKind::Convert | CastKind::FloatToIntSat => {
                let (_, o) = lanes_of(&v);
                let out = o.iter().map(|x| ops::convert(fb, kind, ss, ds, x)).collect();
                Val::Lanes { elem: ds, ops: out }
            }
        }
    }

    fn call(&mut self, fb: &mut Fb, func: usize, args: &[Expr], cx: &FnCx) -> Val {
        let m = self.m;
        let f = &m.functions[func];
        let mut xs: Vec<String> = Vec::new();
        let hidden = match &f.ret_ty {
            Some(t) if !returns_direct(t) => {
                let p = fb.alloca(self.sz(t), self.al(t));
                xs.push(format!("i8* {p}"));
                Some((p, t.clone()))
            }
            _ => None,
        };
        for a in args {
            match self.expr(fb, a, cx) {
                Val::Lanes { elem, ops } => xs.extend(ops.iter().map(|o| format!("{} {o}", llty(elem)))),
                Val::Mem { ptr, .. } => xs.push(format!("i8* {ptr}")),
            }
        }
        let r = self.res[func].clone();
        xs.extend(self.ctx_args(&r, cx));
        let args = xs.join(", ");
        match (&f.ret_ty, hidden) {
            (None, _) => {
                fb.ins(&format!("call void @f{func}({args})"));
                scalar_val(ScalarTy::Bool, "false".into())
            }
            (Some(KirType::Scalar(s) | KirType::Atomic(s)), _) => {
                let o = fb.def(&format!("call {} @f{func}({args})", llty(*s)));
                scalar_val(*s, o)
            }
            (Some(_), Some((p, t))) => {
                fb.ins(&format!("call void @f{func}({args})"));
                match t {
                    KirType::Vec { .. } => self.read_place(fb, &p, &t, true),
                    // The slot is private to this call: no copy needed.
                    _ => Val::Mem { ptr: p, ty: t },
                }
            }
            (Some(_), None) => unreachable!(),
        }
    }

    /// The arguments a helper function needs to be handed besides its own.
    fn ctx_args(&self, r: &Res, cx: &FnCx) -> Vec<String> {
        let mut v = Vec::new();
        if r.grid {
            let g = cx.grid.clone().unwrap_or_else(|| invariant("the launch grid is not available in this function"));
            v.push(format!("i8* {g}"));
        }
        for b in &r.binds {
            v.push(format!("i8* {}", self.bind_ptr(*b, cx)));
        }
        for s in &r.shared {
            v.push(format!("i8* {}", self.shared_ptr(*s, cx)));
        }
        for p in &r.privs {
            let o = cx.privs.get(p).unwrap_or_else(|| invariant(format!("private {p} is not available in this function")));
            v.push(format!("i8* {o}"));
        }
        for l in &r.lens {
            let o = cx.lens.get(l).unwrap_or_else(|| invariant(format!("length of binding {l} is not available")));
            v.push(format!("i32 {o}"));
        }
        v
    }

    /// Declares the context parameters of a function and fills `cx` with their names.
    fn ctx_params(&self, r: &Res, cx: &mut FnCx) -> Vec<String> {
        let mut v = Vec::new();
        if r.grid {
            v.push("i8* %grid".to_string());
            cx.grid = Some("%grid".into());
        }
        for b in &r.binds {
            let name = if self.m.bindings[*b as usize].kind == BindingKind::Uniform { format!("%uni{b}") } else { format!("%buf{b}") };
            v.push(format!("i8* {name}"));
            cx.binds.insert(*b, name);
        }
        for s in &r.shared {
            v.push(format!("i8* %sh{s}"));
            cx.shared.insert(*s, format!("%sh{s}"));
        }
        for p in &r.privs {
            v.push(format!("i8* %pv{p}"));
            cx.privs.insert(*p, format!("%pv{p}"));
        }
        for l in &r.lens {
            v.push(format!("i32 %_luma_len{l}"));
            cx.lens.insert(*l, format!("%_luma_len{l}"));
        }
        v
    }

    // ----- statements -----------------------------------------------------------------------

    fn stmts(&mut self, fb: &mut Fb, block: &[Stmt], cx: &FnCx) {
        for s in block {
            self.stmt(fb, s, cx);
        }
    }

    fn atomic_scalar_store(&self, fb: &mut Fb, cx: &FnCx, local: LocalId, s: ScalarTy, v: &Op) {
        fb.store(&cx.local_ptrs[local.0 as usize], 0, s, v, true);
    }

    fn stmt(&mut self, fb: &mut Fb, s: &Stmt, cx: &FnCx) {
        let m = self.m;
        match s {
            Stmt::Let { local, init } if cx.ssa[local.0 as usize] => {
                let v = self.expr(fb, init, cx);
                cx.ssa_vals.borrow_mut()[local.0 as usize] = Some(match (v, &cx.local_decls[local.0 as usize].1) {
                    // The slot would have been read back as the declared type.
                    (Val::Lanes { ops, .. }, KirType::Scalar(s)) => Val::Lanes { elem: *s, ops },
                    (v, _) => v,
                });
            }
            Stmt::Let { local, init } => {
                let v = self.expr(fb, init, cx);
                self.write_place(fb, &cx.local_ptrs[local.0 as usize], &cx.local_decls[local.0 as usize].1, &v, true);
            }
            Stmt::Var { local, init } => {
                let ptr = &cx.local_ptrs[local.0 as usize];
                let ty = &cx.local_decls[local.0 as usize].1;
                match init {
                    Some(e) => {
                        let v = self.expr(fb, e, cx);
                        self.write_place(fb, ptr, ty, &v, true);
                    }
                    None => match ty {
                        KirType::Array { .. } | KirType::Struct(_) => self.zero(fb, ptr, ty),
                        _ => {
                            let z = self.zero_val(fb, ty);
                            self.write_place(fb, ptr, ty, &z, true);
                        }
                    },
                }
            }
            Stmt::Assign { dst, val } => {
                let (p, ty) = self.lvalue(fb, dst, cx);
                let v = self.expr(fb, val, cx);
                self.write_place(fb, &p, &ty, &v, true);
            }
            Stmt::Store { binding, index, val } => {
                let i = self.expr(fb, index, cx);
                let v = self.expr(fb, val, cx);
                let p = self.bind_elem(fb, *binding, &one(&i).1, cx);
                self.write_place(fb, &p, &m.bindings[*binding as usize].elem_ty, &v, true);
            }
            Stmt::SharedStore { shared, index, val } => {
                let i = self.expr(fb, index, cx);
                let v = self.expr(fb, val, cx);
                let p = self.shared_elem(fb, *shared, &one(&i).1, cx);
                self.write_place(fb, &p, &m.shared[*shared as usize].elem_ty, &v, true);
            }
            Stmt::If { cond, then_body, else_body } => {
                let c = one(&self.expr(fb, cond, cx)).1;
                let then_l = fb.label();
                let else_l = if else_body.is_empty() { None } else { Some(fb.label()) };
                let merge = fb.label();
                fb.cond_br(&c, &then_l, else_l.as_deref().unwrap_or(&merge));
                fb.start(&then_l);
                self.stmts(fb, then_body, cx);
                fb.br(&merge);
                if let Some(el) = &else_l {
                    fb.start(el);
                    self.stmts(fb, else_body, cx);
                    fb.br(&merge);
                }
                fb.start(&merge);
            }
            Stmt::Loop { body, continuing, break_if } => {
                let head = fb.label();
                let cont = fb.label();
                let exit = fb.label();
                fb.start(&head);
                self.jumps.push(Jump { brk: exit.clone(), cont: Some(cont.clone()) });
                self.stmts(fb, body, cx);
                fb.start(&cont);
                self.stmts(fb, continuing, cx);
                match break_if {
                    Some(b) => {
                        let c = one(&self.expr(fb, b, cx)).1;
                        fb.cond_br(&c, &exit, &head);
                    }
                    None => fb.br(&head),
                }
                self.jumps.pop();
                fb.start(&exit);
            }
            Stmt::Switch { selector, cases, default } => {
                let sel = one(&self.expr(fb, selector, cx)).1;
                let labels: Vec<String> = cases.iter().map(|_| fb.label()).collect();
                let def_l = fb.label();
                let end = fb.label();
                let arms: String = cases
                    .iter()
                    .zip(&labels)
                    .map(|((c, _, _), l)| match c {
                        SwitchCase::I32(v) => format!("i32 {v}, label %{l} "),
                        SwitchCase::U32(v) => format!("i32 {v}, label %{l} "),
                    })
                    .collect();
                fb.ins(&format!("switch i32 {sel}, label %{def_l} [ {arms}]"));
                fb.terminated = true;
                let outer_cont = self.jumps.last().and_then(|j| j.cont.clone());
                self.jumps.push(Jump { brk: end.clone(), cont: outer_cont });
                for (k, (_, body, fall)) in cases.iter().enumerate() {
                    fb.start(&labels[k]);
                    self.stmts(fb, body, cx);
                    let next = if *fall { labels.get(k + 1).unwrap_or(&def_l) } else { &end };
                    fb.br(next);
                }
                fb.start(&def_l);
                self.stmts(fb, default, cx);
                fb.br(&end);
                self.jumps.pop();
                fb.start(&end);
            }
            Stmt::Break => {
                let l = self.jumps.last().unwrap_or_else(|| invariant("`break` outside a loop or switch")).brk.clone();
                fb.br(&l);
            }
            Stmt::Continue => {
                let l = self
                    .jumps
                    .last()
                    .and_then(|j| j.cont.clone())
                    .unwrap_or_else(|| invariant("`continue` outside a loop"));
                fb.br(&l);
            }
            Stmt::Return(None) => fb.ret("ret void"),
            Stmt::Return(Some(e)) => {
                let v = self.expr(fb, e, cx);
                match &cx.ret {
                    Some((p, ty)) => {
                        self.write_place(fb, p, ty, &v, true);
                        fb.ret("ret void");
                    }
                    None => {
                        let (s, o) = one(&v);
                        fb.ret(&format!("ret {} {o}", llty(s)));
                    }
                }
            }
            Stmt::Barrier(scope) => {
                fb.declare("declare void @llvm.nvvm.barrier0()");
                if !matches!(scope, BarrierScope::Workgroup) {
                    fb.declare("declare void @llvm.nvvm.membar.gl()");
                    fb.ins("call void @llvm.nvvm.membar.gl()");
                }
                fb.ins("call void @llvm.nvvm.barrier0()");
            }
            Stmt::AtomicStore { target, val } => {
                let s = self
                    .expr_ty(val, cx)
                    .elem_scalar()
                    .unwrap_or_else(|| invariant("atomic store of a non-scalar"));
                let p = self.atomic_ptr(fb, target, cx);
                let v = one(&self.expr(fb, val, cx)).1;
                ops::atomic_store(fb, s, &p, &v);
            }
            Stmt::AtomicRmw { dst, op, elem_ty, target, val } => {
                let p = self.atomic_ptr(fb, target, cx);
                let v = one(&self.expr(fb, val, cx)).1;
                let prev = ops::atomic_rmw(fb, *op, *elem_ty, &p, &v);
                if let Some(d) = dst {
                    self.atomic_scalar_store(fb, cx, *d, *elem_ty, &prev);
                }
            }
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, elem_ty, target, cmp, val } => {
                let p = self.atomic_ptr(fb, target, cx);
                let c = one(&self.expr(fb, cmp, cx)).1;
                let v = one(&self.expr(fb, val, cx)).1;
                let (old, exchanged) = ops::atomic_cas(fb, *elem_ty, &p, &c, &v);
                self.atomic_scalar_store(fb, cx, *dst_old, *elem_ty, &old);
                self.atomic_scalar_store(fb, cx, *dst_exchanged, ScalarTy::Bool, &exchanged);
            }
            Stmt::Discard(e) => {
                self.expr(fb, e, cx);
            }
        }
    }

    // ----- module-level pieces --------------------------------------------------------------

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
                Some(CudaArgAbi::InlineUniform { size }) => (size, align_of(&b.elem_ty, &m.structs) as usize),
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
            .map(|i| (size_of(&m.bindings[i].elem_ty, &m.structs) as usize, i))
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

    /// Moves what `fb` collected (declarations, helper functions) into the module and renders it.
    fn absorb(&mut self, mut fb: Fb) -> String {
        self.decls.append(&mut fb.decls);
        self.helpers.append(&mut fb.helpers);
        fb.finish()
    }

    fn function(&mut self, i: usize, f: &KirFunction) -> String {
        let r = self.res[i].clone();
        let mut fb = Fb::new();
        let mut cx = FnCx::new(&f.locals, &f.params);
        let mut params: Vec<String> = Vec::new();
        let rt = match &f.ret_ty {
            None => "void",
            Some(KirType::Scalar(s) | KirType::Atomic(s)) => llty(*s),
            Some(t) => {
                params.push("i8* %ret".into());
                cx.ret = Some(("%ret".into(), t.clone()));
                "void"
            }
        };
        for (k, (_, t)) in f.params.iter().enumerate() {
            match t {
                KirType::Scalar(s) | KirType::Atomic(s) => {
                    params.push(format!("{} %a{k}", llty(*s)));
                    cx.param_vals.push(scalar_val(*s, format!("%a{k}")));
                }
                KirType::Vec { elem, lanes } => {
                    let names: Vec<Op> = (0..*lanes).map(|l| format!("%a{k}_{l}")).collect();
                    params.extend(names.iter().map(|n| format!("{} {n}", llty(*elem))));
                    cx.param_vals.push(Val::Lanes { elem: *elem, ops: names });
                }
                _ => {
                    params.push(format!("i8* %a{k}"));
                    cx.param_vals.push(Val::Mem { ptr: format!("%a{k}"), ty: t.clone() });
                }
            }
        }
        params.extend(self.ctx_params(&r, &mut cx));
        for o in cx.grid.iter().chain(cx.binds.values()).chain(cx.shared.values()).chain(cx.privs.values()) {
            fb.mark_entry(o);
        }
        if let Some((ret, _)) = &cx.ret {
            fb.mark_entry(ret);
        }
        for v in &cx.param_vals {
            if let Val::Mem { ptr, .. } = v {
                fb.mark_entry(ptr);
            }
        }
        cx.ssa = ssa_locals(&f.locals, &f.body, &[]);
        cx.local_ptrs = f.locals.iter().enumerate().map(|(i, (_, t, _))| if cx.ssa[i] { Op::new() } else { fb.alloca(self.sz(t), self.al(t)) }).collect();
        self.jumps.clear();
        self.stmts(&mut fb, &f.body, &cx);
        if !fb.terminated {
            match &f.ret_ty {
                Some(KirType::Scalar(s) | KirType::Atomic(s)) => fb.ret(&format!("ret {} {}", llty(*s), ops::zero(*s))),
                _ => fb.ret("ret void"),
            }
        }
        let body = self.absorb(fb);
        format!("define internal {rt} @f{i}({}) alwaysinline nounwind {{\n{body}}}\n", params.join(", "))
    }

    /// The kernel definition and its function type (for the annotation node).
    fn kernel(&mut self, lens: &[u32]) -> (String, String, Vec<CudaArgAbi>) {
        let m = self.m;
        let [wx, wy, wz] = m.workgroup_size;
        let total = wx * wy * wz;
        let mut fb = Fb::new();
        let mut cx = FnCx::new(&m.entry_locals, &[]);
        let mut kparams: Vec<String> = vec!["%LumaGrid* byval align 4 %luma".into(), "i8* %luma_indirect".into()];
        let mut ktypes: Vec<String> = vec!["%LumaGrid*".into(), "i8*".into()];
        cx.grid = Some(fb.def_entry("bitcast %LumaGrid* %luma to i8*"));
        let ind: Op = "%luma_indirect".into();
        fb.mark_entry(&ind);
        let mut param_abis = Vec::new();
        for (i, d) in m.bindings.iter().enumerate() {
            if !self.active[i] {
                continue;
            }
            let abi = self.abi[i].unwrap_or(CudaArgAbi::DevicePtr);
            param_abis.push(abi);
            let i32_ = i as u32;
            match (d.kind, abi) {
                (BindingKind::StorageRead, _) => {
                    kparams.push(format!("i8* noalias %buf{i}"));
                    ktypes.push("i8*".into());
                    cx.binds.insert(i32_, format!("%buf{i}"));
                    fb.mark_entry(&format!("%buf{i}"));
                }
                (BindingKind::StorageReadWrite, _) => {
                    kparams.push(format!("i8* %buf{i}"));
                    ktypes.push("i8*".into());
                    cx.binds.insert(i32_, format!("%buf{i}"));
                    fb.mark_entry(&format!("%buf{i}"));
                }
                (BindingKind::Uniform, CudaArgAbi::InlineUniform { size }) => {
                    let n = size.max(1);
                    let a = align_of(&d.elem_ty, &m.structs);
                    kparams.push(format!("[{n} x i8]* byval align {a} %unib{i}"));
                    ktypes.push(format!("[{n} x i8]*"));
                    let p = fb.def_entry(&format!("bitcast [{n} x i8]* %unib{i} to i8*"));
                    cx.binds.insert(i32_, p);
                }
                (BindingKind::Uniform, _) => {
                    kparams.push(format!("i8* noalias %uni{i}"));
                    ktypes.push("i8*".into());
                    cx.binds.insert(i32_, format!("%uni{i}"));
                    fb.mark_entry(&format!("%uni{i}"));
                }
            }
        }
        for l in lens {
            kparams.push(format!("i32 %_luma_len{l}"));
            ktypes.push("i32".into());
            cx.lens.insert(*l, format!("%_luma_len{l}"));
        }

        // 1. Thread coordinates.
        // A dimension of one is never anything but 0 at launch.
        let w = [wx, wy, wz];
        let tid: Vec<Op> = (0..3).map(|k| if w[k] == 1 { "0".to_string() } else { sreg(&mut fb, "tid", k) }).collect();
        let ctaid: Vec<Op> = (0..3).map(|k| sreg(&mut fb, "ctaid", k)).collect();
        let nctaid: Vec<Op> = (0..3).map(|k| sreg(&mut fb, "nctaid", k)).collect();
        let t = fb.def(&format!("mul i32 {}, {wy}", tid[2]));
        let t = fb.def(&format!("add i32 {}, {t}", tid[1]));
        let t = fb.def(&format!("mul i32 {t}, {wx}"));
        let linear = fb.def(&format!("add i32 {}, {t}", tid[0]));

        // 2. Indirect-dispatch guard.
        let has_ind = fb.def(&format!("icmp ne i8* {ind}, null"));
        let chk = fb.label();
        let go = fb.label();
        let bail = fb.label();
        fb.cond_br(&has_ind, &chk, &go);
        fb.start(&chk);
        let mut oob: Op = "false".into();
        for (k, c) in ctaid.iter().enumerate() {
            let lim = fb.load(&ind, 4 * k as u32, ScalarTy::U32, true);
            let c = fb.def(&format!("icmp uge i32 {c}, {lim}"));
            oob = if k == 0 { c } else { fb.def(&format!("or i1 {oob}, {c}")) };
        }
        fb.cond_br(&oob, &bail, &go);
        fb.start(&bail);
        fb.ret("ret void");
        fb.start(&go);

        // 3. Workgroup memory, zero-initialised cooperatively.
        let mut any_init = false;
        for i in 0..m.shared.len() {
            let bytes = self.shared_bytes(i as u32);
            let n = bytes.max(1);
            let p = fb.def_entry(&format!(
                "addrspacecast i8 addrspace(3)* getelementptr inbounds ([{n} x i8], [{n} x i8] addrspace(3)* @sh{i}, i64 0, i64 0) to i8*"
            ));
            cx.shared.insert(i as u32, p.clone());
            if self.init_shared {
                any_init = true;
                let zp = fb.alloca(4, 4);
                fb.store(&zp, 0, ScalarTy::U32, &linear, false);
                let (head, body, exit) = (fb.label(), fb.label(), fb.label());
                fb.start(&head);
                let z = fb.load(&zp, 0, ScalarTy::U32, false);
                let c = fb.def(&format!("icmp ult i32 {z}, {bytes}"));
                fb.cond_br(&c, &body, &exit);
                fb.start(&body);
                let z64 = fb.def(&format!("zext i32 {z} to i64"));
                let q = fb.gep(&p, &z64);
                fb.ins(&format!("store i8 0, i8* {q}"));
                let zn = fb.def(&format!("add i32 {z}, {total}"));
                fb.store(&zp, 0, ScalarTy::U32, &zn, false);
                fb.br(&head);
                fb.start(&exit);
            }
        }
        if any_init {
            fb.declare("declare void @llvm.nvvm.barrier0()");
            fb.ins("call void @llvm.nvvm.barrier0()");
        }

        // 4. Privates and locals.
        for (i, p) in m.privates.iter().enumerate() {
            let ptr = fb.alloca(self.sz(&p.ty), self.al(&p.ty));
            cx.privs.insert(i as u32, ptr);
        }
        let builtin_locals: Vec<LocalId> = m.entry_builtins.iter().map(|(_, l)| *l).collect();
        cx.ssa = ssa_locals(&m.entry_locals, &m.entry_body, &builtin_locals);
        cx.local_ptrs = m.entry_locals.iter().enumerate().map(|(i, (_, t, _))| if cx.ssa[i] { Op::new() } else { fb.alloca(self.sz(t), self.al(t)) }).collect();
        for (i, p) in m.privates.iter().enumerate() {
            let ptr = cx.privs[&(i as u32)].clone();
            match &p.init {
                Some(e) => {
                    let v = self.expr(&mut fb, e, &cx);
                    self.write_place(&mut fb, &ptr, &p.ty, &v, true);
                }
                None => self.zero(&mut fb, &ptr, &p.ty),
            }
        }

        // 5. Builtins: SSA values where the local is never written, else stored to its slot.
        for (b, l) in &m.entry_builtins {
            let vals: Vec<Op> = match b {
                BuiltinVar::GlobalInvocationId => (0..3)
                    .map(|k| {
                        let t = fb.def(&format!("mul i32 {}, {}", ctaid[k], w[k]));
                        fb.def(&format!("add i32 {t}, {}", tid[k]))
                    })
                    .collect(),
                BuiltinVar::LocalInvocationId => tid.clone(),
                BuiltinVar::LocalInvocationIndex => vec![linear.clone()],
                BuiltinVar::WorkgroupId => ctaid.clone(),
                BuiltinVar::NumWorkgroups => {
                    let (from_ind, from_grid, done) = (fb.label(), fb.label(), fb.label());
                    fb.cond_br(&has_ind, &from_ind, &from_grid);
                    fb.start(&from_ind);
                    let loaded: Vec<Op> = (0..3).map(|k| fb.load(&ind, 4 * k as u32, ScalarTy::U32, true)).collect();
                    fb.br(&done);
                    fb.start(&from_grid);
                    fb.br(&done);
                    fb.start(&done);
                    (0..3).map(|k| fb.def(&format!("phi i32 [ {}, %{from_ind} ], [ {}, %{from_grid} ]", loaded[k], nctaid[k]))).collect()
                }
                BuiltinVar::SubgroupInvocationId => vec![ops::lane_id(&mut fb)],
                BuiltinVar::SubgroupSize => vec!["32".to_string()],
            };
            self.bind_local(&mut fb, &cx, *l, vals);
        }

        // 6. Body.
        self.jumps.clear();
        self.stmts(&mut fb, &m.entry_body, &cx);
        if !fb.terminated {
            fb.ret("ret void");
        }
        let body = self.absorb(fb);
        let def = format!("define void @luma_main({}) nounwind {{\n{body}}}\n", kparams.join(", "));
        (def, format!("void ({})*", ktypes.join(", ")), param_abis)
    }

    /// Bytes of the global behind shared variable `i`.
    fn shared_bytes(&self, i: u32) -> u32 {
        let d = &self.m.shared[i as usize];
        match d.len {
            Some(n) => n * self.shared_stride(i),
            None => self.sz(&d.elem_ty),
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

        let mut fns = String::new();
        for (i, f) in m.functions.iter().enumerate() {
            if live[i] {
                fns.push_str(&self.function(i, f));
            }
        }
        let (kernel, ktype, param_abis) = self.kernel(&lens);

        let [wx, wy, wz] = m.workgroup_size;
        let mut src = String::new();
        src.push_str(&format!("; {}\n", m.label.replace(['\n', '\r'], " ")));
        src.push_str(TRIPLE);
        src.push_str(DATALAYOUT);
        src.push_str("%LumaGrid = type { [3 x i32], i32 }\n");
        for i in 0..m.shared.len() {
            let n = self.shared_bytes(i as u32).max(1);
            let a = align_of(&m.shared[i].elem_ty, &m.structs);
            src.push_str(&format!("@sh{i} = internal addrspace(3) global [{n} x i8] undef, align {a}\n"));
        }
        for h in self.helpers.values() {
            src.push_str(h);
        }
        src.push_str(&fns);
        src.push_str(&kernel);
        for d in &self.decls {
            src.push_str(d);
            src.push('\n');
        }
        src.push_str("!nvvm.annotations = !{!20}\n");
        src.push_str(&format!("!20 = !{{{ktype} @luma_main, !\"kernel\", i32 1, !\"maxntidx\", i32 {wx}, !\"maxntidy\", i32 {wy}, !\"maxntidz\", i32 {wz}}}\n"));
        src.push_str("!nvvmir.version = !{!21}\n!21 = !{i32 2, i32 0}\n");
        src.push_str(TBAA);

        CudaTranslated { source: src, param_abis, array_lengths: lens }
    }
}

/// Translates `module` to NVVM IR. `used` is the reachability mask of bindings
/// (`None` keeps all); inactive bindings get no kernel parameter. Workgroup memory is zeroed
/// (see [`emit_with_workgroup_init`]).
pub fn emit(module: &KernelModule, used: Option<&[bool]>) -> CudaTranslated {
    emit_with_workgroup_init(module, used, true)
}

/// Like [`emit`]; with `initialize == false` no workgroup-memory zeroing loop or barrier is
/// emitted (for trusted generated kernels that write before they read). With `true` every
/// shared variable is cleared byte-wise over its whole size (so half, bool, atomic and
/// aggregate representations, tail padding included, start at zero) and a barrier follows.
/// The kernel ABI is identical either way.
pub fn emit_with_workgroup_init(module: &KernelModule, used: Option<&[bool]>, initialize: bool) -> CudaTranslated {
    let n = module.bindings.len();
    let active: Vec<bool> = (0..n).map(|i| used.is_none_or(|u| u.get(i).copied().unwrap_or(false))).collect();
    Em {
        m: module,
        active,
        abi: vec![None; n],
        res: Vec::new(),
        init_shared: initialize,
        decls: BTreeSet::new(),
        helpers: BTreeMap::new(),
        jumps: Vec::new(),
        converged: conv::convergent_subgroups(module),
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
    }

    fn local(i: u32) -> Box<Expr> {
        Box::new(Expr::Local(LocalId(i)))
    }

    fn lit(v: u32) -> Box<Expr> {
        Box::new(Expr::LitU32(v))
    }

    fn add(a: Box<Expr>, b: Box<Expr>) -> Expr {
        Expr::Binary { op: IrBinOp::Add, lhs: a, rhs: b, ty: KirType::U32 }
    }

    fn decls(n: usize) -> Vec<(String, KirType, bool)> {
        (0..n).map(|i| (format!("l{i}"), KirType::U32, false)).collect()
    }

    fn let_(l: u32, e: Expr) -> Stmt {
        Stmt::Let { local: LocalId(l), init: e }
    }

    #[test]
    fn locals_are_ssa_values_only_where_their_let_dominates_every_use() {
        // let a; let b = a + 1; if .. { let c = a + b; } use(b)
        let body = vec![
            let_(0, Expr::LitU32(1)),
            let_(1, add(local(0), lit(1))),
            Stmt::If { cond: Expr::LitBool(true), then_body: vec![let_(2, add(local(0), local(1)))], else_body: vec![] },
            Stmt::Discard(*local(1)),
        ];
        assert_eq!(ssa_locals(&decls(3), &body, &[]), vec![true, true, true]);
        // A `let` in a branch is not visible after it.
        let body = vec![
            Stmt::If { cond: Expr::LitBool(true), then_body: vec![let_(0, Expr::LitU32(1))], else_body: vec![] },
            Stmt::Discard(*local(0)),
        ];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
        // Neither is one in the other branch.
        let body = vec![Stmt::If { cond: Expr::LitBool(true), then_body: vec![let_(0, Expr::LitU32(1))], else_body: vec![Stmt::Discard(*local(0))] }];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
        // Used before it is declared.
        let body = vec![Stmt::Discard(*local(0)), let_(0, Expr::LitU32(1))];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
    }

    #[test]
    fn written_and_redeclared_locals_keep_their_slots() {
        let assign = Stmt::Assign { dst: LValue::Local(LocalId(0)), val: Expr::LitU32(2) };
        let body = vec![let_(0, Expr::LitU32(1)), assign, Stmt::Discard(*local(0))];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
        let body = vec![Stmt::Var { local: LocalId(0), init: Some(Expr::LitU32(1)) }, Stmt::Discard(*local(0))];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
        let body = vec![let_(0, Expr::LitU32(1)), let_(0, Expr::LitU32(2)), Stmt::Discard(*local(0))];
        assert_eq!(ssa_locals(&decls(1), &body, &[]), vec![false]);
        // The result of an atomic is stored into its local.
        let rmw = Stmt::AtomicRmw { dst: Some(LocalId(0)), op: IrAtomicOp::Add, elem_ty: ScalarTy::U32, target: AtomicTarget::Storage { binding: 0, index: None }, val: Expr::LitU32(1) };
        assert_eq!(ssa_locals(&decls(1), &[rmw, Stmt::Discard(*local(0))], &[]), vec![false]);
        // Aggregates and vectors are never SSA values.
        let mut d = decls(2);
        d[0].1 = KirType::vec(ScalarTy::F32, 4);
        d[1].1 = KirType::Array { elem: Box::new(KirType::U32), len: 4, stride: 4 };
        let body = vec![let_(0, Expr::ZeroVal(d[0].1.clone())), let_(1, Expr::ZeroVal(d[1].1.clone()))];
        assert_eq!(ssa_locals(&d, &body, &[]), vec![false, false]);
        // The builtins the prologue declares are values.
        assert_eq!(ssa_locals(&decls(1), &[Stmt::Discard(*local(0))], &[LocalId(0)]), vec![true]);
    }

    #[test]
    fn a_continue_hides_the_body_from_the_continuing_block() {
        let uses_in_continuing = |body: Vec<Stmt>| {
            let l = Stmt::Loop { body, continuing: vec![Stmt::Discard(*local(0))], break_if: None };
            ssa_locals(&decls(1), &[l], &[])[0]
        };
        assert!(uses_in_continuing(vec![let_(0, Expr::LitU32(1))]));
        let skip = Stmt::If { cond: Expr::LitBool(true), then_body: vec![Stmt::Continue], else_body: vec![] };
        assert!(!uses_in_continuing(vec![skip, let_(0, Expr::LitU32(1))]));
        // The loop's own `break_if` sees the same.
        let l = Stmt::Loop { body: vec![let_(0, Expr::LitU32(1))], continuing: vec![], break_if: Some(Expr::Local(LocalId(0))) };
        assert!(ssa_locals(&decls(1), &[l], &[])[0]);
    }

    /// A kernel with a loop, a `let`, a subgroup reduction every lane reaches and a store.
    fn reducing_module() -> KernelModule {
        let mut mb = ModuleBuilder::new("t", "main", [64, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::U32);
        mb.entry_builder().finish();
        let mut m = mb.finish();
        m.entry_locals = vec![
            ("lid".into(), KirType::U32, false),
            ("sg".into(), KirType::U32, false),
            ("x".into(), KirType::U32, false),
            ("n".into(), KirType::U32, true),
            ("s".into(), KirType::F32, false),
        ];
        m.entry_builtins = vec![(BuiltinVar::LocalInvocationIndex, LocalId(0)), (BuiltinVar::SubgroupSize, LocalId(1))];
        let total = Expr::Subgroup { op: IrSubgroupOp::Add, arg: Some(Box::new(Expr::Cast { kind: CastKind::Convert, expr: local(2), src_ty: KirType::U32, dst_ty: KirType::F32 })), lane: None, ty: KirType::F32 };
        m.entry_body = vec![
            let_(2, add(local(0), lit(1))),
            Stmt::Var { local: LocalId(3), init: Some(Expr::LitU32(0)) },
            Stmt::Loop {
                body: vec![Stmt::If { cond: Expr::Binary { op: IrBinOp::Ge, lhs: local(3), rhs: lit(4), ty: KirType::BOOL }, then_body: vec![Stmt::Break], else_body: vec![] }],
                continuing: vec![Stmt::Assign { dst: LValue::Local(LocalId(3)), val: add(local(3), lit(1)) }],
                break_if: None,
            },
            Stmt::Let { local: LocalId(4), init: total },
            Stmt::Store {
                binding: out,
                index: *local(0),
                val: add(Box::new(Expr::Cast { kind: CastKind::Convert, expr: local(4), src_ty: KirType::F32, dst_ty: KirType::U32 }), local(3)),
            },
        ];
        m
    }

    #[test]
    fn a_converged_reduction_takes_the_full_warp_path_and_no_slots_for_its_lets() {
        let t = emit(&reducing_module(), None);
        assert!(t.source.contains("@luma_sg_add_f32_c("), "{}", t.source);
        assert!(!t.source.contains("activemask"), "{}", t.source);
        // `x` and `s` are values; only the loop counter has a stack slot.
        assert_eq!(t.source.matches("alloca").count(), 1, "{}", t.source);
        // No typed access goes through a byte offset any more.
        assert!(!t.source.contains("getelementptr inbounds i8"), "{}", t.source);
    }

    #[test]
    fn emitted_ir_compiles_with_nvvm_when_the_driver_has_it() {
        let Ok(nvvm) = crate::nvvm::Nvvm::open(7, 5) else {
            eprintln!("no NVVM in this environment, skipped");
            return;
        };
        let ptx = nvvm.compile(&emit(&reducing_module(), None).source, 7, 5).unwrap();
        assert!(ptx.contains("shfl.sync.bfly.b32"), "{ptx}");
        assert!(!ptx.contains("activemask"), "{ptx}");
    }

}
