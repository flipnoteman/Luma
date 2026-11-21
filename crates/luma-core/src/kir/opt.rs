//! Target-specific rewrites of a [`KernelModule`] before it is printed: the helpers luma's WGSL
//! defines to work around WGSL's gaps become native operations where the target has them.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use super::ir::*;

fn helper_id(module: &KernelModule, name: &str, params: usize) -> Option<FnId> {
    module.functions.iter().position(|f| f.name == name && f.params.len() == params).map(|i| FnId(i as u32))
}

/// `luma_o(x)` for an `f32` sum, difference or product: the correctly rounded, never contracted
/// operation; any other `x` is itself.
fn fold_guard(x: Expr) -> Expr {
    match x {
        Expr::Binary { op: op @ (IrBinOp::Add | IrBinOp::Sub | IrBinOp::Mul), lhs, rhs, ty } if ty == KirType::F32 => {
            let op = match op {
                IrBinOp::Add => IrBinOp::AddExact,
                IrBinOp::Sub => IrBinOp::SubExact,
                _ => IrBinOp::MulExact,
            };
            Expr::Binary { op, lhs, rhs, ty }
        }
        other => other,
    }
}

/// Rewrites the calls and guards of one function body.
///
/// A guard whose argument is an immutable local (a value `bake` named because it is used more
/// than once) cannot see the arithmetic that defines it, so with `ieee` that definition (and
/// those of the locals it merely aliases) is made exact in place. Mutable `var`s, loads and
/// other expressions are left alone, and unguarded arithmetic stays as it was.
fn rewrite_body(body: &mut [Stmt], mulhi: Option<FnId>, guard: Option<FnId>, ieee: bool, guarded: &mut bool) {
    let mut marked: HashSet<LocalId> = HashSet::new();
    let mut rewrite = |e: &mut Expr| {
        let taken = std::mem::replace(e, Expr::LitBool(false));
        *e = match taken {
            Expr::Call { func, mut args, .. } if Some(func) == mulhi && args.len() == 2 => {
                let rhs = args.pop().expect("two arguments");
                let lhs = args.pop().expect("two arguments");
                Expr::Binary { op: IrBinOp::MulHiU32, lhs: lhs.boxed(), rhs: rhs.boxed(), ty: KirType::U32 }
            }
            Expr::Call { func, mut args, .. } if Some(func) == guard && args.len() == 1 => {
                *guarded = true;
                let x = args.pop().expect("one argument");
                if ieee {
                    if let Expr::Local(l) = &x {
                        marked.insert(*l);
                    }
                    fold_guard(x)
                } else {
                    Expr::Unary { op: IrUnOp::OpaqueGuard, arg: x.boxed(), ty: KirType::F32 }
                }
            }
            Expr::Unary { op: IrUnOp::OpaqueGuard, arg, .. } if ieee => {
                *guarded = true;
                if let Expr::Local(l) = &*arg {
                    marked.insert(*l);
                }
                fold_guard(*arg)
            }
            other => other,
        };
    };
    rewrite_exprs(body, &mut rewrite);
    if marked.is_empty() {
        return;
    }
    // `let a = b;` makes `b`'s definition the one that matters too.
    let mut alias: HashMap<LocalId, LocalId> = HashMap::new();
    walk_stmts(body, &mut |s| {
        if let Stmt::Let { local, init: Expr::Local(target) } = s {
            alias.insert(*local, *target);
        }
    });
    let mut frontier: Vec<LocalId> = marked.iter().copied().collect();
    while let Some(l) = frontier.pop() {
        if let Some(&target) = alias.get(&l)
            && marked.insert(target)
        {
            frontier.push(target);
        }
    }
    make_exact(body, &marked);
}

/// Turns the `f32` sum, difference or product that initialises each marked `let` exact.
fn make_exact(block: &mut [Stmt], marked: &HashSet<LocalId>) {
    for s in block {
        if let Stmt::Let { local, init } = s
            && marked.contains(&*local)
        {
            *init = fold_guard(std::mem::replace(init, Expr::LitBool(false)));
        }
        s.for_each_block_mut(&mut |b| make_exact(b, marked));
    }
}

/// Rewrites `module` for a target.
///
/// - `luma_mulhi(a, b)` becomes [`IrBinOp::MulHiU32`].
/// - `luma_o(x)` (and [`IrUnOp::OpaqueGuard`]) becomes, with `ieee_f32_intrinsics` (a target whose
///   `f32` operations can be asked to be exact), [`IrBinOp::AddExact`] / `SubExact` / `MulExact`
///   of `x`'s operands or plain `x`; without it, [`IrUnOp::OpaqueGuard`] of `x`.
/// - Helper functions nothing calls are dropped.
///
/// Dropping helpers can stop some bindings being referenced by the remaining code. The `used`
/// mask computed before this call stays the authoritative layout; backends must not re-derive it.
/// Returns the module itself when nothing applies.
pub fn optimize_for_target(module: &KernelModule, ieee_f32_intrinsics: bool) -> Cow<'_, KernelModule> {
    let mulhi = helper_id(module, "luma_mulhi", 2);
    let guard = helper_id(module, "luma_o", 1);

    let mut needed = module.reachable_functions().iter().any(|live| !live);
    let mut scan = |block: &[Stmt]| {
        walk_stmts(block, &mut |s| {
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| match x {
                    Expr::Call { func, .. } if Some(*func) == mulhi || Some(*func) == guard => needed = true,
                    Expr::Unary { op: IrUnOp::OpaqueGuard, .. } if ieee_f32_intrinsics => needed = true,
                    _ => {}
                })
            })
        });
    };
    scan(&module.entry_body);
    module.functions.iter().for_each(|f| scan(&f.body));
    if !needed {
        return Cow::Borrowed(module);
    }

    let mut out = module.clone();
    let mut guarded = false;
    rewrite_body(&mut out.entry_body, mulhi, guard, ieee_f32_intrinsics, &mut guarded);
    for f in &mut out.functions {
        rewrite_body(&mut f.body, mulhi, guard, ieee_f32_intrinsics, &mut guarded);
    }
    out.uses_exact_f32 |= guarded;

    // Drop the helpers nothing calls any more and renumber the rest.
    let live = out.reachable_functions();
    if live.iter().any(|l| !l) {
        let mut remap = vec![FnId(u32::MAX); live.len()];
        let mut next = 0;
        for (old, keep) in live.iter().enumerate() {
            if *keep {
                remap[old] = FnId(next);
                next += 1;
            }
        }
        let mut keep = live.iter();
        out.functions.retain(|_| *keep.next().expect("one flag per function"));
        let mut renumber = |e: &mut Expr| {
            if let Expr::Call { func, .. } = e {
                *func = remap[func.0 as usize];
            }
        };
        rewrite_exprs(&mut out.entry_body, &mut renumber);
        for f in &mut out.functions {
            rewrite_exprs(&mut f.body, &mut renumber);
        }
    }
    Cow::Owned(out)
}

// Runtime bounds checks.

/// Whether an access is in bounds always, never, or exactly when the condition holds.
enum Check {
    Ok,
    Never,
    Cond(Expr),
}

impl Check {
    fn and(self, other: Check) -> Check {
        match (self, other) {
            (Check::Never, _) | (_, Check::Never) => Check::Never,
            (Check::Ok, c) | (c, Check::Ok) => c,
            (Check::Cond(a), Check::Cond(b)) => Check::Cond(binary(IrBinOp::And, a, b)),
        }
    }
}

/// What an index is compared with.
#[derive(Clone, Copy)]
enum Bound {
    /// A length known from the type.
    Fixed(u32),
    /// The element count of the unsized array binding ([`Expr::ArrayLength`]).
    Runtime(u32),
}

/// What one access asks for. Both the preflight (does anything need a check?) and the rewrite
/// read this one description, so they cannot disagree about which accesses are safe.
enum Req<'e> {
    /// The access is never in bounds.
    Never,
    /// `index` must be below `bound`.
    Index(&'e Expr, Bound),
    /// The whole of a binding that is not an array must fit: `ArrayLength(binding) >= 1`.
    Whole(u32),
}

/// One step of a path, without owning its index.
#[derive(Clone, Copy)]
enum Step<'e> {
    Field(u32),
    Index(&'e Expr),
}

impl<'e> From<&'e AccessStep> for Step<'e> {
    fn from(step: &'e AccessStep) -> Self {
        match step {
            AccessStep::Field(k) => Step::Field(*k),
            AccessStep::Index(e) => Step::Index(e),
        }
    }
}

/// Where a walk along a path currently is, borrowing the declaration it started in.
#[derive(Clone, Copy)]
enum Cur<'t> {
    Ty(&'t KirType),
    /// A lane of a vector.
    Lane,
    /// A whole unsized or workgroup array: not indexable by value.
    Opaque,
}

impl<'t> Cur<'t> {
    fn len(self) -> u32 {
        match self {
            Cur::Ty(KirType::Array { len, .. }) => *len,
            Cur::Ty(KirType::Vec { lanes, .. }) => u32::from(*lanes),
            _ => panic!("kir: indexing into something that is not an array or vector"),
        }
    }

    fn elem(self) -> Cur<'t> {
        match self {
            Cur::Ty(KirType::Array { elem, .. }) => Cur::Ty(elem),
            Cur::Ty(KirType::Vec { .. }) => Cur::Lane,
            _ => panic!("kir: indexing into something that is not an array or vector"),
        }
    }
}

fn len_of(ty: &KirType) -> u32 {
    Cur::Ty(ty).len()
}

fn binary(op: IrBinOp, lhs: Expr, rhs: Expr) -> Expr {
    let ty = if op.is_comparison() || matches!(op, IrBinOp::And | IrBinOp::Or) { KirType::BOOL } else { KirType::U32 };
    Expr::Binary { op, lhs: lhs.boxed(), rhs: rhs.boxed(), ty }
}

/// The declarations and local types of the function being looked at; reads only.
#[derive(Clone, Copy)]
struct Types<'a> {
    m: &'a KernelModule,
    locals: &'a [(String, KirType, bool)],
    params: &'a [(String, KirType)],
}

impl<'a> Types<'a> {
    /// The type of `e`, borrowed from the expression or a table where it is stored.
    fn ty_of<'x>(&'x self, e: &'x Expr) -> Cow<'x, KirType> {
        match e {
            Expr::LitBool(_) => Cow::Owned(KirType::BOOL),
            Expr::LitU32(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => Cow::Owned(KirType::U32),
            Expr::LitI32(_) => Cow::Owned(KirType::I32),
            Expr::LitF16(_) => Cow::Owned(KirType::Scalar(ScalarTy::F16)),
            Expr::LitF32(_) => Cow::Owned(KirType::F32),
            Expr::LitF64(_) => Cow::Owned(KirType::Scalar(ScalarTy::F64)),
            Expr::ZeroVal(t) => Cow::Borrowed(t),
            Expr::Local(l) => Cow::Borrowed(&self.locals[l.0 as usize].1),
            Expr::FnParam(i) => Cow::Borrowed(&self.params[*i as usize].1),
            Expr::Private(p) => Cow::Borrowed(&self.m.privates[*p as usize].ty),
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
            | Expr::Subgroup { ty, .. } => Cow::Borrowed(ty),
            Expr::Splat { elem, lanes, .. } => Cow::Owned(KirType::vec(*elem, *lanes)),
            Expr::Cast { dst_ty, .. } => Cow::Borrowed(dst_ty),
            Expr::Call { ty, .. } => Cow::Borrowed(ty.as_ref().expect("a call used as a value returns one")),
            Expr::AtomicLoad { ty, .. } => Cow::Owned(KirType::Scalar(*ty)),
        }
    }

    fn step<'t>(&self, cur: Cur<'t>, step: Step<'_>) -> Cur<'t>
    where
        'a: 't,
    {
        match (cur, step) {
            (Cur::Ty(KirType::Struct(s)), Step::Field(k)) => Cur::Ty(&self.m.structs[*s as usize].fields[k as usize].ty),
            _ => cur.elem(),
        }
    }

    /// Says what `e` as an index into `bound` asks for. Literals are decided here.
    fn index_req(&self, e: &Expr, bound: Bound, emit: &mut dyn FnMut(Req<'_>)) {
        let literal = match e {
            Expr::LitU32(k) => Some(i64::from(*k)),
            Expr::LitI32(k) => Some(i64::from(*k)),
            _ => None,
        };
        match (literal, bound) {
            (Some(k), Bound::Fixed(n)) => {
                if !(0..i64::from(n)).contains(&k) {
                    emit(Req::Never);
                }
            }
            // The runtime bound may be zero: even index 0 needs its check.
            (Some(k), Bound::Runtime(_)) if k < 0 => emit(Req::Never),
            _ => emit(Req::Index(e, bound)),
        }
    }

    fn walk<'t, 'e>(&self, mut cur: Cur<'t>, steps: impl IntoIterator<Item = Step<'e>>, emit: &mut dyn FnMut(Req<'_>)) -> Cur<'t>
    where
        'a: 't,
    {
        for step in steps {
            if let Step::Index(e) = step {
                self.index_req(e, Bound::Fixed(cur.len()), emit);
            }
            cur = self.step(cur, step);
        }
        cur
    }

    fn storage_place<'t, 'e>(&self, binding: u32, steps: impl IntoIterator<Item = Step<'e>>, emit: &mut dyn FnMut(Req<'_>)) -> Cur<'t>
    where
        'a: 't,
    {
        let decl = &self.m.bindings[binding as usize];
        let mut steps = steps.into_iter();
        if decl.is_unsized_array {
            return match steps.next() {
                Some(Step::Index(e)) => {
                    self.index_req(e, Bound::Runtime(binding), emit);
                    self.walk(Cur::Ty(&decl.elem_ty), steps, emit)
                }
                None => Cur::Opaque,
                Some(Step::Field(_)) => panic!("kir: field of an unsized array"),
            };
        }
        // The whole object fits the bound bytes exactly when it makes at least one "element".
        emit(Req::Whole(binding));
        self.walk(Cur::Ty(&decl.elem_ty), steps, emit)
    }

    fn shared_place<'t, 'e>(&self, shared: u32, steps: impl IntoIterator<Item = Step<'e>>, emit: &mut dyn FnMut(Req<'_>)) -> Cur<'t>
    where
        'a: 't,
    {
        let decl = &self.m.shared[shared as usize];
        let mut steps = steps.into_iter();
        match decl.len {
            Some(n) => match steps.next() {
                Some(Step::Index(e)) => {
                    self.index_req(e, Bound::Fixed(n), emit);
                    self.walk(Cur::Ty(&decl.elem_ty), steps, emit)
                }
                None => Cur::Opaque,
                Some(Step::Field(_)) => panic!("kir: field of a workgroup array"),
            },
            None => self.walk(Cur::Ty(&decl.elem_ty), steps, emit),
        }
    }

    fn lvalue<'t>(&self, lv: &LValue, emit: &mut dyn FnMut(Req<'_>)) -> Cur<'t>
    where
        'a: 't,
    {
        match lv {
            LValue::Local(l) => Cur::Ty(&self.locals[l.0 as usize].1),
            LValue::Private(p) => Cur::Ty(&self.m.privates[*p as usize].ty),
            LValue::Storage { binding, path } => self.storage_place(*binding, path.iter().map(Step::from), emit),
            LValue::Shared { shared, path } => self.shared_place(*shared, path.iter().map(Step::from), emit),
            LValue::Index { base, index } => {
                let base = self.lvalue(base, emit);
                self.index_req(index, Bound::Fixed(base.len()), emit);
                base.elem()
            }
            LValue::Field { base, field } => {
                let base = self.lvalue(base, emit);
                self.step(base, Step::Field(*field))
            }
        }
    }

    fn atomic_req(&self, target: &AtomicTarget, emit: &mut dyn FnMut(Req<'_>)) {
        match target {
            AtomicTarget::Storage { binding, index } => {
                self.storage_place(*binding, index.as_deref().map(Step::Index), emit);
            }
            AtomicTarget::Shared { shared, index } => {
                self.shared_place(*shared, index.as_deref().map(Step::Index), emit);
            }
        }
    }

    /// The requirements of the memory read `e` itself (not of its operands); none for any other
    /// expression.
    fn read_reqs(&self, e: &Expr, emit: &mut dyn FnMut(Req<'_>)) {
        match e {
            Expr::Load { binding, index, .. } => {
                self.storage_place(*binding, [Step::Index(index)], emit);
            }
            Expr::StorageRead { binding, path, .. } => {
                self.storage_place(*binding, path.iter().map(Step::from), emit);
            }
            Expr::SharedLoad { shared, index, .. } => {
                self.shared_place(*shared, [Step::Index(index)], emit);
            }
            Expr::SharedRead { shared, path, .. } => {
                self.shared_place(*shared, path.iter().map(Step::from), emit);
            }
            Expr::UniformRead { binding, path, .. } => {
                self.walk(Cur::Ty(&self.m.bindings[*binding as usize].elem_ty), path.iter().map(Step::from), emit);
            }
            Expr::Access { base, step: AccessStep::Index(i), .. } => {
                let n = len_of(&self.ty_of(base));
                self.index_req(i, Bound::Fixed(n), emit);
            }
            Expr::AtomicLoad { target, .. } => self.atomic_req(target, emit),
            _ => {}
        }
    }

    /// The requirements of the memory access `s` makes.
    fn stmt_reqs(&self, s: &Stmt, emit: &mut dyn FnMut(Req<'_>)) {
        match s {
            Stmt::Store { binding, index, .. } => {
                self.storage_place(*binding, [Step::Index(index)], emit);
            }
            Stmt::SharedStore { shared, index, .. } => {
                self.shared_place(*shared, [Step::Index(index)], emit);
            }
            Stmt::Assign { dst, .. } => {
                self.lvalue(dst, emit);
            }
            Stmt::AtomicStore { target, .. } | Stmt::AtomicRmw { target, .. } | Stmt::AtomicCompareExchange { target, .. } => self.atomic_req(target, emit),
            _ => {}
        }
    }

    /// `e` as a `u32` whose out-of-range values include every negative `i32`.
    fn as_u32(&self, e: &Expr) -> Expr {
        match e {
            Expr::LitU32(_) => e.clone(),
            Expr::LitI32(k) if *k >= 0 => Expr::LitU32(*k as u32),
            _ => match &*self.ty_of(e) {
                KirType::Scalar(ScalarTy::U32) => e.clone(),
                KirType::Scalar(ScalarTy::I32) => Expr::Cast { kind: CastKind::Bitcast, expr: e.clone().boxed(), src_ty: KirType::I32, dst_ty: KirType::U32 },
                other => panic!("kir: index of type {other:?}"),
            },
        }
    }
}

/// Whether anything in `module` needs a check. Reads the module only: no clone, no allocation.
fn needs_checks(module: &KernelModule) -> bool {
    let scan = |types: Types<'_>, body: &[Stmt]| {
        let mut need = false;
        let mut emit = |_: Req<'_>| need = true;
        walk_stmts(body, &mut |s| {
            types.stmt_reqs(s, &mut emit);
            s.for_each_expr(&mut |e| e.walk(&mut |x| types.read_reqs(x, &mut emit)));
        });
        need
    };
    scan(Types { m: module, locals: &module.entry_locals, params: &[] }, &module.entry_body)
        || module.functions.iter().any(|f| scan(Types { m: module, locals: &f.locals, params: &f.params }, &f.body))
}

/// Builds the condition of a [`Req`] stream.
struct Collect<'t> {
    types: Types<'t>,
    ranges: &'t Ranges,
    check: Check,
    whole: Option<u32>,
}

impl Collect<'_> {
    fn emit(&mut self, req: Req<'_>) {
        let next = match req {
            Req::Never => Check::Never,
            // An index proven below a fixed bound needs no check.
            Req::Index(e, Bound::Fixed(n)) if self.ranges.span(e).is_some_and(|(_, hi)| hi < n) => return,
            Req::Index(e, bound) => {
                let limit = match bound {
                    Bound::Fixed(n) => Expr::LitU32(n),
                    Bound::Runtime(binding) => Expr::ArrayLength { binding },
                };
                Check::Cond(binary(IrBinOp::Lt, self.types.as_u32(e), limit))
            }
            Req::Whole(binding) => {
                self.whole = Some(binding);
                Check::Cond(binary(IrBinOp::Ge, Expr::ArrayLength { binding }, Expr::LitU32(1)))
            }
        };
        self.check = std::mem::replace(&mut self.check, Check::Ok).and(next);
    }
}

/// Calls `f` on each index of a place.
fn lvalue_indices_mut(lv: &mut LValue, f: &mut dyn FnMut(&mut Expr)) {
    match lv {
        LValue::Local(_) | LValue::Private(_) => {}
        LValue::Storage { path, .. } | LValue::Shared { path, .. } => {
            for s in path {
                if let AccessStep::Index(e) = s {
                    f(e);
                }
            }
        }
        LValue::Index { base, index } => {
            lvalue_indices_mut(base, f);
            f(index);
        }
        LValue::Field { base, .. } => lvalue_indices_mut(base, f),
    }
}

/// Calls `f` on each operand of the memory read `e` that the read evaluates itself.
fn read_operands_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Load { index, .. } | Expr::SharedLoad { index, .. } => f(index),
        Expr::StorageRead { path, .. } | Expr::SharedRead { path, .. } | Expr::UniformRead { path, .. } => {
            for s in path {
                if let AccessStep::Index(i) = s {
                    f(i);
                }
            }
        }
        Expr::Access { base, step, .. } => {
            f(base);
            if let AccessStep::Index(i) = step {
                f(i);
            }
        }
        Expr::AtomicLoad { target, .. } => target.for_each_expr_mut(f),
        _ => {}
    }
}

/// Whether evaluating `e` twice could differ from evaluating it once.
fn has_effect(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| found |= matches!(x, Expr::Call { .. } | Expr::AtomicLoad { .. }));
    found
}

fn mem_read_ty(e: &Expr) -> KirType {
    match e {
        Expr::Load { ty, .. }
        | Expr::StorageRead { ty, .. }
        | Expr::SharedLoad { ty, .. }
        | Expr::SharedRead { ty, .. }
        | Expr::UniformRead { ty, .. }
        | Expr::Access { ty, .. } => ty.clone(),
        Expr::AtomicLoad { ty, .. } => KirType::Scalar(*ty),
        other => unreachable!("not a memory read: {other:?}"),
    }
}

/// The locals of the function being rewritten.
struct Scope<'a> {
    /// The module as it was before the pass: declarations are read from it.
    m: &'a KernelModule,
    locals: &'a mut Vec<(String, KirType, bool)>,
    params: &'a [(String, KirType)],
    /// Non-array storage bindings whose whole object is guarded through `ArrayLength`, so that
    /// the output module gives them `element_stride = size`.
    whole: &'a mut [bool],
    /// Statements that evaluate an effectful index once, to run before the statement being
    /// rewritten.
    pre: Vec<Stmt>,
    /// What is known of the values of `u32` locals (see [`Ranges`]).
    ranges: Ranges,
    /// How many times each local is assigned in the function (declarations do not count).
    assigns: Vec<u32>,
    /// The counter of the loop whose body is about to be rewritten; taken by its first block.
    loop_counter: Option<Counter>,
}

/// Inclusive bounds of the value of an expression of type `u32`.
type Span = (u32, u32);

/// A loop variable that starts at `first`, is assigned only `step` more at the end of every trip
/// (its `continuing` block) and is left by a guard `if !(var < limit) { break }` at the top of
/// the body: past the guard it stays within `first ..= first + step * k`.
#[derive(Clone, Copy)]
struct Counter {
    var: u32,
    first: u32,
    step: u32,
}

/// Value ranges of the `u32` locals of one function, to leave out the checks that cannot fail:
/// the workgroup-memory reads of a tiled loop (`tile[(k + 1u) * 32u + lid.y]`) are in bounds by
/// construction, and a branch or clamp per access there costs a few percent of a kernel.
///
/// Only what is certain is recorded: immutable `let`s whose value has no wrapping arithmetic,
/// `local_invocation_id` / `local_invocation_index` of the entry point (below the workgroup
/// size) and loop counters of the shape [`Counter`] describes. A range exists only where the
/// rewrite has seen the statement that proves it.
#[derive(Default)]
struct Ranges {
    spans: HashMap<u32, Span>,
    /// `let a = counter` copies that the guard of a loop may test instead of the counter.
    aliases: HashMap<u32, u32>,
    /// Loop counters being rewritten, with their first value and step.
    counters: HashMap<u32, (u32, u32)>,
    /// Variables declared with a literal initial value (or none: zero).
    inits: HashMap<u32, u32>,
    /// The `v = literal` assignment that was the previous statement of the block being rewritten.
    last_assign: Option<(u32, u32)>,
    /// The local holding `local_invocation_id` and the workgroup size.
    local_id: Option<(u32, [u32; 3])>,
}

impl Ranges {
    /// The builtin locals of the entry point of `m`; `assigns` and `locals` are its own.
    fn for_entry(m: &KernelModule, assigns: &[u32]) -> Ranges {
        let mut ranges = Ranges::default();
        let size = m.workgroup_size;
        let fixed = |l: LocalId| !m.entry_locals[l.0 as usize].2 && assigns.get(l.0 as usize).copied().unwrap_or(0) == 0;
        for &(var, local) in &m.entry_builtins {
            if !fixed(local) {
                continue;
            }
            match var {
                BuiltinVar::LocalInvocationId => ranges.local_id = Some((local.0, size)),
                BuiltinVar::LocalInvocationIndex => {
                    let total = u64::from(size[0]) * u64::from(size[1]) * u64::from(size[2]);
                    if let Ok(total) = u32::try_from(total) {
                        ranges.spans.insert(local.0, (0, total.saturating_sub(1)));
                    }
                }
                _ => {}
            }
        }
        ranges
    }

    /// Bounds of `e` that hold whenever it is evaluated, if it is a `u32` made of constants,
    /// known locals and arithmetic that cannot wrap.
    fn span(&self, e: &Expr) -> Option<Span> {
        match e {
            Expr::LitU32(k) => Some((*k, *k)),
            Expr::Local(l) => self.spans.get(&l.0).copied(),
            Expr::Access { base, step: AccessStep::Field(i), ty, .. } if *ty == KirType::U32 => {
                let (local, size) = self.local_id?;
                match &**base {
                    Expr::Local(l) if l.0 == local && *i < 3 => Some((0, size[*i as usize].checked_sub(1)?)),
                    _ => None,
                }
            }
            Expr::Binary { op, lhs, rhs, ty } if *ty == KirType::U32 => {
                let (a, b) = (self.span(lhs)?, self.span(rhs)?);
                let (alo, ahi, blo, bhi) = (u64::from(a.0), u64::from(a.1), u64::from(b.0), u64::from(b.1));
                let (lo, hi) = match op {
                    IrBinOp::Add => (alo + blo, ahi + bhi),
                    IrBinOp::Mul => (alo * blo, ahi * bhi),
                    IrBinOp::Sub if alo >= bhi => (alo - bhi, ahi - blo),
                    IrBinOp::Div if blo > 0 => (alo / bhi, ahi / blo),
                    IrBinOp::Rem if blo > 0 => (0, ahi.min(bhi - 1)),
                    IrBinOp::Min => (alo.min(blo), ahi.min(bhi)),
                    IrBinOp::Max => (alo.max(blo), ahi.max(bhi)),
                    IrBinOp::BitAnd => (0, ahi.min(bhi)),
                    IrBinOp::Shr if blo == bhi && blo < 32 => (alo >> blo, ahi >> blo),
                    IrBinOp::Shl if blo == bhi && blo < 32 => (alo << blo, ahi << blo),
                    _ => return None,
                };
                Some((u32::try_from(lo).ok()?, u32::try_from(hi).ok()?))
            }
            _ => None,
        }
    }

    /// The counter `continuing` advances, if it is one (see [`Counter`]): a variable declared
    /// with a literal that nothing but this block assigns, by a literal step.
    fn counter_of(&self, continuing: &[Stmt], locals: &[(String, KirType, bool)], assigns: &[u32], before: Option<(u32, u32)>) -> Option<Counter> {
        let mut copies: Vec<(u32, u32)> = Vec::new();
        for s in continuing {
            match s {
                Stmt::Let { local, init: Expr::Local(v) } => copies.push((local.0, v.0)),
                Stmt::Assign { dst: LValue::Local(v), val: Expr::Binary { op: IrBinOp::Add, lhs, rhs, .. } } => {
                    let Expr::LitU32(step) = **rhs else { continue };
                    let Expr::Local(from) = **lhs else { continue };
                    let reads_v = from == *v || copies.contains(&(from.0, v.0));
                    // Either the declaration starts it and this block is the only assignment, or
                    // the statement right before the loop assigns it a literal and this block is
                    // the only other assignment.
                    let (first, unique) = match (assigns.get(v.0 as usize).copied(), before) {
                        (Some(1), _) => (self.inits.get(&v.0).copied(), true),
                        (Some(2), Some((var, first))) if var == v.0 => (Some(first), true),
                        _ => (None, false),
                    };
                    if reads_v && step > 0 && unique && locals[v.0 as usize].1 == KirType::U32 && let Some(first) = first {
                        return Some(Counter { var: v.0, first, step });
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Notes what the statement `s` (as written) shows about locals, before it is rewritten.
    /// `guard` is the counter of the loop whose body `s` is a direct statement of, if any.
    fn note(&mut self, s: &Stmt, guard: Option<Counter>) {
        match s {
            Stmt::Var { local, init: Some(Expr::LitU32(k)) } => {
                self.inits.insert(local.0, *k);
            }
            Stmt::Var { local, init: None } => {
                self.inits.insert(local.0, 0);
            }
            Stmt::Let { local, init } => {
                if let Expr::Local(v) = init
                    && self.counters.contains_key(&v.0)
                {
                    self.aliases.insert(local.0, v.0);
                }
                if let Some(span) = self.span(init) {
                    self.spans.insert(local.0, span);
                }
            }
            Stmt::If { cond: Expr::Binary { op, lhs, rhs, .. }, then_body, else_body } if guard.is_some() => {
                let counter = guard.expect("checked");
                let leaves = match op {
                    IrBinOp::Lt => then_body.is_empty() && matches!(else_body[..], [Stmt::Break]),
                    IrBinOp::Ge => else_body.is_empty() && matches!(then_body[..], [Stmt::Break]),
                    _ => false,
                };
                let tests_counter = matches!(**lhs, Expr::Local(a) if a.0 == counter.var || self.aliases.get(&a.0) == Some(&counter.var));
                if let (true, true, Expr::LitU32(limit)) = (leaves, tests_counter, &**rhs)
                    && u64::from(*limit) + u64::from(counter.step) <= 1 << 32
                {
                    // The values the variable takes while it is below `limit`.
                    let last = match limit.checked_sub(1) {
                        Some(top) if top >= counter.first => counter.first + counter.step * ((top - counter.first) / counter.step),
                        _ => counter.first,
                    };
                    self.spans.insert(counter.var, (counter.first, last));
                }
            }
            _ => {}
        }
    }
}

/// How many times each local of a function is assigned.
fn count_assigns(body: &[Stmt], locals: usize) -> Vec<u32> {
    fn root(lv: &LValue) -> Option<u32> {
        match lv {
            LValue::Local(l) => Some(l.0),
            LValue::Index { base, .. } | LValue::Field { base, .. } => root(base),
            _ => None,
        }
    }
    let mut counts = vec![0u32; locals];
    let mut bump = |l: u32| {
        if let Some(c) = counts.get_mut(l as usize) {
            *c += 1;
        }
    };
    walk_stmts(body, &mut |s| match s {
        Stmt::Assign { dst, .. } => {
            if let Some(l) = root(dst) {
                bump(l);
            }
        }
        Stmt::AtomicRmw { dst: Some(l), .. } => bump(l.0),
        Stmt::AtomicCompareExchange { dst_old, dst_exchanged, .. } => {
            bump(dst_old.0);
            bump(dst_exchanged.0);
        }
        _ => {}
    });
    counts
}

impl Scope<'_> {
    fn types(&self) -> Types<'_> {
        Types { m: self.m, locals: &self.locals[..], params: self.params }
    }

    fn collect(&mut self, f: impl FnOnce(&Types<'_>, &mut dyn FnMut(Req<'_>))) -> Check {
        let types = Types { m: self.m, locals: &self.locals[..], params: self.params };
        let mut collect = Collect { types, ranges: &self.ranges, check: Check::Ok, whole: None };
        f(&types, &mut |req: Req<'_>| collect.emit(req));
        let (check, whole) = (collect.check, collect.whole);
        if let Some(binding) = whole {
            self.whole[binding as usize] = true;
        }
        check
    }

    fn temp(&mut self, name: &str, ty: KirType) -> LocalId {
        self.locals.push((name.to_owned(), ty, false));
        LocalId(self.locals.len() as u32 - 1)
    }

    /// Makes `e`, which a check is about to repeat, evaluate once: an effectful `e` becomes a
    /// `let` before the statement (or before the select arm it is in) and `e` its local. Whether
    /// anything changed.
    fn stabilize(&mut self, e: &mut Expr) -> bool {
        if !has_effect(e) {
            return false;
        }
        let ty = self.types().ty_of(e).into_owned();
        let local = self.temp("bc_index", ty);
        let init = std::mem::replace(e, Expr::Local(local));
        self.pre.push(Stmt::Let { local, init });
        true
    }

    /// Post-order rewrite of one expression.
    ///
    /// The arms of a select are evaluated only when selected, so statements an arm needs before
    /// it (see [`stabilize`](Self::stabilize)) must not run unconditionally. A scalar select
    /// whose arms need some is lowered to a branch; the select becomes the local the branch
    /// assigns.
    fn expr(&mut self, e: &mut Expr) {
        let mut branches = None;
        match &mut *e {
            Expr::Ternary { op: IrTernaryOp::Select, a, b, c, .. } => {
                self.expr(a);
                let outer = std::mem::take(&mut self.pre);
                self.expr(b);
                let pre_b = std::mem::take(&mut self.pre);
                self.expr(c);
                let pre_c = std::mem::replace(&mut self.pre, outer);
                if !pre_b.is_empty() || !pre_c.is_empty() {
                    if matches!(&*self.types().ty_of(a), KirType::Scalar(ScalarTy::Bool)) {
                        branches = Some((pre_b, pre_c));
                    } else {
                        // A vector select computes both arms: they only need their lets first.
                        self.pre.extend(pre_b);
                        self.pre.extend(pre_c);
                    }
                }
            }
            other => other.for_each_child_mut(&mut |child| self.expr(child)),
        }
        if let Some((mut then_body, mut else_body)) = branches {
            let Expr::Ternary { a, b, c, ty, .. } = std::mem::replace(e, Expr::LitBool(false)) else { unreachable!("a select was matched above") };
            let result = self.temp("bc_select", ty);
            self.locals[result.0 as usize].2 = true;
            then_body.push(Stmt::Assign { dst: LValue::Local(result), val: *b });
            else_body.push(Stmt::Assign { dst: LValue::Local(result), val: *c });
            self.pre.push(Stmt::Var { local: result, init: None });
            self.pre.push(Stmt::If { cond: *a, then_body, else_body });
            *e = Expr::Local(result);
        }
        self.guard_read(e);
    }

    /// A read that may leave its memory becomes `cond ? read : zero`, which native targets print
    /// as a short-circuit conditional.
    fn guard_read(&mut self, e: &mut Expr) {
        let mut check = self.collect(|t, emit| t.read_reqs(&*e, emit));
        if matches!(check, Check::Ok) {
            return;
        }
        let mut moved = false;
        read_operands_mut(e, &mut |operand| moved |= self.stabilize(operand));
        if moved {
            check = self.collect(|t, emit| t.read_reqs(&*e, emit));
        }
        let ty = mem_read_ty(e);
        match check {
            Check::Ok => {}
            Check::Never => *e = Expr::ZeroVal(ty),
            Check::Cond(cond) => {
                let read = std::mem::replace(e, Expr::LitBool(false));
                *e = Expr::Ternary { op: IrTernaryOp::Select, a: cond.boxed(), b: read.boxed(), c: Expr::ZeroVal(ty.clone()).boxed(), ty };
            }
        }
    }

    fn stabilize_places(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Store { index, .. } | Stmt::SharedStore { index, .. } => {
                self.stabilize(index);
            }
            Stmt::Assign { dst, .. } => lvalue_indices_mut(dst, &mut |e| {
                self.stabilize(e);
            }),
            Stmt::AtomicStore { target, .. } | Stmt::AtomicRmw { target, .. } | Stmt::AtomicCompareExchange { target, .. } => {
                target.for_each_expr_mut(&mut |e| {
                    self.stabilize(e);
                })
            }
            _ => {}
        }
    }

    /// A guarded statement skips its values' effects with it, so effectful values are evaluated
    /// before the branch.
    fn stabilize_values(&mut self, s: &mut Stmt) {
        match s {
            Stmt::Store { val, .. }
            | Stmt::SharedStore { val, .. }
            | Stmt::Assign { val, .. }
            | Stmt::AtomicStore { val, .. }
            | Stmt::AtomicRmw { val, .. } => {
                self.stabilize(val);
            }
            Stmt::AtomicCompareExchange { cmp, val, .. } => {
                self.stabilize(cmp);
                self.stabilize(val);
            }
            _ => {}
        }
    }

    /// Pushes `s` into `out`, guarded when its memory access may be out of bounds. A result local
    /// of an atomic stays declared outside the branch (zero when the access is skipped); the
    /// branch computes into fresh temporaries and copies them over.
    fn guard_stmt(&mut self, s: Stmt, check: Check, out: &mut Vec<Stmt>) {
        let cond = match check {
            Check::Ok => return out.push(s),
            Check::Never => Expr::LitBool(false),
            Check::Cond(c) => c,
        };
        match s {
            Stmt::AtomicRmw { dst: Some(old), op, elem_ty, target, val } => {
                let ty = KirType::Scalar(elem_ty);
                self.locals[old.0 as usize].2 = true;
                let tmp = self.temp("bc_old", ty.clone());
                out.push(Stmt::Var { local: old, init: Some(Expr::ZeroVal(ty)) });
                out.push(Stmt::If {
                    cond,
                    then_body: vec![
                        Stmt::AtomicRmw { dst: Some(tmp), op, elem_ty, target, val },
                        Stmt::Assign { dst: LValue::Local(old), val: Expr::Local(tmp) },
                    ],
                    else_body: Vec::new(),
                });
            }
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, elem_ty, target, cmp, val } => {
                let ty = KirType::Scalar(elem_ty);
                self.locals[dst_old.0 as usize].2 = true;
                self.locals[dst_exchanged.0 as usize].2 = true;
                let old = self.temp("bc_old", ty.clone());
                let exchanged = self.temp("bc_exchanged", KirType::BOOL);
                out.push(Stmt::Var { local: dst_old, init: Some(Expr::ZeroVal(ty)) });
                out.push(Stmt::Var { local: dst_exchanged, init: Some(Expr::LitBool(false)) });
                out.push(Stmt::If {
                    cond,
                    then_body: vec![
                        Stmt::AtomicCompareExchange { dst_old: old, dst_exchanged: exchanged, elem_ty, target, cmp, val },
                        Stmt::Assign { dst: LValue::Local(dst_old), val: Expr::Local(old) },
                        Stmt::Assign { dst: LValue::Local(dst_exchanged), val: Expr::Local(exchanged) },
                    ],
                    else_body: Vec::new(),
                });
            }
            s => out.push(Stmt::If { cond, then_body: vec![s], else_body: Vec::new() }),
        }
    }

    fn block(&mut self, block: &mut Vec<Stmt>) {
        let old = std::mem::take(block);
        block.reserve(old.len());
        // Only the statements directly in a loop body are past its guard once it has run.
        let guard = self.loop_counter.take();
        for mut s in old {
            let before = self.ranges.last_assign.take();
            self.ranges.note(&s, guard);
            if let Stmt::Assign { dst: LValue::Local(v), val: Expr::LitU32(c) } = &s {
                self.ranges.last_assign = Some((v.0, *c));
            }
            if let Stmt::Loop { body, continuing, break_if } = &mut s {
                let counter = self.ranges.counter_of(continuing, &self.locals[..], &self.assigns, before);
                if let Some(c) = counter {
                    self.ranges.counters.insert(c.var, (c.first, c.step));
                }
                self.loop_counter = counter;
                self.block(body);
                self.loop_counter = None;
                // The variable moves on before `continuing` reads it, and is unknown after the loop.
                if let Some(c) = counter {
                    self.ranges.spans.remove(&c.var);
                    self.ranges.counters.remove(&c.var);
                }
                self.block(continuing);
                if let Some(e) = break_if {
                    // Evaluated after `continuing` on every iteration, so its `let`s go there.
                    self.expr(e);
                    continuing.append(&mut self.pre);
                }
                block.push(s);
                continue;
            }
            s.for_each_expr_mut(&mut |e| self.expr(e));
            self.stabilize_places(&mut s);
            let check = self.collect(|t, emit| t.stmt_reqs(&s, emit));
            if !matches!(check, Check::Ok) {
                self.stabilize_values(&mut s);
            }
            let mut pre = std::mem::take(&mut self.pre);
            s.for_each_block_mut(&mut |b| self.block(b));
            block.append(&mut pre);
            self.guard_stmt(s, check, block);
        }
    }
}

/// Makes every memory access of `module` safe at run time, for kernels that were not generated
/// by luma (backends without the robustness of WebGPU, on untrusted WGSL).
///
/// An access that may leave its memory is skipped: a read yields zero (`cond ? read : zero`,
/// which native targets print as a short-circuit conditional, so the unsafe access is never
/// evaluated), a write or atomic does nothing, and the result locals of an atomic are zero
/// (`false` for `exchanged`). Dynamic indices are checked against the length their type declares
/// (arrays, vectors; shared, private, local, uniform and storage places alike), constant
/// in-range indices need no check, constant invalid ones skip the access outright. Struct fields
/// are never checked. The statements keep their order and no barrier moves.
///
/// Whether anything needs a check is decided first, from the module alone: when nothing does,
/// the module is returned without a copy. An access to an array whose bound is only known at
/// run time needs its check even at a literal index (the bound may be zero).
///
/// Metadata assumptions for the backend running the result:
/// - A storage array whose bound is unknown (`is_unsized_array`) is guarded by
///   [`Expr::ArrayLength`], so the backend must answer it from the bytes it binds and the
///   binding's `element_stride`.
/// - Any other storage binding is guarded by `ArrayLength(binding) >= 1`. In the returned module
///   only, such a binding has `element_stride = Some(size of its whole type)`, so the same
///   `bytes / element_stride` answers "does the whole object fit". Bindings the pass does not
///   guard keep their `element_stride`. The caller's [`BindingInfo`](crate::hal::BindingInfo)s
///   stay those of the module it passed in.
///
/// An index that has effects (a call, an atomic load) is evaluated once: it moves into a `let`
/// before its statement and the check and the access both read that local. The same goes for
/// the operands and values of a guarded access, which the skipped branch would otherwise drop.
/// An effectful index inside an arm of a select stays inside it: a scalar select with such an
/// arm becomes `var r; if cond { arm's lets; r = a } else { arm's lets; r = b }`, so the
/// unselected arm's effects never happen. A vector select evaluates both arms anyway, so their
/// lets simply come first, in order.
///
/// No binding, parameter or function is added or renumbered, so `used` masks stay valid.
pub fn with_bounds_checks(module: &KernelModule) -> Cow<'_, KernelModule> {
    if !needs_checks(module) {
        return Cow::Borrowed(module);
    }
    let mut out = module.clone();
    let mut whole = vec![false; module.bindings.len()];
    let mut body = std::mem::take(&mut out.entry_body);
    let assigns = count_assigns(&body, module.entry_locals.len());
    let ranges = Ranges::for_entry(module, &assigns);
    let mut scope = Scope { m: module, locals: &mut out.entry_locals, params: &[], whole: &mut whole, pre: Vec::new(), ranges, assigns, loop_counter: None };
    scope.block(&mut body);
    out.entry_body = body;
    for f in &mut out.functions {
        let mut body = std::mem::take(&mut f.body);
        let assigns = count_assigns(&body, f.locals.len());
        let mut scope = Scope { m: module, locals: &mut f.locals, params: &f.params, whole: &mut whole, pre: Vec::new(), ranges: Ranges::default(), assigns, loop_counter: None };
        scope.block(&mut body);
        f.body = body;
    }
    for (decl, _) in out.bindings.iter_mut().zip(&whole).filter(|(_, whole)| **whole) {
        decl.element_stride = Some(decl.elem_ty.layout(&module.structs).0);
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod bounds_tests {
    use super::*;
    use crate::hal::BindingKind;

    /// A value of the evaluator: scalars, and arrays or vectors as `A`.
    #[derive(Clone, Debug, PartialEq)]
    enum V {
        B(bool),
        U(u32),
        I(i32),
        A(Vec<V>),
    }

    fn zero(ty: &KirType) -> V {
        match ty {
            KirType::Scalar(ScalarTy::Bool) => V::B(false),
            KirType::Scalar(ScalarTy::U32) | KirType::Atomic(ScalarTy::U32) => V::U(0),
            KirType::Scalar(ScalarTy::I32) | KirType::Atomic(ScalarTy::I32) => V::I(0),
            KirType::Vec { elem, lanes } => V::A(vec![zero(&KirType::Scalar(*elem)); usize::from(*lanes)]),
            KirType::Array { elem, len, .. } => V::A(vec![zero(elem); *len as usize]),
            other => unimplemented!("{other:?}"),
        }
    }

    fn idx(v: &V) -> usize {
        match v {
            V::U(u) => *u as usize,
            V::I(i) => usize::try_from(*i).expect("negative index reached memory"),
            other => panic!("index {other:?}"),
        }
    }

    macro_rules! int_ops {
        ($c:path, $x:expr, $y:expr, $op:expr) => {
            match $op {
                IrBinOp::Add => $c($x.wrapping_add($y)),
                IrBinOp::Sub => $c($x.wrapping_sub($y)),
                IrBinOp::Mul => $c($x.wrapping_mul($y)),
                IrBinOp::Lt => V::B($x < $y),
                IrBinOp::Le => V::B($x <= $y),
                IrBinOp::Gt => V::B($x > $y),
                IrBinOp::Ge => V::B($x >= $y),
                IrBinOp::Eq => V::B($x == $y),
                IrBinOp::Ne => V::B($x != $y),
                other => unimplemented!("{other:?}"),
            }
        };
    }

    enum Root {
        Local(usize),
        Storage(usize),
        Shared(usize),
    }

    /// Interprets the entry body. Storage is memory with a capacity (the vectors' length) and a
    /// bound (`counts`, what `ArrayLength` answers); any access past the capacity panics, so a
    /// test that survives an out-of-range index proves it was never evaluated.
    struct Machine<'a> {
        m: &'a KernelModule,
        storage: Vec<Vec<V>>,
        counts: Vec<u32>,
        shared: Vec<Vec<V>>,
        locals: Vec<Option<V>>,
        /// Atomic loads evaluated.
        loads: std::cell::Cell<u32>,
    }

    impl<'a> Machine<'a> {
        fn run(m: &'a KernelModule, storage: Vec<Vec<V>>, counts: Vec<u32>) -> Self {
            let shared = m.shared.iter().map(|d| vec![zero(&d.elem_ty); d.len.unwrap_or(1) as usize]).collect();
            let mut machine = Machine { m, storage, counts, shared, locals: vec![None; m.entry_locals.len()], loads: std::cell::Cell::new(0) };
            machine.block(&m.entry_body);
            machine
        }

        fn local(&self, l: usize) -> &V {
            self.locals[l].as_ref().expect("local was never declared")
        }

        fn follow(&self, mut v: V, steps: &[AccessStep]) -> V {
            for step in steps {
                let i = match step {
                    AccessStep::Field(k) => *k as usize,
                    AccessStep::Index(e) => idx(&self.eval(e)),
                };
                let V::A(items) = v else { panic!("access into a scalar") };
                v = items.get(i).expect("out-of-range access reached memory").clone();
            }
            v
        }

        fn eval(&self, e: &Expr) -> V {
            match e {
                Expr::LitBool(b) => V::B(*b),
                Expr::LitU32(u) => V::U(*u),
                Expr::LitI32(i) => V::I(*i),
                Expr::ZeroVal(t) => zero(t),
                Expr::Local(l) => self.local(l.0 as usize).clone(),
                Expr::ArrayLength { binding } => V::U(self.counts[*binding as usize]),
                Expr::Load { binding, index, .. } => self.storage[*binding as usize][idx(&self.eval(index))].clone(),
                Expr::StorageRead { binding, path, .. } => {
                    let b = *binding as usize;
                    let root = if self.m.bindings[b].is_unsized_array { V::A(self.storage[b].clone()) } else { self.storage[b][0].clone() };
                    self.follow(root, path)
                }
                Expr::SharedLoad { shared, index, .. } => self.shared[*shared as usize][idx(&self.eval(index))].clone(),
                Expr::SharedRead { shared, path, .. } => {
                    let s = *shared as usize;
                    let root = if self.m.shared[s].len.is_some() { V::A(self.shared[s].clone()) } else { self.shared[s][0].clone() };
                    self.follow(root, path)
                }
                Expr::Access { base, step, .. } => self.follow(self.eval(base), std::slice::from_ref(step)),
                Expr::Binary { op, lhs, rhs, .. } => match (self.eval(lhs), self.eval(rhs)) {
                    (V::B(x), V::B(y)) => match op {
                        IrBinOp::And => V::B(x && y),
                        IrBinOp::Or => V::B(x || y),
                        other => unimplemented!("{other:?}"),
                    },
                    (V::U(x), V::U(y)) => int_ops!(V::U, x, y, op),
                    (V::I(x), V::I(y)) => int_ops!(V::I, x, y, op),
                    other => unimplemented!("{other:?}"),
                },
                Expr::Ternary { op: IrTernaryOp::Select, a, b, c, .. } => match self.eval(a) {
                    V::B(true) => self.eval(b),
                    V::B(false) => self.eval(c),
                    other => panic!("select on {other:?}"),
                },
                Expr::Cast { expr, dst_ty, .. } => match (self.eval(expr), dst_ty) {
                    (V::I(x), KirType::Scalar(ScalarTy::U32)) => V::U(x as u32),
                    (V::U(x), KirType::Scalar(ScalarTy::I32)) => V::I(x as i32),
                    other => unimplemented!("{other:?}"),
                },
                Expr::AtomicLoad { target, .. } => {
                    self.loads.set(self.loads.get() + 1);
                    self.atomic_ref(target).clone()
                }
                other => unimplemented!("{other:?}"),
            }
        }

        fn atomic_index(&self, index: &Option<Box<Expr>>) -> usize {
            index.as_ref().map_or(0, |e| idx(&self.eval(e)))
        }

        fn atomic_ref(&self, t: &AtomicTarget) -> &V {
            match t {
                AtomicTarget::Storage { binding, index } => &self.storage[*binding as usize][self.atomic_index(index)],
                AtomicTarget::Shared { shared, index } => &self.shared[*shared as usize][self.atomic_index(index)],
            }
        }

        fn atomic_mut(&mut self, t: &AtomicTarget) -> &mut V {
            let i = match t {
                AtomicTarget::Storage { index, .. } | AtomicTarget::Shared { index, .. } => self.atomic_index(index),
            };
            match t {
                AtomicTarget::Storage { binding, .. } => &mut self.storage[*binding as usize][i],
                AtomicTarget::Shared { shared, .. } => &mut self.shared[*shared as usize][i],
            }
        }

        /// The root of a place and the indices from it down.
        fn flatten(&self, lv: &LValue) -> (Root, Vec<usize>) {
            let steps = |path: &[AccessStep]| -> Vec<usize> {
                path.iter()
                    .map(|s| match s {
                        AccessStep::Field(k) => *k as usize,
                        AccessStep::Index(e) => idx(&self.eval(e)),
                    })
                    .collect()
            };
            match lv {
                LValue::Local(l) => (Root::Local(l.0 as usize), Vec::new()),
                LValue::Storage { binding, path } => (Root::Storage(*binding as usize), steps(path)),
                LValue::Shared { shared, path } => (Root::Shared(*shared as usize), steps(path)),
                LValue::Index { base, index } => {
                    let (root, mut s) = self.flatten(base);
                    s.push(idx(&self.eval(index)));
                    (root, s)
                }
                LValue::Field { base, field } => {
                    let (root, mut s) = self.flatten(base);
                    s.push(*field as usize);
                    (root, s)
                }
                LValue::Private(_) => unimplemented!("private"),
            }
        }

        fn place(&mut self, lv: &LValue) -> &mut V {
            let (root, steps) = self.flatten(lv);
            let (mut cur, rest): (&mut V, &[usize]) = match root {
                Root::Local(l) => (self.locals[l].as_mut().expect("local was never declared"), &steps[..]),
                Root::Storage(b) if self.m.bindings[b].is_unsized_array => (&mut self.storage[b][steps[0]], &steps[1..]),
                Root::Storage(b) => (&mut self.storage[b][0], &steps[..]),
                Root::Shared(s) if self.m.shared[s].len.is_some() => (&mut self.shared[s][steps[0]], &steps[1..]),
                Root::Shared(s) => (&mut self.shared[s][0], &steps[..]),
            };
            for &i in rest {
                let V::A(items) = cur else { panic!("write into a scalar") };
                cur = &mut items[i];
            }
            cur
        }

        fn block(&mut self, block: &[Stmt]) {
            for s in block {
                self.stmt(s);
            }
        }

        fn stmt(&mut self, s: &Stmt) {
            match s {
                Stmt::Let { local, init } => self.locals[local.0 as usize] = Some(self.eval(init)),
                Stmt::Var { local, init } => {
                    let v = match init {
                        Some(e) => self.eval(e),
                        None => zero(&self.m.entry_locals[local.0 as usize].1),
                    };
                    self.locals[local.0 as usize] = Some(v);
                }
                Stmt::Assign { dst, val } => {
                    let v = self.eval(val);
                    *self.place(dst) = v;
                }
                Stmt::Store { binding, index, val } => {
                    let (i, v) = (idx(&self.eval(index)), self.eval(val));
                    self.storage[*binding as usize][i] = v;
                }
                Stmt::SharedStore { shared, index, val } => {
                    let (i, v) = (idx(&self.eval(index)), self.eval(val));
                    self.shared[*shared as usize][i] = v;
                }
                Stmt::If { cond, then_body, else_body } => match self.eval(cond) {
                    V::B(true) => self.block(then_body),
                    V::B(false) => self.block(else_body),
                    other => panic!("if on {other:?}"),
                },
                Stmt::AtomicStore { target, val } => {
                    let v = self.eval(val);
                    *self.atomic_mut(target) = v;
                }
                Stmt::AtomicRmw { dst, op, target, val, .. } => {
                    let v = self.eval(val);
                    let cell = self.atomic_mut(target);
                    let old = cell.clone();
                    let (V::U(a), V::U(b)) = (&old, &v) else { unimplemented!("non-u32 atomics") };
                    *cell = V::U(match op {
                        IrAtomicOp::Add => a.wrapping_add(*b),
                        IrAtomicOp::Sub => a.wrapping_sub(*b),
                        IrAtomicOp::Min => *a.min(b),
                        IrAtomicOp::Max => *a.max(b),
                        IrAtomicOp::And => a & b,
                        IrAtomicOp::Or => a | b,
                        IrAtomicOp::Xor => a ^ b,
                        IrAtomicOp::Exchange => *b,
                    });
                    if let Some(d) = dst {
                        self.locals[d.0 as usize] = Some(old);
                    }
                }
                Stmt::AtomicCompareExchange { dst_old, dst_exchanged, target, cmp, val, .. } => {
                    let (cmp, v) = (self.eval(cmp), self.eval(val));
                    let cell = self.atomic_mut(target);
                    let old = cell.clone();
                    let exchanged = old == cmp;
                    if exchanged {
                        *cell = v;
                    }
                    self.locals[dst_old.0 as usize] = Some(old);
                    self.locals[dst_exchanged.0 as usize] = Some(V::B(exchanged));
                }
                Stmt::Loop { body, continuing, break_if } => {
                    for _ in 0..1000 {
                        self.block(body);
                        self.block(continuing);
                        if let Some(c) = break_if
                            && self.eval(c) == V::B(true)
                        {
                            return;
                        }
                    }
                    panic!("loop did not end");
                }
                other => unimplemented!("{other:?}"),
            }
        }
    }

    fn lu(k: u32) -> Expr {
        Expr::LitU32(k)
    }

    fn local(l: u32) -> Expr {
        Expr::Local(LocalId(l))
    }

    fn id(l: u32) -> LocalId {
        LocalId(l)
    }

    fn unsized_binding(name: &str, kind: BindingKind, elem_ty: KirType) -> BindingDecl {
        BindingDecl { name: name.into(), kind, elem_ty, is_unsized_array: true, element_stride: Some(4), min_size: None }
    }

    fn load(binding: u32, index: Expr) -> Expr {
        Expr::Load { binding, index: index.boxed(), ty: KirType::U32 }
    }

    fn locals(types: &[(KirType, bool)]) -> Vec<(String, KirType, bool)> {
        types.iter().enumerate().map(|(i, (ty, m))| (format!("l{i}"), ty.clone(), *m)).collect()
    }

    fn module(bindings: Vec<BindingDecl>, shared: Vec<SharedDecl>, entry_locals: Vec<(String, KirType, bool)>, entry_body: Vec<Stmt>) -> KernelModule {
        KernelModule {
            label: "t".into(),
            entry: "main".into(),
            workgroup_size: [1, 1, 1],
            uses_exact_f32: false,
            structs: Vec::new(),
            bindings,
            shared,
            privates: Vec::new(),
            functions: Vec::new(),
            entry_builtins: Vec::new(),
            entry_locals,
            entry_body,
        }
    }

    fn words(values: impl IntoIterator<Item = u32>) -> Vec<V> {
        values.into_iter().map(V::U).collect()
    }

    fn src_dst() -> Vec<BindingDecl> {
        vec![
            unsized_binding("src", BindingKind::StorageRead, KirType::U32),
            unsized_binding("dst", BindingKind::StorageReadWrite, KirType::U32),
        ]
    }

    #[test]
    fn constant_in_range_accesses_need_no_checks() {
        let arr = KirType::Array { elem: Box::new(KirType::U32), len: 4, stride: 4 };
        let m = module(
            Vec::new(),
            vec![SharedDecl { name: "s".into(), elem_ty: KirType::U32, len: Some(4) }],
            locals(&[(arr.clone(), true), (KirType::U32, false), (KirType::U32, false)]),
            vec![
                Stmt::Var { local: id(0), init: None },
                Stmt::SharedStore { shared: 0, index: lu(3), val: lu(1) },
                Stmt::Assign { dst: LValue::Index { base: Box::new(LValue::Local(id(0))), index: lu(2).boxed() }, val: lu(5) },
                Stmt::Let { local: id(1), init: Expr::SharedLoad { shared: 0, index: lu(3).boxed(), ty: KirType::U32 } },
                Stmt::Let { local: id(2), init: Expr::Access { base: local(0).boxed(), step: AccessStep::Index(lu(1).boxed()), ty: KirType::U32 } },
            ],
        );
        assert!(matches!(with_bounds_checks(&m), Cow::Borrowed(_)));
    }

    #[test]
    fn in_range_storage_accesses_behave_as_before() {
        let m = module(
            src_dst(),
            Vec::new(),
            locals(&[(KirType::U32, false), (KirType::U32, false)]),
            vec![
                Stmt::Let { local: id(0), init: lu(2) },
                Stmt::Let { local: id(1), init: load(0, local(0)) },
                Stmt::Store { binding: 1, index: local(0), val: local(1) },
            ],
        );
        let checked = with_bounds_checks(&m);
        assert!(matches!(checked, Cow::Owned(_)));
        let run = |m: &KernelModule| Machine::run(m, vec![words(100..108), words([50; 8])], vec![4, 4]).storage[1].clone();
        assert_eq!(run(&m), run(&checked));
        assert_eq!(run(&checked)[2], V::U(102));
    }

    #[test]
    fn out_of_range_storage_reads_are_zero_and_writes_are_skipped() {
        let prefix = vec![
            Stmt::Let { local: id(0), init: lu(5) },
            Stmt::Let { local: id(1), init: load(0, local(0)) },
            Stmt::Store { binding: 1, index: lu(0), val: local(1) },
            Stmt::Store { binding: 1, index: local(0), val: lu(7) },
            Stmt::Store { binding: 1, index: lu(2), val: load(0, lu(1)) },
        ];
        let mut body = prefix.clone();
        body.extend([
            Stmt::Let { local: id(2), init: Expr::LitI32(-1) },
            Stmt::Let { local: id(3), init: load(0, local(2)) },
            Stmt::Store { binding: 1, index: lu(1), val: local(3) },
            Stmt::Store { binding: 1, index: local(2), val: lu(9) },
            Stmt::Store { binding: 1, index: lu(6), val: lu(9) },
        ]);
        let types = locals(&[(KirType::U32, false), (KirType::U32, false), (KirType::I32, false), (KirType::U32, false)]);
        let storage = || vec![words(100..108), words([50; 8])];

        // Without the pass the capacity (8), not the bound (4), limits the accesses.
        let unchecked = module(src_dst(), Vec::new(), types.clone(), prefix);
        let raw = Machine::run(&unchecked, storage(), vec![4, 4]);
        assert_eq!((&raw.storage[1][0], &raw.storage[1][5]), (&V::U(105), &V::U(7)));

        let m = module(src_dst(), Vec::new(), types, body);
        let checked = with_bounds_checks(&m);
        let run = Machine::run(&checked, storage(), vec![4, 4]);
        assert_eq!(run.storage[1], words([0, 0, 101, 50, 50, 50, 50, 50]), "reads yield zero, writes past the bound are skipped");
        assert_eq!(run.storage[0], words(100..108), "the source is untouched");
    }

    #[test]
    fn fixed_arrays_in_shared_and_local_memory_are_bounded() {
        let arr = KirType::Array { elem: Box::new(KirType::U32), len: 4, stride: 4 };
        let access = |i| Expr::Access { base: local(0).boxed(), step: AccessStep::Index(i), ty: KirType::U32 };
        let m = module(
            Vec::new(),
            vec![SharedDecl { name: "s".into(), elem_ty: KirType::U32, len: Some(4) }],
            locals(&[
                (arr, true),
                (KirType::U32, false),
                (KirType::U32, false),
                (KirType::U32, false),
                (KirType::U32, false),
                (KirType::I32, false),
            ]),
            vec![
                Stmt::Var { local: id(0), init: None },
                Stmt::Let { local: id(1), init: lu(6) },
                Stmt::SharedStore { shared: 0, index: lu(1), val: lu(11) },
                Stmt::SharedStore { shared: 0, index: local(1), val: lu(99) },
                Stmt::Assign { dst: LValue::Index { base: Box::new(LValue::Local(id(0))), index: lu(2).boxed() }, val: lu(22) },
                Stmt::Assign { dst: LValue::Index { base: Box::new(LValue::Local(id(0))), index: local(1).boxed() }, val: lu(98) },
                Stmt::Let { local: id(2), init: Expr::SharedLoad { shared: 0, index: local(1).boxed(), ty: KirType::U32 } },
                Stmt::Let { local: id(3), init: Expr::SharedLoad { shared: 0, index: lu(1).boxed(), ty: KirType::U32 } },
                Stmt::Let { local: id(4), init: access(local(1).boxed()) },
                Stmt::Let { local: id(5), init: Expr::LitI32(-2) },
                Stmt::SharedStore { shared: 0, index: local(5), val: lu(97) },
            ],
        );
        let checked = with_bounds_checks(&m);
        let run = Machine::run(&checked, Vec::new(), Vec::new());
        assert_eq!(run.shared[0], words([0, 11, 0, 0]));
        assert_eq!(run.local(0), &V::A(words([0, 0, 22, 0])));
        assert_eq!((run.local(2), run.local(3), run.local(4)), (&V::U(0), &V::U(11), &V::U(0)));
    }

    #[test]
    fn nested_paths_check_every_dynamic_index() {
        let row = KirType::Array { elem: Box::new(KirType::U32), len: 3, stride: 4 };
        let at = |j: u32| vec![AccessStep::Index(local(0).boxed()), AccessStep::Index(local(j).boxed())];
        let m = module(
            Vec::new(),
            vec![SharedDecl { name: "s".into(), elem_ty: row, len: Some(2) }],
            locals(&[(KirType::U32, false), (KirType::U32, false), (KirType::U32, false), (KirType::U32, false), (KirType::U32, false)]),
            vec![
                Stmt::Let { local: id(0), init: lu(1) },
                Stmt::Let { local: id(1), init: lu(2) },
                Stmt::Let { local: id(2), init: lu(3) },
                Stmt::Assign { dst: LValue::Shared { shared: 0, path: at(1) }, val: lu(5) },
                Stmt::Assign { dst: LValue::Shared { shared: 0, path: at(2) }, val: lu(6) },
                Stmt::Let { local: id(3), init: Expr::SharedRead { shared: 0, path: at(1), ty: KirType::U32 } },
                Stmt::Let { local: id(4), init: Expr::SharedRead { shared: 0, path: at(2), ty: KirType::U32 } },
            ],
        );
        let checked = with_bounds_checks(&m);
        let run = Machine::run(&checked, Vec::new(), Vec::new());
        assert_eq!(run.shared[0], vec![V::A(words([0, 0, 0])), V::A(words([0, 0, 5]))]);
        assert_eq!((run.local(3), run.local(4)), (&V::U(5), &V::U(0)));
    }

    #[test]
    fn a_binding_that_is_not_an_array_is_guarded_as_a_whole() {
        let x = BindingDecl { name: "x".into(), kind: BindingKind::StorageReadWrite, elem_ty: KirType::U32, is_unsized_array: false, element_stride: None, min_size: None };
        let m = module(
            vec![x],
            Vec::new(),
            locals(&[(KirType::U32, false)]),
            vec![
                Stmt::Assign { dst: LValue::Storage { binding: 0, path: Vec::new() }, val: lu(7) },
                Stmt::Let { local: id(0), init: Expr::StorageRead { binding: 0, path: Vec::new(), ty: KirType::U32 } },
            ],
        );
        let checked = with_bounds_checks(&m);
        assert_eq!(checked.bindings[0].element_stride, Some(4), "bytes / 4 counts whole objects");
        assert_eq!(m.bindings[0].element_stride, None, "the caller's module is untouched");
        let fits = Machine::run(&checked, vec![words([3])], vec![1]);
        assert_eq!((&fits.storage[0][0], fits.local(0)), (&V::U(7), &V::U(7)));
        let short = Machine::run(&checked, vec![words([3])], vec![0]);
        assert_eq!((&short.storage[0][0], short.local(0)), (&V::U(3), &V::U(0)));
    }

    #[test]
    fn atomic_results_stay_in_scope_and_are_zero_when_skipped() {
        let hist = unsized_binding("hist", BindingKind::StorageReadWrite, KirType::Atomic(ScalarTy::U32));
        let out = unsized_binding("out", BindingKind::StorageReadWrite, KirType::U32);
        let at = |i: Expr| AtomicTarget::Storage { binding: 0, index: Some(i.boxed()) };
        let flag = |l: u32| Expr::Ternary { op: IrTernaryOp::Select, a: local(l).boxed(), b: lu(1).boxed(), c: lu(0).boxed(), ty: KirType::U32 };
        let u = (KirType::U32, false);
        let b = (KirType::BOOL, false);
        let m = module(
            vec![hist, out],
            Vec::new(),
            locals(&[u.clone(), u.clone(), u.clone(), u.clone(), u.clone(), b.clone(), u.clone(), b]),
            vec![
                Stmt::Let { local: id(0), init: lu(1) },
                Stmt::AtomicRmw { dst: Some(id(1)), op: IrAtomicOp::Add, elem_ty: ScalarTy::U32, target: at(local(0)), val: lu(5) },
                Stmt::Store { binding: 1, index: lu(0), val: local(1) },
                Stmt::Let { local: id(2), init: lu(6) },
                Stmt::AtomicRmw { dst: Some(id(3)), op: IrAtomicOp::Add, elem_ty: ScalarTy::U32, target: at(local(2)), val: lu(5) },
                Stmt::Store { binding: 1, index: lu(1), val: local(3) },
                Stmt::AtomicRmw { dst: None, op: IrAtomicOp::Add, elem_ty: ScalarTy::U32, target: at(local(2)), val: lu(5) },
                Stmt::AtomicCompareExchange { dst_old: id(4), dst_exchanged: id(5), elem_ty: ScalarTy::U32, target: at(lu(0)), cmp: lu(10), val: lu(77) },
                Stmt::Store { binding: 1, index: lu(2), val: local(4) },
                Stmt::Store { binding: 1, index: lu(3), val: flag(5) },
                Stmt::AtomicCompareExchange { dst_old: id(6), dst_exchanged: id(7), elem_ty: ScalarTy::U32, target: at(local(2)), cmp: lu(0), val: lu(1) },
                Stmt::Store { binding: 1, index: lu(4), val: local(6) },
                Stmt::Store { binding: 1, index: lu(5), val: flag(7) },
                Stmt::AtomicStore { target: at(local(2)), val: lu(1) },
            ],
        );
        let checked = with_bounds_checks(&m);
        let before = m.entry_locals.len();
        assert!(checked.entry_locals.len() > before, "fresh temporaries");
        for l in [1, 3, 4, 5, 6, 7] {
            assert!(checked.entry_locals[l].2, "result local {l} is assigned after its declaration");
        }
        let run = Machine::run(&checked, vec![words([10, 20, 30, 40, 0, 0, 0, 0]), words([99; 8])], vec![4, 8]);
        assert_eq!(run.storage[0], words([77, 25, 30, 40, 0, 0, 0, 0]), "only the in-range cells changed");
        assert_eq!(run.storage[1], words([20, 0, 10, 1, 0, 0, 99, 99]), "skipped atomics report zero and false");
    }

    fn loaded() -> Expr {
        Expr::AtomicLoad { target: AtomicTarget::Storage { binding: 0, index: Some(lu(0).boxed()) }, ty: ScalarTy::U32 }
    }

    #[test]
    fn an_index_with_effects_is_evaluated_once() {
        let hist = unsized_binding("hist", BindingKind::StorageReadWrite, KirType::Atomic(ScalarTy::U32));
        let m = module(
            vec![hist, unsized_binding("src", BindingKind::StorageRead, KirType::U32), unsized_binding("dst", BindingKind::StorageReadWrite, KirType::U32)],
            Vec::new(),
            locals(&[(KirType::U32, false)]),
            vec![
                Stmt::Store { binding: 2, index: loaded(), val: lu(7) },
                Stmt::Let { local: id(0), init: load(1, loaded()) },
                Stmt::Store { binding: 2, index: lu(0), val: local(0) },
            ],
        );
        let checked = with_bounds_checks(&m);
        let run = |first: u32| {
            let storage = vec![words([first, 0, 0, 0, 0, 0, 0, 0]), words(100..108), words([50; 8])];
            let machine = Machine::run(&checked, storage, vec![4, 4, 4]);
            (machine.loads.get(), machine.storage[2].clone())
        };
        assert_eq!(run(2), (2, words([102, 50, 7, 50, 50, 50, 50, 50])), "each index is loaded once, checked and used");
        assert_eq!(run(6), (2, words([0, 50, 50, 50, 50, 50, 50, 50])), "out of range: write skipped, read zero, loads unchanged");
    }

    fn select(cond: Expr, then: Expr, otherwise: Expr) -> Expr {
        Expr::Ternary { op: IrTernaryOp::Select, a: cond.boxed(), b: then.boxed(), c: otherwise.boxed(), ty: KirType::U32 }
    }

    fn effect_bindings() -> Vec<BindingDecl> {
        vec![
            unsized_binding("hist", BindingKind::StorageReadWrite, KirType::Atomic(ScalarTy::U32)),
            unsized_binding("src", BindingKind::StorageRead, KirType::U32),
            unsized_binding("dst", BindingKind::StorageReadWrite, KirType::U32),
        ]
    }

    #[test]
    fn a_select_arm_with_effects_runs_only_when_selected() {
        // (condition, hist[0], atomic loads that must happen, dst[0])
        for (cond, first, loads, expected) in [(true, 2, 1, 102), (true, 6, 1, 0), (false, 2, 0, 9)] {
            let m = module(
                effect_bindings(),
                Vec::new(),
                locals(&[(KirType::U32, false)]),
                vec![
                    Stmt::Let { local: id(0), init: select(Expr::LitBool(cond), load(1, loaded()), lu(9)) },
                    Stmt::Store { binding: 2, index: lu(0), val: local(0) },
                ],
            );
            let checked = with_bounds_checks(&m);
            let run = Machine::run(&checked, vec![words([first, 0, 0, 0, 0, 0, 0, 0]), words(100..108), words([50; 8])], vec![4, 4, 4]);
            assert_eq!((run.loads.get(), run.storage[2][0].clone()), (loads, V::U(expected)), "condition {cond}, hist[0] = {first}");
        }
    }

    #[test]
    fn effects_in_a_loop_exit_condition_run_once_per_iteration() {
        let m = module(
            effect_bindings(),
            Vec::new(),
            locals(&[(KirType::U32, true)]),
            vec![
                Stmt::Var { local: id(0), init: Some(lu(0)) },
                Stmt::Loop {
                    body: vec![Stmt::Assign {
                        dst: LValue::Local(id(0)),
                        val: Expr::Binary { op: IrBinOp::Add, lhs: local(0).boxed(), rhs: lu(1).boxed(), ty: KirType::U32 },
                    }],
                    continuing: Vec::new(),
                    break_if: Some(Expr::Binary {
                        op: IrBinOp::Ge,
                        lhs: local(0).boxed(),
                        rhs: select(Expr::LitBool(true), load(1, loaded()), lu(0)).boxed(),
                        ty: KirType::BOOL,
                    }),
                },
                Stmt::Store { binding: 2, index: lu(0), val: local(0) },
            ],
        );
        let checked = with_bounds_checks(&m);
        assert!(matches!(&checked.entry_body[1], Stmt::Loop { continuing, .. } if !continuing.is_empty()), "the lets run after `continuing`, before the exit test");
        // hist[0] = 2 selects src[2] = 3: the loop ends at n = 3, after three tests.
        let run = Machine::run(&checked, vec![words([2, 0, 0, 0, 0, 0, 0, 0]), words(1..9), words([50; 8])], vec![4, 4, 4]);
        assert_eq!((run.loads.get(), run.storage[2][0].clone()), (3, V::U(3)));
    }

    #[test]
    fn constant_invalid_accesses_are_rewritten() {
        let m = module(
            Vec::new(),
            vec![SharedDecl { name: "s".into(), elem_ty: KirType::U32, len: Some(4) }],
            locals(&[(KirType::U32, false)]),
            vec![
                Stmt::SharedStore { shared: 0, index: lu(9), val: lu(1) },
                Stmt::SharedStore { shared: 0, index: lu(2), val: lu(3) },
                Stmt::Let { local: id(0), init: Expr::SharedLoad { shared: 0, index: lu(9).boxed(), ty: KirType::U32 } },
            ],
        );
        let checked = with_bounds_checks(&m);
        assert!(matches!(checked, Cow::Owned(_)));
        let run = Machine::run(&checked, Vec::new(), Vec::new());
        assert_eq!(run.shared[0], words([0, 0, 3, 0]));
        assert_eq!(run.local(0), &V::U(0));
    }
    /// The number of guards (selects and `if`s the pass added) in the entry body of `src` after
    /// the pass.
    fn guards_of(src: &str) -> usize {
        let parsed = crate::kir::parse_wgsl(src, "main", None, crate::hal::Features::SUBGROUP, [256, 256, 64], 256, 8, false).expect("parses").1;
        let checked = with_bounds_checks(&parsed);
        let mut count = 0;
        let mut body = checked.entry_body.clone();
        walk_stmts(&body, &mut |s| {
            if let Stmt::If { cond, .. } = s {
                // The kernels below have no `if` of their own besides loop guards.
                count += usize::from(!matches!(cond, Expr::Binary { op: IrBinOp::Lt, rhs, .. } if matches!(**rhs, Expr::LitU32(_))));
            }
        });
        rewrite_exprs(&mut body, &mut |e| count += usize::from(matches!(e, Expr::Ternary { op: IrTernaryOp::Select, .. })));
        count
    }

    fn tiled(shared_len: u32, limit: &str, step: &str, body: &str) -> String {
        format!(
            "var<workgroup> t: array<vec4<f32>, {shared_len}>;
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<uniform> p: vec4<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    var acc = vec4<f32>(0.0);
    for (var kk = 0u; kk < {limit}; kk = kk + {step}) {{
        {body}
    }}
    t[li] = acc;
    out[li] = acc.x;
}}"
        )
    }

    #[test]
    fn workgroup_reads_proven_in_range_need_no_guard() {
        // `li` < 256 and the sum stays below the array; the global store is of unknown bound.
        let unrolled = "let o = (kk + 1u) * 32u; acc = acc + t[o + lid.y] + t[o + lid.y + 16u];";
        // kk takes 0, 2, .. 14: the largest index is 15 * 32 + 15 + 16 = 511.
        assert_eq!(guards_of(&tiled(512, "16u", "2u", unrolled)), 1, "only the store to `out`");
        // One element fewer and the last index is out of range: both reads (and the store) are guarded.
        assert_eq!(guards_of(&tiled(511, "16u", "2u", unrolled)), 2, "the read past the end and the store to `out`");
    }

    #[test]
    fn a_counter_in_a_nested_loop_is_proven_too() {
        // The inner variable is declared and assigned inside a loop of unknown length.
        let nested = "for (var k0 = 0u; k0 < p.x; k0 = k0 + 16u) {
            for (var j = 0u; j < 16u; j = j + 2u) { acc = acc + t[(j + 1u) * 32u + lid.y + 16u]; }
        }";
        let src = tiled(512, "1u", "1u", nested);
        assert_eq!(guards_of(&src), 2, "the outer loop's test and the store to `out`");
        assert_eq!(guards_of(&src.replace("512", "511")), 3, "one short: the read is guarded too");
    }

    #[test]
    fn counters_without_a_proof_keep_their_guards() {
        let read = "acc = acc + t[kk * 32u + lid.y];";
        // The limit is a runtime value (its test is an `if` of the kernel too).
        assert_eq!(guards_of(&tiled(512, "p.x", "1u", read)), 3);
        // The counter is also assigned in the body.
        assert_eq!(guards_of(&tiled(512, "16u", "1u", "acc = acc + t[kk * 32u + lid.y]; kk = kk + 1u;")), 2);
        // The step makes the last value 15 and 15 * 32 + 15 does not fit 480 elements.
        assert_eq!(guards_of(&tiled(480, "16u", "1u", read)), 2);
        // Proven: 15 * 32 + 15 = 495 < 512.
        assert_eq!(guards_of(&tiled(512, "16u", "1u", read)), 1);
    }
}
