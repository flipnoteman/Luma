//! Which subgroup operations are provably reached by every lane of their warp.
//!
//! A subgroup operation normally reads `activemask` and takes a partial-warp path (a loop over the
//! active lanes) when some lanes are missing, which WGSL's non-uniform control flow allows. When
//! every lane of the warp must arrive, the operation is the full-warp form on a constant full mask
//! (`shfl.sync` and `vote.sync` wait for the whole mask), and the fallback is dead code that the
//! PTX-to-SASS compiler would otherwise have to compile.
//!
//! The analysis follows the entry body's structured control flow. A point is *converged* when every
//! lane of a warp that executes it is accompanied by all the others: that needs whole warps (the
//! workgroup size is a multiple of 32) and that no lane of a warp left or skipped anything the
//! others did not, so
//!
//! * every `if`/`switch` above it has a warp-uniform condition, or contains no `return`, `break` or
//!   `continue` and so merges again afterwards;
//! * every `loop` around it has warp-uniform exits (`break_if`, `break`, `continue` and `return`
//!   under uniform conditions only): lanes leaving a loop at different iterations meet again after
//!   it, but the iterations before are not converged;
//! * nothing before it (in its block or the blocks around) made some lanes `return`, or `break` or
//!   `continue` out of an enclosing construct, while others went on.
//!
//! Warp-uniform values are literals, uniform-buffer reads, `workgroup_id`, `num_workgroups`,
//! `subgroup_size`, `local_invocation_index / 32` (and `>> 5`), and what is computed from those
//! only by locals that are assigned uniform values at converged points. Anything loaded from
//! memory or computed from a thread id is divergent. Subgroup operations in helper functions are
//! never converged (their callers are not analysed).

use std::collections::HashSet;

use luma_core::kir::*;

/// The subgroup operations (by address) of `m`'s entry body that every lane of the warp reaches.
pub(super) fn convergent_subgroups(m: &KernelModule) -> HashSet<*const Expr> {
    let [wx, wy, wz] = m.workgroup_size;
    if !(wx * wy * wz).is_multiple_of(32) {
        return HashSet::new();
    }
    let mut any = false;
    walk_stmts(&m.entry_body, &mut |s| {
        s.for_each_expr(&mut |e| e.walk(&mut |x| any |= matches!(x, Expr::Subgroup { .. })));
    });
    if !any {
        return HashSet::new();
    }
    let n = m.entry_locals.len();
    let mut a = An { div: vec![false; n], lid: vec![false; n], lid_vec: vec![false; n], sg_size: vec![false; n], one_d: wy * wz == 1, changed: false, collect: false, out: HashSet::new() };
    for (b, l) in &m.entry_builtins {
        let l = l.0 as usize;
        match b {
            BuiltinVar::GlobalInvocationId | BuiltinVar::SubgroupInvocationId => a.div[l] = true,
            BuiltinVar::LocalInvocationId => {
                a.div[l] = true;
                a.lid_vec[l] = true;
            }
            BuiltinVar::LocalInvocationIndex => {
                a.div[l] = true;
                a.lid[l] = true;
            }
            BuiltinVar::SubgroupSize => a.sg_size[l] = true,
            BuiltinVar::WorkgroupId | BuiltinVar::NumWorkgroups => {}
        }
    }
    // Divergence only grows, so this ends.
    loop {
        a.changed = false;
        a.block(&m.entry_body, true);
        if !a.changed {
            break;
        }
    }
    a.collect = true;
    a.block(&m.entry_body, true);
    a.out
}

struct An {
    /// Locals that may differ between the lanes of a warp.
    div: Vec<bool>,
    /// Locals holding `local_invocation_index`, `local_invocation_id` and `subgroup_size`.
    lid: Vec<bool>,
    lid_vec: Vec<bool>,
    sg_size: Vec<bool>,
    /// The workgroup is `[x, 1, 1]`: `local_invocation_id.x` is the linear index.
    one_d: bool,
    changed: bool,
    collect: bool,
    out: HashSet<*const Expr>,
}

fn root(lv: &LValue) -> Option<LocalId> {
    match lv {
        LValue::Local(l) => Some(*l),
        LValue::Index { base, .. } | LValue::Field { base, .. } => root(base),
        _ => None,
    }
}

fn lvalue_exprs(lv: &LValue, f: &mut dyn FnMut(&Expr)) {
    match lv {
        LValue::Storage { path, .. } | LValue::Shared { path, .. } => {
            for s in path {
                if let AccessStep::Index(e) = s {
                    f(e);
                }
            }
        }
        LValue::Index { base, index } => {
            lvalue_exprs(base, f);
            f(index);
        }
        LValue::Field { base, .. } => lvalue_exprs(base, f),
        LValue::Local(_) | LValue::Private(_) => {}
    }
}

impl An {
    fn taint(&mut self, l: LocalId) {
        let d = &mut self.div[l.0 as usize];
        if !*d {
            *d = true;
            self.changed = true;
        }
    }

    fn is_lid(&self, e: &Expr) -> bool {
        match e {
            Expr::Local(l) => self.lid[l.0 as usize],
            Expr::Access { base, step: AccessStep::Field(0), .. } => self.one_d && matches!(&**base, Expr::Local(l) if self.lid_vec[l.0 as usize]),
            _ => false,
        }
    }

    /// Whether `e` has the same value in every lane of a warp.
    fn uniform(&self, e: &Expr) -> bool {
        match e {
            Expr::LitBool(_) | Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF16(_) | Expr::LitF32(_) | Expr::LitF64(_) | Expr::ZeroVal(_) | Expr::GridSize(_) | Expr::ArrayLength { .. } => true,
            Expr::Local(l) => !self.div[l.0 as usize],
            Expr::UniformRead { path, .. } => path.iter().all(|s| match s {
                AccessStep::Index(i) => self.uniform(i),
                AccessStep::Field(_) => true,
            }),
            Expr::FnParam(_) | Expr::Private(_) | Expr::Load { .. } | Expr::StorageRead { .. } | Expr::SharedLoad { .. } | Expr::SharedRead { .. } | Expr::Call { .. } | Expr::AtomicLoad { .. } | Expr::Subgroup { .. } | Expr::Texture { .. } => false,
            Expr::Access { base, step, .. } => {
                self.uniform(base)
                    && match step {
                        AccessStep::Index(i) => self.uniform(i),
                        AccessStep::Field(_) => true,
                    }
            }
            Expr::Swizzle { base, .. } => self.uniform(base),
            Expr::Compose { comps, .. } => comps.iter().all(|c| self.uniform(c)),
            Expr::Splat { val, .. } => self.uniform(val),
            Expr::Unary { arg, .. } => self.uniform(arg),
            Expr::Binary { op, lhs, rhs, .. } => {
                // A warp is 32 consecutive linear thread indices.
                let by_warp = match (op, &**rhs) {
                    (IrBinOp::Div, Expr::LitU32(32)) | (IrBinOp::Shr, Expr::LitU32(5)) => true,
                    (IrBinOp::Div, Expr::Local(l)) => self.sg_size[l.0 as usize],
                    _ => false,
                };
                (by_warp && self.is_lid(lhs)) || (self.uniform(lhs) && self.uniform(rhs))
            }
            Expr::Ternary { a, b, c, .. } => self.uniform(a) && self.uniform(b) && self.uniform(c),
            Expr::Cast { expr, .. } => self.uniform(expr),
        }
    }

    /// A `return` under a condition (or in a loop) that splits the warp.
    fn ret_splits(&self, b: &[Stmt], div: bool) -> bool {
        b.iter().any(|s| match s {
            Stmt::Return(_) => div,
            Stmt::If { cond, then_body, else_body } => {
                let d = div || !self.uniform(cond);
                self.ret_splits(then_body, d) || self.ret_splits(else_body, d)
            }
            Stmt::Switch { selector, cases, default } => {
                let d = div || !self.uniform(selector);
                cases.iter().any(|(_, body, _)| self.ret_splits(body, d)) || self.ret_splits(default, d)
            }
            Stmt::Loop { body, continuing, break_if } => {
                let d = div || !self.loop_uniform(body, continuing, break_if.as_ref());
                self.ret_splits(body, d) || self.ret_splits(continuing, d)
            }
            _ => false,
        })
    }

    /// A `break` or `continue` that leaves the enclosing loop under a condition that splits the
    /// warp (`in_switch`: a `break` leaves only the switch).
    fn jump_splits(&self, b: &[Stmt], div: bool, in_switch: bool) -> bool {
        b.iter().any(|s| match s {
            Stmt::Break => div && !in_switch,
            Stmt::Continue => div,
            Stmt::If { cond, then_body, else_body } => {
                let d = div || !self.uniform(cond);
                self.jump_splits(then_body, d, in_switch) || self.jump_splits(else_body, d, in_switch)
            }
            Stmt::Switch { selector, cases, default } => {
                let d = div || !self.uniform(selector);
                cases.iter().any(|(_, body, _)| self.jump_splits(body, d, true)) || self.jump_splits(default, d, true)
            }
            _ => false,
        })
    }

    /// Whether every lane of a warp runs every iteration of the loop.
    fn loop_uniform(&self, body: &[Stmt], continuing: &[Stmt], break_if: Option<&Expr>) -> bool {
        break_if.is_none_or(|e| self.uniform(e))
            && !self.jump_splits(body, false, false)
            && !self.jump_splits(continuing, false, false)
            && !self.ret_splits(body, false)
            && !self.ret_splits(continuing, false)
    }

    fn exprs(&mut self, s: &Stmt, conv: bool) {
        s.for_each_expr(&mut |e| self.expr(e, conv));
    }

    /// Marks the subgroup operations of `e`; `conv`: every lane evaluates `e`.
    fn expr(&mut self, e: &Expr, conv: bool) {
        if let Expr::Subgroup { .. } = e
            && conv
            && self.collect
        {
            self.out.insert(e as *const Expr);
        }
        if let Expr::Ternary { a, b, c, .. } = e {
            // A scalar condition evaluates only the chosen operand.
            let arm = conv && self.uniform(a);
            self.expr(a, conv);
            self.expr(b, arm);
            self.expr(c, arm);
            return;
        }
        e.for_each_child(&mut |c| self.expr(c, conv));
    }

    fn block(&mut self, b: &[Stmt], conv: bool) -> bool {
        let mut c = conv;
        for s in b {
            c = self.stmt(s, c);
        }
        c
    }

    /// Analyses `s` reached with `conv`; returns whether the lanes are together after it.
    fn stmt(&mut self, s: &Stmt, conv: bool) -> bool {
        match s {
            Stmt::Let { local, init } => {
                self.exprs(s, conv);
                if !conv || !self.uniform(init) {
                    self.taint(*local);
                }
                conv
            }
            Stmt::Var { local, init } => {
                self.exprs(s, conv);
                if !conv || init.as_ref().is_some_and(|e| !self.uniform(e)) {
                    self.taint(*local);
                }
                conv
            }
            Stmt::Assign { dst, val } => {
                self.exprs(s, conv);
                if let Some(l) = root(dst) {
                    let mut uni = self.uniform(val);
                    lvalue_exprs(dst, &mut |e| uni &= self.uniform(e));
                    if !conv || !uni {
                        self.taint(l);
                    }
                }
                conv
            }
            Stmt::AtomicRmw { dst, .. } => {
                self.exprs(s, conv);
                if let Some(d) = dst {
                    self.taint(*d);
                }
                conv
            }
            Stmt::AtomicCompareExchange { dst_old, dst_exchanged, .. } => {
                self.exprs(s, conv);
                self.taint(*dst_old);
                self.taint(*dst_exchanged);
                conv
            }
            Stmt::If { cond, then_body, else_body } => {
                self.exprs(s, conv);
                let arm = conv && self.uniform(cond);
                self.block(then_body, arm);
                self.block(else_body, arm);
                conv && !self.ret_splits(std::slice::from_ref(s), false) && !self.jump_splits(std::slice::from_ref(s), false, false)
            }
            Stmt::Switch { selector, cases, default } => {
                self.exprs(s, conv);
                let arm = conv && self.uniform(selector);
                for (_, body, _) in cases {
                    self.block(body, arm);
                }
                self.block(default, arm);
                conv && !self.ret_splits(std::slice::from_ref(s), false) && !self.jump_splits(std::slice::from_ref(s), false, true)
            }
            Stmt::Loop { body, continuing, break_if } => {
                let inside = conv && self.loop_uniform(body, continuing, break_if.as_ref());
                let through = self.block(body, inside);
                let after = self.block(continuing, inside && through);
                if let Some(e) = break_if {
                    self.expr(e, after);
                }
                conv && !self.ret_splits(std::slice::from_ref(s), false)
            }
            Stmt::Break | Stmt::Continue | Stmt::Return(_) | Stmt::Barrier(_) => {
                self.exprs(s, conv);
                conv
            }
            Stmt::Store { .. } | Stmt::SharedStore { .. } | Stmt::AtomicStore { .. } | Stmt::Discard(_) => {
                self.exprs(s, conv);
                conv
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(v: u32) -> Box<Expr> {
        Box::new(Expr::LitU32(v))
    }

    fn bin(op: IrBinOp, a: Box<Expr>, b: Box<Expr>, ty: KirType) -> Box<Expr> {
        Box::new(Expr::Binary { op, lhs: a, rhs: b, ty })
    }

    /// `lid` is local 0 (`local_invocation_index`), `sg` local 1 (`subgroup_size`).
    fn module(size: [u32; 3], body: Vec<Stmt>, extra_locals: usize) -> KernelModule {
        let mut mb = ModuleBuilder::new("t", "main", size);
        mb.entry_builder().finish();
        let mut m = mb.finish();
        m.entry_locals = vec![("lid".into(), KirType::U32, false), ("sg".into(), KirType::U32, false)];
        m.entry_locals.extend((0..extra_locals).map(|i| (format!("v{i}"), KirType::U32, true)));
        m.entry_builtins = vec![(BuiltinVar::LocalInvocationIndex, LocalId(0)), (BuiltinVar::SubgroupSize, LocalId(1))];
        m.entry_body = body;
        m
    }

    fn sum() -> Expr {
        Expr::Subgroup { op: IrSubgroupOp::Add, arg: Some(Box::new(Expr::LitF32(1.0))), lane: None, ty: KirType::F32 }
    }

    /// `let _ = subgroupAdd(1.0)` into local `into`.
    fn sg(into: u32) -> Stmt {
        Stmt::Var { local: LocalId(into), init: Some(Expr::Cast { kind: CastKind::Convert, expr: Box::new(sum()), src_ty: KirType::F32, dst_ty: KirType::U32 }) }
    }

    fn lid() -> Box<Expr> {
        Box::new(Expr::Local(LocalId(0)))
    }

    fn lt(a: Box<Expr>, b: u32) -> Expr {
        *bin(IrBinOp::Lt, a, lit(b), KirType::BOOL)
    }

    fn count(m: &KernelModule) -> usize {
        convergent_subgroups(m).len()
    }

    #[test]
    fn a_top_level_operation_is_converged_with_whole_warps_only() {
        assert_eq!(count(&module([64, 1, 1], vec![sg(2)], 1)), 1);
        assert_eq!(count(&module([48, 1, 1], vec![sg(2)], 1)), 0);
        assert_eq!(count(&module([32, 2, 1], vec![sg(2)], 1)), 1);
    }

    #[test]
    fn divergent_branches_and_early_exits_are_not_converged() {
        let div = || lt(lid(), 5);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond: div(), then_body: vec![sg(2)], else_body: vec![] }], 1)), 0);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond: div(), then_body: vec![], else_body: vec![sg(2)] }], 1)), 0);
        // Lanes that returned are missing afterwards.
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond: div(), then_body: vec![Stmt::Return(None)], else_body: vec![] }, sg(2)], 1)), 0);
        // A divergent branch without exits merges again.
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond: div(), then_body: vec![Stmt::Barrier(BarrierScope::Workgroup)], else_body: vec![] }, sg(2)], 1)), 1);
    }

    #[test]
    fn a_warp_uniform_branch_is_converged() {
        let warp = bin(IrBinOp::Div, lid(), Box::new(Expr::Local(LocalId(1))), KirType::U32);
        let cond = *bin(IrBinOp::Eq, warp, lit(0), KirType::BOOL);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond, then_body: vec![sg(2)], else_body: vec![] }], 1)), 1);
        let shr = bin(IrBinOp::Shr, lid(), lit(5), KirType::U32);
        let cond = *bin(IrBinOp::Eq, shr, lit(0), KirType::BOOL);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond, then_body: vec![sg(2)], else_body: vec![] }], 1)), 1);
        // Dividing by anything else is not uniform over a warp.
        let cond = *bin(IrBinOp::Eq, bin(IrBinOp::Div, lid(), lit(16), KirType::U32), lit(0), KirType::BOOL);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::If { cond, then_body: vec![sg(2)], else_body: vec![] }], 1)), 0);
    }

    #[test]
    fn locals_assigned_under_divergence_are_divergent() {
        let v = || Box::new(Expr::Local(LocalId(2)));
        let body = vec![
            Stmt::Var { local: LocalId(2), init: Some(Expr::LitU32(0)) },
            Stmt::If { cond: lt(lid(), 5), then_body: vec![Stmt::Assign { dst: LValue::Local(LocalId(2)), val: Expr::LitU32(1) }], else_body: vec![] },
            Stmt::If { cond: *bin(IrBinOp::Eq, v(), lit(1), KirType::BOOL), then_body: vec![sg(3)], else_body: vec![] },
        ];
        assert_eq!(count(&module([64, 1, 1], body, 2)), 0);
        // The same with a uniform assignment.
        let body = vec![
            Stmt::Var { local: LocalId(2), init: Some(Expr::LitU32(0)) },
            Stmt::Assign { dst: LValue::Local(LocalId(2)), val: Expr::LitU32(1) },
            Stmt::If { cond: *bin(IrBinOp::Eq, v(), lit(1), KirType::BOOL), then_body: vec![sg(3)], else_body: vec![] },
        ];
        assert_eq!(count(&module([64, 1, 1], body, 2)), 1);
    }

    #[test]
    fn loops_with_divergent_exits_converge_only_after_them() {
        // loop { if lid < 5 { break } ; sg }  sg
        let body = vec![Stmt::Loop {
            body: vec![Stmt::If { cond: lt(lid(), 5), then_body: vec![Stmt::Break], else_body: vec![] }, sg(2)],
            continuing: vec![],
            break_if: None,
        }, sg(2)];
        let m = module([64, 1, 1], body, 1);
        assert_eq!(count(&m), 1);
        // loop { sg; break_if lid < 5 }: the exit is divergent, the first iteration is converged
        // but the later ones are not, and the code is the same.
        let body = vec![Stmt::Loop { body: vec![sg(2)], continuing: vec![], break_if: Some(lt(lid(), 5)) }];
        assert_eq!(count(&module([64, 1, 1], body, 1)), 0);
        // A uniform exit keeps the body converged.
        let body = vec![Stmt::Loop { body: vec![sg(2)], continuing: vec![], break_if: Some(lt(Box::new(Expr::LitU32(1)), 5)) }];
        assert_eq!(count(&module([64, 1, 1], body, 1)), 1);
        // A return that splits the warp inside a loop: nothing after is converged.
        let body = vec![Stmt::Loop { body: vec![Stmt::If { cond: lt(lid(), 5), then_body: vec![Stmt::Return(None)], else_body: vec![] }], continuing: vec![], break_if: Some(Expr::LitBool(true)) }, sg(2)];
        assert_eq!(count(&module([64, 1, 1], body, 1)), 0);
    }

    #[test]
    fn a_lazily_evaluated_select_arm_is_not_converged() {
        let arm = Expr::Cast { kind: CastKind::Convert, expr: Box::new(sum()), src_ty: KirType::F32, dst_ty: KirType::U32 };
        let sel = |c: Expr| Expr::Ternary { op: IrTernaryOp::Select, a: Box::new(c), b: Box::new(arm.clone()), c: Box::new(Expr::LitU32(0)), ty: KirType::U32 };
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::Var { local: LocalId(2), init: Some(sel(lt(lid(), 5))) }], 1)), 0);
        assert_eq!(count(&module([64, 1, 1], vec![Stmt::Var { local: LocalId(2), init: Some(sel(Expr::LitBool(true))) }], 1)), 1);
    }
}
