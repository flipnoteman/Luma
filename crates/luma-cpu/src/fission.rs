//! Barrier loop fission and local scratch classification for `luma-cpu`.

use std::collections::{BTreeSet, HashMap, HashSet};

use luma_core::kir::{
    AccessStep, BarrierScope, Expr, KernelModule, KirType, LValue, LocalId, Stmt, walk_stmts,
};

use crate::abi::{CpuPlan, PhaseStmt};

fn is_workgroup_barrier(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Barrier(BarrierScope::Workgroup | BarrierScope::StorageAndWorkgroup)
    )
}

/// Whether any statement (recursively) is a workgroup-scope barrier.
fn has_workgroup_barrier(stmts: &[Stmt]) -> bool {
    let mut found = false;
    walk_stmts(stmts, &mut |s| found |= is_workgroup_barrier(s));
    found
}

/// Whether a `break` of the enclosing loop is reachable through `if`s only.
fn has_direct_break(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Break => true,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => has_direct_break(then_body) || has_direct_break(else_body),
        _ => false,
    })
}

fn flush(cur: &mut Vec<Stmt>, out: &mut Vec<PhaseStmt>) {
    if !cur.is_empty() {
        out.push(PhaseStmt::Segment {
            stmts: std::mem::take(cur),
            may_break_outer: false,
        });
    }
}

fn split_block(stmts: &[Stmt]) -> Vec<PhaseStmt> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for s in stmts {
        match s {
            s if is_workgroup_barrier(s) => flush(&mut cur, &mut out),
            Stmt::If {
                cond,
                then_body,
                else_body,
            } if has_workgroup_barrier(then_body) || has_workgroup_barrier(else_body) => {
                flush(&mut cur, &mut out);
                out.push(PhaseStmt::If {
                    cond: cond.clone(),
                    then_body: split_block(then_body),
                    else_body: split_block(else_body),
                });
            }
            Stmt::Loop {
                body,
                continuing,
                break_if,
            } if has_workgroup_barrier(body) || has_workgroup_barrier(continuing) => {
                flush(&mut cur, &mut out);
                let mut body_phases = split_block(body);
                for phase in &mut body_phases {
                    if let PhaseStmt::Segment {
                        stmts,
                        may_break_outer,
                    } = phase
                    {
                        *may_break_outer = has_direct_break(stmts);
                    }
                }
                out.push(PhaseStmt::Loop {
                    body: body_phases,
                    continuing: split_block(continuing),
                    break_if: break_if.clone(),
                });
            }
            s => cur.push(s.clone()),
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn lvalue_root_local(lv: &LValue) -> Option<LocalId> {
    match lv {
        LValue::Local(l) => Some(*l),
        LValue::Index { base, .. } | LValue::Field { base, .. } => lvalue_root_local(base),
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
        _ => {}
    }
}

fn expr_locals(e: &Expr, f: &mut dyn FnMut(LocalId)) {
    e.walk(&mut |x| {
        if let Expr::Local(l) = x {
            f(*l);
        }
    });
}

/// Calls `f(local, declared)` for every local mentioned by `stmts`, nested blocks included.
fn stmt_locals(stmts: &[Stmt], f: &mut dyn FnMut(LocalId, bool)) {
    walk_stmts(stmts, &mut |s| {
        match s {
            Stmt::Let { local, .. } | Stmt::Var { local, .. } => f(*local, true),
            Stmt::AtomicRmw { dst: Some(l), .. } => f(*l, true),
            Stmt::AtomicCompareExchange {
                dst_old,
                dst_exchanged,
                ..
            } => {
                f(*dst_old, true);
                f(*dst_exchanged, true);
            }
            Stmt::Assign { dst, .. } => {
                if let Some(l) = lvalue_root_local(dst) {
                    f(l, false);
                }
            }
            _ => {}
        }
        if let Stmt::Assign { dst, .. } = s {
            lvalue_exprs(dst, &mut |e| expr_locals(e, &mut |l| f(l, false)));
        }
        s.for_each_expr(&mut |e| expr_locals(e, &mut |l| f(l, false)));
    });
}

#[derive(Default)]
struct LocalUse {
    segments: HashSet<usize>,
    declared_in: HashSet<usize>,
    outer: bool,
}

fn collect_uses(phases: &[PhaseStmt], next_seg: &mut usize, uses: &mut HashMap<u32, LocalUse>) {
    for p in phases {
        match p {
            PhaseStmt::Segment { stmts, .. } => {
                let id = *next_seg;
                *next_seg += 1;
                stmt_locals(stmts, &mut |l, declared| {
                    let u = uses.entry(l.0).or_default();
                    u.segments.insert(id);
                    if declared {
                        u.declared_in.insert(id);
                    }
                });
            }
            PhaseStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                expr_locals(cond, &mut |l| uses.entry(l.0).or_default().outer = true);
                collect_uses(then_body, next_seg, uses);
                collect_uses(else_body, next_seg, uses);
            }
            PhaseStmt::Loop {
                body,
                continuing,
                break_if,
            } => {
                collect_uses(body, next_seg, uses);
                collect_uses(continuing, next_seg, uses);
                if let Some(c) = break_if {
                    expr_locals(c, &mut |l| uses.entry(l.0).or_default().outer = true);
                }
            }
        }
    }
}

pub fn slot_bytes(ty: &KirType, module: &KernelModule) -> u32 {
    match ty {
        KirType::Scalar(luma_core::kir::ScalarTy::F64) => 8,
        KirType::Scalar(_) | KirType::Atomic(_) => 4,
        KirType::Vec { elem, lanes } => {
            u32::from(*lanes)
                * if *elem == luma_core::kir::ScalarTy::F64 {
                    8
                } else {
                    4
                }
        }
        KirType::Array { elem, len, .. } => len.max(&1) * slot_bytes(elem, module),
        KirType::Struct(id) => module.structs[*id as usize]
            .fields
            .iter()
            .map(|f| slot_bytes(&f.ty, module))
            .sum::<u32>()
            .max(4),
    }
}

fn collect_array_lengths(module: &KernelModule) -> BTreeSet<u32> {
    let mut set = BTreeSet::new();
    let mut scan = |block: &[Stmt]| {
        walk_stmts(block, &mut |s| {
            s.for_each_expr(&mut |e| {
                e.walk(&mut |x| {
                    if let Expr::ArrayLength { binding } = x {
                        set.insert(*binding);
                    }
                })
            })
        });
    };
    scan(&module.entry_body);
    for (f, live) in module.functions.iter().zip(module.reachable_functions()) {
        if live {
            scan(&f.body);
        }
    }
    set
}

/// Builds the CPU lowering plan: bindings, geometry, shared/scratch layout and barrier phases.
pub fn plan(module: &KernelModule, used: Option<&[bool]>, initialize_shared: bool) -> CpuPlan {
    let mut active_bindings = Vec::new();
    let mut binding_ranks = Vec::with_capacity(module.bindings.len());
    for b in 0..module.bindings.len() {
        if used.is_none_or(|u| u.get(b).copied().unwrap_or(false)) {
            binding_ranks.push(Some(active_bindings.len() as u32));
            active_bindings.push(b as u32);
        } else {
            binding_ranks.push(None);
        }
    }

    let array_lengths = collect_array_lengths(module)
        .into_iter()
        .map(|binding| {
            let d = module.bindings.get(binding as usize);
            let stride = d
                .and_then(|d| d.element_stride)
                .unwrap_or_else(|| d.map_or(1, |d| d.elem_ty.layout(&module.structs).0));
            (binding, stride.max(1))
        })
        .collect();

    let workgroup_size = module.workgroup_size.map(|d| d.max(1));
    let invocations = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    let padded = invocations.next_multiple_of(4);
    let simd_width = if invocations.is_multiple_of(4) { 4 } else { 1 };

    let mut shared_offsets = Vec::with_capacity(module.shared.len());
    let mut off = 0u32;
    for s in &module.shared {
        let (size, align) = s.elem_ty.layout(&module.structs);
        off = off.next_multiple_of(align.max(16));
        shared_offsets.push(off);
        off += size * s.len.unwrap_or(1).max(1);
    }
    let shared_bytes = off.next_multiple_of(16);

    let phases = split_block(&module.entry_body);

    let mut uses = HashMap::new();
    collect_uses(&phases, &mut 0, &mut uses);
    let builtin: HashSet<u32> = module.entry_builtins.iter().map(|&(_, l)| l.0).collect();

    let mut scratch = 0u32;
    let local_scratch_offsets = module
        .entry_locals
        .iter()
        .enumerate()
        .map(|(i, (_, ty, mutable))| {
            let id = i as u32;
            let needs_slot = if builtin.contains(&id) {
                *mutable
            } else {
                match uses.get(&id) {
                    None => false,
                    Some(u) => {
                        let private =
                            !u.outer && u.segments.len() == 1 && u.declared_in == u.segments;
                        !private
                    }
                }
            };
            needs_slot.then(|| {
                scratch = scratch.next_multiple_of(16);
                let o = scratch;
                scratch += padded * slot_bytes(ty, module);
                o
            })
        })
        .collect();

    let priv_scratch_offsets = module
        .privates
        .iter()
        .map(|p| {
            scratch = scratch.next_multiple_of(16);
            let o = scratch;
            scratch += padded * slot_bytes(&p.ty, module);
            o
        })
        .collect();
    let mut has_ret = false;
    walk_stmts(&module.entry_body, &mut |s| {
        has_ret |= matches!(s, Stmt::Return(_))
    });
    if has_ret && phases.len() > 1 {
        scratch = scratch.next_multiple_of(16) + padded * 4;
    }

    CpuPlan {
        active_bindings,
        binding_ranks,
        array_lengths,
        workgroup_size,
        invocations,
        simd_width,
        shared_offsets,
        shared_bytes,
        local_scratch_offsets,
        priv_scratch_offsets,
        scratch_bytes: scratch.next_multiple_of(16),
        initialize_shared,
        phases,
    }
}

#[cfg(test)]
mod tests {
    use luma_core::kir::{IrBinOp, ScalarTy};

    use super::*;

    fn module(locals: usize, body: Vec<Stmt>) -> KernelModule {
        KernelModule {
            label: "t".into(),
            entry: "main".into(),
            workgroup_size: [64, 1, 1],
            uses_exact_f32: false,
            structs: vec![],
            bindings: vec![],
            shared: vec![],
            privates: vec![],
            functions: vec![],
            entry_builtins: vec![],
            entry_locals: (0..locals)
                .map(|i| (format!("l{i}"), KirType::U32, true))
                .collect(),
            entry_body: body,
        }
    }

    fn var(l: u32, v: u32) -> Stmt {
        Stmt::Var {
            local: LocalId(l),
            init: Some(Expr::LitU32(v)),
        }
    }

    fn assign(dst: u32, src: u32) -> Stmt {
        Stmt::Assign {
            dst: LValue::Local(LocalId(dst)),
            val: Expr::Local(LocalId(src)),
        }
    }

    fn lt(l: u32) -> Expr {
        Expr::Binary {
            op: IrBinOp::Lt,
            lhs: Box::new(Expr::Local(LocalId(l))),
            rhs: Box::new(Expr::LitU32(4)),
            ty: KirType::BOOL,
        }
    }

    #[test]
    fn elementwise_is_one_register_segment() {
        let m = module(2, vec![var(0, 1), var(1, 2), assign(1, 0)]);
        let p = plan(&m, None, false);
        assert_eq!(p.phases.len(), 1);
        assert!(matches!(
            p.phases[0],
            PhaseStmt::Segment {
                may_break_outer: false,
                ..
            }
        ));
        assert!(p.local_scratch_offsets.iter().all(Option::is_none));
        assert_eq!(p.scratch_bytes, 0);
        assert_eq!(p.simd_width, 4);
    }

    #[test]
    fn local_crossing_barrier_gets_scratch() {
        let m = module(
            2,
            vec![
                var(0, 1),
                var(1, 2),
                Stmt::Barrier(BarrierScope::Workgroup),
                assign(1, 0),
            ],
        );
        let p = plan(&m, None, false);
        assert_eq!(p.phases.len(), 2);
        assert_eq!(p.local_scratch_offsets[0], Some(0));
        assert_eq!(p.local_scratch_offsets[1], Some(64 * 4));
        assert_eq!(p.scratch_bytes, 2 * 64 * 4);

        // Only the local used on both sides of the barrier is spilled.
        let m = module(
            2,
            vec![
                var(0, 1),
                var(1, 2),
                assign(1, 1),
                Stmt::Barrier(BarrierScope::Workgroup),
                assign(0, 0),
            ],
        );
        let p = plan(&m, None, false);
        assert_eq!(p.local_scratch_offsets[0], Some(0));
        assert_eq!(p.local_scratch_offsets[1], None);
    }

    #[test]
    fn storage_barrier_does_not_split() {
        let m = module(
            1,
            vec![
                var(0, 1),
                Stmt::Barrier(BarrierScope::Storage),
                assign(0, 0),
            ],
        );
        assert_eq!(plan(&m, None, false).phases.len(), 1);
    }

    #[test]
    fn if_with_barrier_splits() {
        let m = module(
            1,
            vec![
                var(0, 1),
                Stmt::If {
                    cond: lt(0),
                    then_body: vec![
                        var(0, 2),
                        Stmt::Barrier(BarrierScope::Workgroup),
                        assign(0, 0),
                    ],
                    else_body: vec![],
                },
            ],
        );
        let p = plan(&m, None, false);
        assert_eq!(p.phases.len(), 2);
        let PhaseStmt::If {
            then_body,
            else_body,
            ..
        } = &p.phases[1]
        else {
            panic!("expected If")
        };
        assert_eq!(then_body.len(), 2);
        assert!(else_body.is_empty());
        assert!(p.local_scratch_offsets[0].is_some());
    }

    #[test]
    fn loop_with_barriers_and_break() {
        let body = vec![
            Stmt::If {
                cond: lt(0),
                then_body: vec![Stmt::Break],
                else_body: vec![],
            },
            Stmt::Barrier(BarrierScope::Workgroup),
            assign(0, 0),
            Stmt::Barrier(BarrierScope::Workgroup),
        ];
        let m = module(
            1,
            vec![
                var(0, 0),
                Stmt::Loop {
                    body,
                    continuing: vec![],
                    break_if: None,
                },
            ],
        );
        let p = plan(&m, None, false);
        assert_eq!(p.phases.len(), 2);
        let PhaseStmt::Loop { body, .. } = &p.phases[1] else {
            panic!("expected Loop")
        };
        assert_eq!(body.len(), 2);
        assert!(matches!(
            body[0],
            PhaseStmt::Segment {
                may_break_outer: true,
                ..
            }
        ));
        assert!(matches!(
            body[1],
            PhaseStmt::Segment {
                may_break_outer: false,
                ..
            }
        ));
        assert!(p.local_scratch_offsets[0].is_some());
    }

    #[test]
    fn shared_layout_and_bindings() {
        let mut m = module(0, vec![]);
        m.shared = vec![
            luma_core::kir::SharedDecl {
                name: "a".into(),
                elem_ty: KirType::Scalar(ScalarTy::F32),
                len: Some(3),
            },
            luma_core::kir::SharedDecl {
                name: "b".into(),
                elem_ty: KirType::Scalar(ScalarTy::F32),
                len: None,
            },
        ];
        let p = plan(&m, Some(&[]), true);
        assert_eq!(p.shared_offsets, vec![0, 16]);
        assert_eq!(p.shared_bytes, 32);
        assert!(p.initialize_shared);
        assert!(p.phases.is_empty());
    }
}
