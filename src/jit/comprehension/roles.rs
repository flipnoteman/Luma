//! Role inference: which dims of a contraction are batch, `m`, `n`, `k` (and which are free or reduce over one input only), from the
//! maps and the combiner (W2-R1 §1). MLIR's `inferContractionDims`, with the extension that a body which factors into two operand
//! expressions (`separate`) is a contraction of the union maps of each side.

use smallvec::SmallVec;

use super::{Comprehension, DimKind, Form, MAX_DIMS};
use crate::builtin::{BinOp, UnOp};
use crate::jit::program::{Instr, Ref};
use crate::jit::{ElemTy, OpKind};

use super::{Access, CombineOp, CombinePart};

/// A set of dims of a space, one bit per dim (`MAX_DIMS` = 6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub(crate) struct DimMask(u8);

impl DimMask {
    pub(crate) const EMPTY: DimMask = DimMask(0);

    pub(crate) const fn of(d: usize) -> DimMask {
        DimMask(1 << d)
    }

    pub(crate) const fn contains(self, d: usize) -> bool {
        self.0 >> d & 1 != 0
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub(crate) const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub(crate) const fn union(self, other: DimMask) -> DimMask {
        DimMask(self.0 | other.0)
    }

    pub(crate) const fn meet(self, other: DimMask) -> DimMask {
        DimMask(self.0 & other.0)
    }

    pub(crate) const fn minus(self, other: DimMask) -> DimMask {
        DimMask(self.0 & !other.0)
    }

    pub(crate) fn dims(self) -> impl Iterator<Item = usize> {
        (0..MAX_DIMS).filter(move |&d| self.contains(d))
    }
}

/// The roles of the dims of a contraction. `lhs` and `rhs` are the sides (`m` belongs to `lhs`, `n` to `rhs`): for a form of two
/// inputs they are the input indices, for a separated body the index of the side in [`Separation::sides`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Roles {
    pub(crate) batch: DimMask,
    pub(crate) m: DimMask,
    pub(crate) n: DimMask,
    pub(crate) k: DimMask,
    /// Parallel dims no input uses: the output repeats the value.
    pub(crate) free: DimMask,
    /// Reduce dims used by exactly one side: `sum_k x(i, k) * y(i)` is not a contraction.
    pub(crate) lone: DimMask,
    pub(crate) lhs: u8,
    pub(crate) rhs: u8,
}

/// Whether access `a` reads at a point that depends on dim `d`: some row has a coefficient on it, or the index of an indirect read does.
pub(crate) fn uses(a: &Access, d: usize) -> bool {
    a.map.rows.iter().any(|row| row.coeff[d] != 0) || a.indirect.as_ref().is_some_and(|indirect| indirect.index_map.rows.iter().any(|row| row.coeff[d] != 0))
}

/// The dims of the space with an extent above 1: the others carry no role.
fn live(c: &Comprehension) -> DimMask {
    c.space.iter().enumerate().filter(|(_, dim)| dim.extent > 1).fold(DimMask::EMPTY, |mask, (d, _)| mask.union(DimMask::of(d)))
}

fn parallel(c: &Comprehension) -> DimMask {
    c.space.iter().enumerate().filter(|(_, dim)| !dim.is_reduce()).fold(DimMask::EMPTY, |mask, (d, _)| mask.union(DimMask::of(d)))
}

/// The live dims access `a` uses.
pub(crate) fn used(c: &Comprehension, a: &Access) -> DimMask {
    (0..c.space.len()).filter(|&d| uses(a, d)).fold(DimMask::EMPTY, |mask, d| mask.union(DimMask::of(d))).meet(live(c))
}

/// The roles for two sides, each the union of the dims of the inputs in it. `None` for what is not a contraction of the form
/// (a lone reduce dim, a segmented dim, an indirect access, no `k`), whatever the combiner.
fn roles_of(c: &Comprehension, sides: &[&[u8]; 2]) -> Option<Roles> {
    if c.space.iter().any(|dim| matches!(dim.kind, DimKind::Segmented { .. })) || c.inputs.iter().any(|a| a.indirect.is_some()) {
        return None;
    }
    let live = live(c);
    // A window (a row of two dims) is not a contraction: the operand is not read through strides.
    if is_window(c, live) {
        return None;
    }
    let (p, r) = (parallel(c).meet(live), live.minus(parallel(c)));
    let side = |side: &[u8]| side.iter().fold(DimMask::EMPTY, |mask, &j| mask.union(used(c, &c.inputs[usize::from(j)])));
    let (u0, u1) = (side(sides[0]), side(sides[1]));
    // The side with parallel dims of its own is the lhs; ties go by order.
    let swap = u0.meet(p).minus(u1).is_empty() && !u1.meet(p).minus(u0).is_empty();
    let (lhs, rhs, ul, ur) = if swap { (1, 0, u1, u0) } else { (0, 1, u0, u1) };
    let both = ul.meet(ur);
    let roles = Roles {
        batch: both.meet(p),
        m: ul.meet(p).minus(ur),
        n: ur.meet(p).minus(ul),
        k: both.meet(r),
        free: p.minus(ul.union(ur)),
        lone: r.minus(both),
        lhs,
        rhs,
    };
    (roles.lone.is_empty() && !roles.k.is_empty()).then_some(roles)
}

/// Whether some row of some access reads two or more live dims.
fn is_window(c: &Comprehension, live: DimMask) -> bool {
    c.inputs.iter().any(|a| a.map.rows.iter().any(|row| row.coeff.iter().enumerate().filter(|&(d, &coeff)| coeff != 0 && live.contains(d)).count() >= 2))
}

/// What the combiner of a contraction may be, and the type.
fn finish(c: &Comprehension, roles: Roles, add_only: bool) -> Option<Roles> {
    let ok = match c.combiner.as_ref().map(|combiner| &combiner.op) {
        Some(CombineOp::Single(CombinePart::Add)) => true,
        Some(CombineOp::Single(_)) => !add_only,
        _ => false,
    };
    (ok && c.ty == ElemTy::F32).then_some(roles)
}

/// The roles of a contraction (`sum_k lhs * rhs`): `None` if the form is not one (see [`roles_of`]), its combiner is not `Add`, or
/// its type is not `f32` (the GEMM family is `f32`; `Df32` and `f16` go through the reduction tail).
pub(crate) fn infer_roles(c: &Comprehension) -> Option<Roles> {
    contraction_roles(c).and_then(|roles| finish(c, roles, true))
}

/// [`infer_roles`] for any single-part combiner: the roles of `(min, +)`-like contractions the generic lowering takes.
pub(crate) fn infer_contraction_roles(c: &Comprehension) -> Option<Roles> {
    contraction_roles(c).and_then(|roles| finish(c, roles, false))
}

fn contraction_roles(c: &Comprehension) -> Option<Roles> {
    match c.inputs.len() {
        2 => roles_of(c, &[&[0], &[1]]),
        0 | 1 => None,
        _ => separate(c).map(|separation| separation.roles),
    }
}

/// A body `x * y` whose two factors read disjoint sets of the inputs.
#[derive(Clone, Debug)]
pub(crate) struct Separation {
    /// The inputs of each side.
    pub(crate) sides: [SmallVec<[u8; 4]>; 2],
    /// The factors of each side (their product is the side's operand expression): inputs or instructions of the body.
    pub(crate) factors: [SmallVec<[Ref; 2]>; 2],
    pub(crate) roles: Roles,
}

/// The index of the instruction `r` names, looking through `Unary(Id)`; `None` for anything but an instruction.
fn through_id(instrs: &[Instr], mut r: Ref) -> Ref {
    while let Ref::Tmp(t) = r {
        match instrs.get(t as usize) {
            Some(Instr { op, args }) if matches!(op.kind, OpKind::Unary(UnOp::Id)) => r = args[0],
            _ => break,
        }
    }
    r
}

/// The leaves of the tree of `Mul`s `r` is the root of (through `Unary(Id)`); `None` if a leaf is not an input or an instruction.
fn factors(instrs: &[Instr], r: Ref, out: &mut SmallVec<[Ref; 8]>) -> Option<()> {
    match through_id(instrs, r) {
        Ref::Tmp(t) => match instrs.get(t as usize)? {
            Instr { op, args } if matches!(op.kind, OpKind::Binary(BinOp::Mul)) => {
                factors(instrs, args[0], out)?;
                factors(instrs, args[1], out)?
            }
            _ => out.push(Ref::Tmp(t)),
        },
        leaf @ Ref::Input(_) => out.push(leaf),
        _ => return None,
    }
    (out.len() <= 6).then_some(())
}

/// The inputs an instruction (or input) depends on, as a bit mask.
fn deps(instrs: &[Instr], r: Ref, memo: &mut [Option<u8>]) -> Option<u8> {
    match r {
        Ref::Input(j) => 1u8.checked_shl(j),
        Ref::Scalar(_) | Ref::Unused => Some(0),
        Ref::Tmp(t) => {
            if let Some(known) = memo[t as usize] {
                return Some(known);
            }
            let instr = &instrs[t as usize];
            let mut mask = 0;
            for arg in instr.args {
                mask |= deps(instrs, arg, memo)?;
            }
            memo[t as usize] = Some(mask);
            Some(mask)
        }
        Ref::Sub(_) => None,
    }
}

/// The inputs `mask` has, as indices.
fn inputs_of(mask: u8) -> SmallVec<[u8; 4]> {
    (0..8).filter(|j| mask >> j & 1 != 0).collect()
}

/// Splits the body into two operand expressions: the output is a product (looking through `Unary(Id)`) whose factors, grouped in
/// two, read disjoint inputs. A body `(a * b) * s` has the leaves `a`, `b`, `s`; of the groupings that make a contraction the one
/// that has both an `m` and an `n` and the more even sides wins, so a scale `s(k)` goes to the side with fewer inputs.
pub(crate) fn separate(c: &Comprehension) -> Option<Separation> {
    if c.body.outs.len() != 1 || c.inputs.len() < 2 {
        return None;
    }
    let instrs: &[Instr] = &c.body.instrs;
    let mut leaves = SmallVec::<[Ref; 8]>::new();
    let root = through_id(instrs, Ref::Tmp(c.body.outs[0]));
    // The output must be a product itself (a lone input is not).
    match root {
        Ref::Tmp(t) if matches!(instrs.get(t as usize)?.op.kind, OpKind::Binary(BinOp::Mul)) => factors(instrs, root, &mut leaves)?,
        _ => return None,
    }
    let mut memo = vec![None; instrs.len()];
    let leaf_deps: SmallVec<[u8; 8]> = leaves.iter().map(|&leaf| deps(instrs, leaf, &mut memo)).collect::<Option<_>>()?;
    // Fix the first leaf on side 0; every grouping of the others.
    let mut best: Option<((u8, i32), u32, Roles)> = None;
    for mask in 0..1u32 << (leaves.len() - 1) {
        let on_side = |i: usize| i > 0 && mask >> (i - 1) & 1 != 0;
        let (mut d0, mut d1) = (0u8, 0u8);
        for (i, &d) in leaf_deps.iter().enumerate() {
            if on_side(i) { d1 |= d } else { d0 |= d }
        }
        if d0 == 0 || d1 == 0 || d0 & d1 != 0 || d0 | d1 != (1u8 << c.inputs.len()) - 1 {
            continue;
        }
        let (s0, s1) = (inputs_of(d0), inputs_of(d1));
        let Some(roles) = roles_of(c, &[&s0, &s1]) else { continue };
        let key = (u8::from(!roles.m.is_empty() && !roles.n.is_empty()), -(s0.len() as i32 - s1.len() as i32).abs());
        if best.as_ref().is_none_or(|(known, ..)| key > *known) {
            best = Some((key, mask, roles));
        }
    }
    let (_, mask, roles) = best?;
    let mut sides = [SmallVec::new(), SmallVec::new()];
    let mut grouped = [SmallVec::new(), SmallVec::new()];
    for (i, (&leaf, &d)) in leaves.iter().zip(&leaf_deps).enumerate() {
        let s = usize::from(i > 0 && mask >> (i - 1) & 1 != 0);
        grouped[s].push(leaf);
        for j in inputs_of(d) {
            if !sides[s].contains(&j) {
                sides[s].push(j);
            }
        }
    }
    Some(Separation { sides, factors: grouped, roles })
}

/// Whether the one output of the body is a product of two values (through `Unary(Id)`).
fn body_is_product(c: &Comprehension) -> bool {
    let instrs: &[Instr] = &c.body.instrs;
    c.body.outs.len() == 1 && matches!(through_id(instrs, Ref::Tmp(c.body.outs[0])), Ref::Tmp(t) if matches!(instrs[t as usize].op.kind, OpKind::Binary(BinOp::Mul)))
}

/// Whether every access reads each source coordinate as a dim itself: rows are one dim with coefficient 1 or nothing, offset 0.
fn identity_like(c: &Comprehension) -> bool {
    c.inputs.iter().all(|a| {
        a.indirect.is_none() && a.edge == crate::jit::Edge::None && a.map.rows.iter().all(|row| row.offset == 0 && row.coeff.iter().filter(|&&coeff| coeff != 0).count() <= 1 && row.coeff.iter().all(|&coeff| coeff == 0 || coeff == 1))
    })
}

/// The class of a form: how the strategy of a form without a tag is chosen, and what every `Origin::Api` tag is checked against in
/// debug builds (the guarantee of 07 §4.1).
pub(crate) fn classify(c: &Comprehension) -> Form {
    let live = live(c);
    if is_window(c, live) {
        return Form::Window;
    }
    let live_reduce = c.space.iter().enumerate().any(|(d, dim)| dim.is_reduce() && live.contains(d));
    if !live_reduce {
        let p = parallel(c).meet(live);
        if let [x, y] = &c.inputs[..] {
            let (ux, uy) = (used(c, x).meet(p), used(c, y).meet(p));
            if !ux.is_empty() && !uy.is_empty() && ux.meet(uy).is_empty() && body_is_product(c) {
                return Form::Outer;
            }
        }
        return Form::Elementwise;
    }
    match infer_roles(c) {
        Some(roles) if roles.m.is_empty() && roles.n.is_empty() => Form::Dot,
        Some(roles) if roles.m.is_empty() || roles.n.is_empty() => Form::Gemv,
        Some(_) => Form::Gemm,
        None if identity_like(c) && c.combiner.as_ref().is_some_and(|combiner| combiner.op.parts().len() <= crate::builtin::MAX_REDUCTIONS) => Form::Reduce,
        None => Form::Generic,
    }
}
