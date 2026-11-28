//! The rules of tier 1 for comprehensions that move indices or use what a product's operands are known to be (07 §3.2, rules 1, 3,
//! 5, 6a, 6b and 11 of W3-R4R5). Each is a pure function of one form, the way [`T1Rule`] asks.
//!
//! * 1 `contract_view`, `Exact`: a pending view an operand reads is composed into the access's map (the K order is unchanged).
//! * 3 `contract_k1`, `Exact`: a sum over one element is the element plus `+0.0` (the kernels accumulate from `+0.0`, so the `-0`
//!   of a product becomes `+0`; a plain product would differ in that bit).
//! * 6b `diag_of_product`, `Exact`: `diag(a·b)` is one contraction per diagonal element (two parallel dims become one). It changes
//!   the output shape, so it is not a [`T1Rule`] of the hook; [`diag_of_product`] is called by `Array::diag`.
//! * 11 `gram_symmetric`, `Exact`: `aᵀa` and `a·aᵀ` are tagged `SYMMETRIC`.
//! * 5 `contract_identity` and 6a `contract_diag`, `Reformulating`: a product with the identity is the other operand; with a
//!   diagonal matrix a broadcast product. They need the property of the operand (`Access::props`, set where the form of a product
//!   is made) and, for 6a, the vector, recovered from the constructor of `diag_matrix` (its node names).
//!
//! [`product`] is the eager path's use of the same rules: the pre-filter [`needs_t1`] says whether a `matmul` call may be rewritten at
//! all, and only a rewrite that leaves no contraction (3, 5, 6a) replaces the call; for the rest the product runs as it always did
//! (the native GEMM reads a transposed operand in place, and nothing changes there), and a tag (11) is put on its result.

use std::panic::Location;
use std::sync::Arc;

use smallvec::smallvec;

use super::registry::{Rid, RULES};
use super::t1::{t1, T1Rule};
use crate::array::Array;
use crate::context::Context;
use crate::jit::comprehension::build::Bodies;
use crate::jit::comprehension::lower;
use crate::jit::comprehension::{Access, AffineMap, Body, CombineOp, CombinePart, Comprehension, Compensation, DimKind, Epilogue, MapRef, OpTag, Origin, OutMap, Props, MAX_DIMS};
use crate::jit::program::{Instr, Ref};
use crate::jit::structured::RunState;
use crate::jit::{BinOp, Edge, ElemTy, Op, OpKind, Scalar, ShiftKind, Src};
use crate::numerics::Numerics;
use crate::profile::{Decision, DecisionSite};

use super::scope::same_src;

fn info(id: Rid) -> &'static super::registry::RuleInfo {
    &RULES[id as usize]
}

/// The comprehension rules of tier 1 that [`t1`](super::t1) runs, in the order of [`RULES`].
pub(crate) static INDEX_RULES: [T1Rule; 5] = [
    T1Rule { info: &RULES[Rid::ContractView as usize], apply: contract_view },
    T1Rule { info: &RULES[Rid::ContractK1 as usize], apply: contract_k1 },
    T1Rule { info: &RULES[Rid::GramSymmetric as usize], apply: gram_symmetric },
    T1Rule { info: &RULES[Rid::ContractIdentity as usize], apply: contract_identity },
    T1Rule { info: &RULES[Rid::ContractDiag as usize], apply: contract_diag },
];

// -- shared pieces of the forms --

/// The form is a product of two operands: `n` and `m` parallel, `k` reduce, `x0 * x1` added, as the builders make it or after a
/// rule moved the maps.
pub(super) fn is_product(form: &Comprehension) -> bool {
    form.inputs.len() == 2
        && form.space.len() == 3
        && matches!(form.combiner.as_ref().map(|combiner| (&combiner.op, combiner.compensation)), Some((CombineOp::Single(CombinePart::Add), Compensation::None)))
        && matches!(form.origin, Origin::Api(OpTag::Matmul { .. }) | Origin::Rewritten(_))
        && matches!(form.epilogue, Epilogue::None)
        && form.ty == ElemTy::F32
        && !form.space[0].is_reduce()
        && !form.space[1].is_reduce()
        && matches!(form.space[2].kind, DimKind::Reduce)
        && form.output.dim_of_axis == [Some(0), Some(1), None, None]
        && form.inputs.iter().all(|access| access.indirect.is_none() && access.edge == Edge::None)
}

/// The product of `x0 * x1` (the body of a product), not anything else over two accesses.
fn product_body(form: &Comprehension) -> bool {
    matches!(&form.body.instrs[..], [Instr { op, args: [Ref::Input(0), Ref::Input(1), _] }] if matches!(op.kind, OpKind::Binary(BinOp::Mul)))
}

const LHS: AffineMap = AffineMap::following([2, 1]);
const RHS: AffineMap = AffineMap::following([0, 2]);

/// The form of the API product (maps of `contract`), which the property rules need: they read what the operands are by position.
pub(super) fn api_product(form: &Comprehension) -> bool {
    is_product(form) && product_body(form) && *form.inputs[0].map == LHS && *form.inputs[1].map == RHS && form.inputs.iter().all(|access| access.shape[2..] == [1, 1])
}

fn elementwise_form(form: &Comprehension, inputs: smallvec::SmallVec<[Access; 2]>, body: Arc<Vec<Instr>>) -> Comprehension {
    Comprehension {
        space: smallvec![form.space[0].clone(), form.space[1].clone()],
        inputs,
        body: Body { instrs: body, scalars: smallvec![], outs: smallvec![0] },
        combiner: None,
        epilogue: Epilogue::None,
        output: OutMap::new([Some(0), Some(1), None, None]),
        props: Props::NONE,
        origin: form.origin.clone(),
        ty: form.ty,
    }
}

fn access_of(src: &Src, template: &Access, shape: [u32; 4], map: AffineMap) -> Access {
    Access { src: src.clone(), shape, ty: template.ty, map: MapRef::owned(map), edge: Edge::None, fill: None, indirect: None, props: Props::NONE }
}

// -- rule 1 --

/// Composes a pending view an access reads into the access's map. The sum order is the order of the reduce dim, which the maps
/// keep.
fn contract_view(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    if !form.has_reduce() || form.inputs.len() != 2 || !matches!(form.origin, Origin::Api(OpTag::Matmul { .. } | OpTag::MatmulBatched) | Origin::Rewritten(_)) {
        return None;
    }
    let mut out: Option<Comprehension> = None;
    for j in 0..2 {
        let current = out.as_ref().unwrap_or(form);
        let Src::Node(node) = &current.inputs[j].src else { continue };
        if !node.is_pending_view() || node.is_looped() || node.deferred_value().is_some() {
            continue;
        }
        let access = &current.inputs[j];
        if access.edge != Edge::None || access.indirect.is_some() {
            continue;
        }
        let Some([source, fill, _]) = node.operands() else { continue };
        let OpKind::Shift(ShiftKind::View { map: view, edge }) = node.op().kind else { continue };
        if access.shape != view.out || !matches!(source, Src::Node(_) | Src::Buffer(_)) {
            continue;
        }
        let fill = match (edge, &fill) {
            (Edge::Fill, Src::Scalar(value)) => Some(*value),
            (Edge::Fill, _) => continue,
            _ => None,
        };
        let Some(map) = AffineMap::from_view(&view, [0, 1, 2, 3]).then(&access.map) else { continue };
        let next = out.get_or_insert_with(|| form.clone());
        let access = &mut next.inputs[j];
        access.src = source;
        access.shape = view.src;
        access.map = MapRef::owned(map);
        access.edge = edge;
        access.fill = fill;
    }
    out
}

// -- rule 3 --

/// Removes the dim `d` (of extent 1, not on an output axis) from `form`: the coefficients of the later dims move down.
fn drop_dim(form: &mut Comprehension, d: usize) {
    form.space.remove(d);
    for access in &mut form.inputs {
        let mut map = *access.map;
        for row in &mut map.rows {
            for at in d..MAX_DIMS - 1 {
                row.coeff[at] = row.coeff[at + 1];
            }
            row.coeff[MAX_DIMS - 1] = 0;
        }
        access.map = MapRef::owned(map);
    }
    for axis in form.output.dim_of_axis.iter_mut().flatten() {
        if usize::from(*axis) > d {
            *axis -= 1;
        }
    }
}

/// A reduction over one element: the element plus `+0.0`.
fn contract_k1(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    let combiner = form.combiner.as_ref()?;
    if !matches!(combiner.op, CombineOp::Single(CombinePart::Add)) || combiner.compensation != Compensation::None || form.ty != ElemTy::F32 || !matches!(form.epilogue, Epilogue::None) {
        return None;
    }
    // An explicit tf32 product asked for the rounding of its operands: it keeps its kernel.
    if matches!(form.origin, Origin::Api(OpTag::Matmul { tf32: Some(true) })) {
        return None;
    }
    let mut reduce = form.space.iter().enumerate().filter(|(_, dim)| dim.is_reduce());
    let (d, dim) = reduce.next()?;
    if reduce.next().is_some() || dim.extent != 1 || !matches!(dim.kind, DimKind::Reduce) {
        return None;
    }
    if form.body.outs.len() != 1 || form.inputs.iter().any(|access| access.indirect.is_some()) || form.body.instrs.iter().any(|instr| matches!(instr.op.kind, OpKind::Generate(_))) {
        return None;
    }
    let mut next = form.clone();
    drop_dim(&mut next, d);
    let mut instrs: Vec<Instr> = form.body.instrs.to_vec();
    let mut scalars = form.body.scalars.clone();
    scalars.push(Scalar::f32(0.0));
    let value = form.body.outs[0];
    instrs.push(Instr { op: Op::new(OpKind::Binary(BinOp::Add), ElemTy::F32), args: [Ref::Tmp(value), Ref::Scalar(scalars.len() as u32 - 1), Ref::Unused] });
    next.body = Body { outs: smallvec![instrs.len() as u32 - 1], instrs: Arc::new(instrs), scalars };
    next.combiner = None;
    Some(next)
}

// -- rule 6b --

/// `diag(a·b)` as one contraction per element: the two parallel dims of the product are one (`m := n`). `None` unless `form` is a
/// square product. Changes the output shape (`[n, n]` to `[n]`).
pub(crate) fn diag_of_product_form(form: &Comprehension) -> Option<Comprehension> {
    if !is_product(form) || form.space[0].extent != form.space[1].extent || form.inputs.iter().any(|access| access.indirect.is_some()) {
        return None;
    }
    let mut next = form.clone();
    for access in &mut next.inputs {
        let mut map = *access.map;
        for row in &mut map.rows {
            row.coeff[0] += row.coeff[1];
            row.coeff[1] = 0;
        }
        access.map = MapRef::owned(map);
    }
    // Dim 1 goes: dim 2 (the reduce dim) becomes dim 1.
    next.space.remove(1);
    for access in &mut next.inputs {
        let mut map = *access.map;
        for row in &mut map.rows {
            for at in 1..MAX_DIMS - 1 {
                row.coeff[at] = row.coeff[at + 1];
            }
            row.coeff[MAX_DIMS - 1] = 0;
        }
        access.map = MapRef::owned(map);
    }
    next.output = OutMap::new([Some(0), None, None, None]);
    next.origin = Origin::Rewritten(info(Rid::DiagOfProduct).id);
    Some(next)
}

/// `diag(array)` where `array` is a product that has not been forced: the product of the diagonal only, `None` where the rule does
/// not apply (the caller makes the view as always).
pub(crate) fn diag_of_product(array: &Array<f32>, location: &'static Location<'static>) -> Option<Array<f32>> {
    let value = array.structured()?;
    if !value.is_unforced() {
        return None;
    }
    let dims = array.shape().dims();
    if dims[0] != dims[1] || dims[2] != 1 || dims[3] != 1 || dims[0] == 1 {
        return None;
    }
    let RunState::Contraction(run) = &value.comp().run else { return None };
    let ctx = array.context();
    if !ctx.jit().rules.t1_active {
        return None;
    }
    let form = run.build_form(location)?;
    if matches!(form.origin, Origin::Api(OpTag::Matmul { tf32: Some(true) })) {
        return None;
    }
    let form = diag_of_product_form(&t1(form, ctx))?;
    if !lower::lowers(&form) {
        return None;
    }
    note(ctx, location, Rid::DiagOfProduct);
    Some(lower::build(ctx, &form))
}

// -- rule 11 --

fn swap01(map: &AffineMap) -> AffineMap {
    let mut swapped = *map;
    for row in &mut swapped.rows {
        row.coeff.swap(0, 1);
    }
    swapped
}

/// `aᵀa` and `a·aᵀ`: both operands read one source through maps that differ by swapping the two output dims.
fn gram_symmetric(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    if !is_product(form) || form.props.has(Props::SYMMETRIC) || form.space[0].extent != form.space[1].extent {
        return None;
    }
    let [lhs, rhs] = &form.inputs[..] else { return None };
    if !same_src(&lhs.src, &rhs.src) || lhs.shape != rhs.shape || *rhs.map != swap01(&lhs.map) {
        return None;
    }
    let mut next = form.clone();
    next.props = Props { flags: next.props.flags | Props::SYMMETRIC, ..next.props }.closed();
    Some(next)
}

// -- rules 5 and 6a --

/// The vector `v` of a `v.diag_matrix(0)` node: its source, shape and the view that reads it.
fn diagonal_vector(src: &Src) -> Option<(Src, [u32; 4])> {
    let Src::Node(node) = src else { return None };
    if node.op().kind != OpKind::Select {
        return None;
    }
    let [_, diagonal, zero] = node.operands()?;
    if !matches!(zero, Src::Scalar(value) if value.bits == 0) {
        return None;
    }
    let Src::Node(view) = diagonal else { return None };
    let OpKind::Shift(ShiftKind::View { map, edge: Edge::Clamp }) = view.op().kind else { return None };
    if view.name != "diag_matrix" || map.axes[0].start != 0 || map.axes[0].step != 1 || map.axes[1].step != 0 || map.src[2..] != [1, 1] || map.src[1] != 1 {
        return None;
    }
    let [source, _, _] = view.operands()?;
    Some((source, map.src))
}

/// `a·identity` and `identity·a` are `a`.
fn contract_identity(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    if !api_product(form) {
        return None;
    }
    // The operand that is not the identity, read at the output's coordinates (`n`, `m`).
    let kept = if form.inputs[1].props.has(Props::IDENTITY) && form.space[2].extent == form.space[0].extent {
        0
    } else if form.inputs[0].props.has(Props::IDENTITY) && form.space[2].extent == form.space[1].extent {
        1
    } else {
        return None;
    };
    let other = &form.inputs[kept];
    let map = AffineMap::following([0, 1]);
    let mut next = elementwise_form(form, smallvec![access_of(&other.src, other, other.shape, map)], Bodies::x0(form.ty));
    next.origin = Origin::Rewritten(info(Rid::ContractIdentity).id);
    Some(next)
}

/// `a·diag_matrix(v)` is `a ⊙ v` (columns scaled), `diag_matrix(v)·a` is `v ⊙ a` (rows scaled).
fn contract_diag(form: &Comprehension, _: &Numerics) -> Option<Comprehension> {
    if !api_product(form) {
        return None;
    }
    let (diagonal, kept, vector_dim) = if form.inputs[1].props.has(Props::DIAGONAL) && form.space[2].extent == form.space[0].extent {
        (1, 0, 0)
    } else if form.inputs[0].props.has(Props::DIAGONAL) && form.space[2].extent == form.space[1].extent {
        (0, 1, 1)
    } else {
        return None;
    };
    let (vector, shape) = diagonal_vector(&form.inputs[diagonal].src)?;
    let other = &form.inputs[kept];
    let a = access_of(&other.src, other, other.shape, AffineMap::following([0, 1]));
    let v = access_of(&vector, other, shape, AffineMap::following([vector_dim]));
    let mut next = elementwise_form(form, smallvec![a, v], Bodies::x0_times_x1(form.ty));
    next.origin = Origin::Rewritten(info(Rid::ContractDiag).id);
    Some(next)
}

// -- the eager path --

/// Whether a `matmul` of `lhs` and `rhs` may be rewritten by a rule of tier 1: a few integer and flag tests, so that the common
/// product pays nothing else. The product is a candidate when `K` is 1 or an operand knows what it is (a property). A gram product
/// (`aᵀa`) is not a candidate here: its tag is set by the product itself (W3-R6) and by rule 11 on the form of a lazy product; a
/// test of the operands' sources on every eager product cost 1.6% of the host time of `launchcost/matmul32`.
#[inline(always)]
pub(crate) fn needs_t1(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>) -> bool {
    ctx.jit().rules.t1_active && (lhs.shape().dims()[0] == 1 || lhs.has_props() || rhs.has_props())
}

/// What the rules of tier 1 make of a product that [`needs_t1`].
pub(crate) enum Product {
    /// Nothing rewrote it, or what they made still contracts: it runs as it always did.
    Unchanged,
    /// It runs as it always did and its result has these properties.
    Props(Props),
    /// A rule replaced it (the elementwise array the form is).
    Replaced(Array<f32>),
}

/// The rules of tier 1 over the form of the product `lhs · rhs` (see the module documentation).
pub(crate) fn product(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, attrs: crate::jit::structured::contraction::ContractionAttrs, location: &'static Location<'static>) -> Product {
    let view = crate::jit::structured::contraction::ContractionView { lhs, rhs, attrs, location };
    let Some(form) = with_operand_props(view.form(), lhs, rhs) else { return Product::Unchanged };
    let before = form.origin.clone();
    let form = t1(form, ctx);
    if form.origin == before {
        return Product::Unchanged;
    }
    if form.has_reduce() {
        return if form.props.is_none() { Product::Unchanged } else { Product::Props(form.props) };
    }
    match lower::elementwise(ctx, &form, location) {
        // (`t1` logged the rule.)
        Some(array) => Product::Replaced(array),
        None => Product::Unchanged,
    }
}

fn note(ctx: &Context, location: &'static Location<'static>, rule: Rid) {
    if ctx.jit().recording_decisions() {
        ctx.jit().note_decision(Decision::new("matmul", location, DecisionSite::Rewrite, info(rule).name));
    }
}

/// The form of a product with the properties of the operands on its inputs.
pub(crate) fn with_operand_props(form: Option<Comprehension>, lhs: &Array<f32>, rhs: &Array<f32>) -> Option<Comprehension> {
    let mut form = form?;
    if lhs.has_props() || rhs.has_props() {
        form.inputs[0].props = lhs.props();
        form.inputs[1].props = rhs.props();
    }
    Some(form)
}
