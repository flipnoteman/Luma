//! The rules of tier 2 that reformulate products (07 §3.2, rules 7 and 9 of W3-R4R5), over a scope whose producers are products that
//! were not forced.
//!
//! * 7 `sum_of_product`, `Reformulating`: `sum(a·b)` is `Σ_k (Σ_m a(k,m))·(Σ_n b(n,k))`, and `sum_axis(a·b, j)` is the product of one
//!   operand with the sums of the other: two small reductions and a vector product instead of an `m·n·k` contraction. The sums
//!   add the terms in another order and a non-finite entry of `a` or `b` reaches the result through fewer terms (the class).
//! * 9 `chain_order`, `Reassociating`: a chain of products `(a·b)·c` is multiplied in the order the matrix-chain program
//!   ([`super::chain::best`]) finds cheapest. The rounding differs.
//!
//! Rule 8 (`factor_invariant`, a factor that does not depend on the reduce dim moves into the epilogue) needs W3-U4's epilogue
//! inputs and is not here.
//!
//! Both need a producer that is an API product with the maps `contract` makes ([`super::index::api_product`]) read by one consumer,
//! and both replace the root form and add forms whose operands are the outputs of earlier steps of the plan (feeds).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};

use super::chain::{LinkCosts, Tree, best};
use super::index::api_product;
use super::registry::{RULES, Rid};
use super::scope::{CompId, SComp, Scope, ScopeRole, Site, SiteKind};
use super::t2::{Feed, Rewrite, T2Rule};
use crate::builtin::{ReduceOp, UnOp};
use crate::context::Context;
use crate::array::Storage;
use crate::jit::comprehension::build::Bodies;
use crate::jit::comprehension::{AffineMap, Combiner, CombineOp, CombinePart, Comprehension, Dim, Epilogue, MapRef, OpTag, Origin, OutMap, Props, Access};
use crate::jit::structured::contraction::{ContractionAttrs, Geometry, Tf32Request};
use crate::jit::{Edge, ElemTy, OpKind, Src};
use crate::numerics::Numerics;

/// The rules of tier 2 of W3-R4R5 (rule 8 waits for W3-U4's epilogue inputs).
pub(crate) static REFORM_RULES: [T2Rule; 2] = [
    T2Rule { info: &RULES[Rid::ChainOrder as usize], sites: SiteKind::Comp, single: false, apply: chain_order },
    T2Rule { info: &RULES[Rid::SumOfProduct as usize], sites: SiteKind::Comp, single: false, apply: sum_of_product },
];

// -- shared pieces --

/// The context a scope's sources belong to (a rule has no context of its own).
fn context_of(scope: &Scope) -> Option<Context> {
    scope.comps.iter().flat_map(|comp| comp.form.inputs.iter()).find_map(|access| match &access.src {
        Src::Node(node) => Some(node.ctx.clone()),
        Src::Buffer(storage) => Some(storage.ctx().clone()),
        Src::Scalar(_) => None,
    })
}

/// A source of `len` elements that holds nothing yet: what an access reads until the plan feeds it the output of an earlier step.
fn placeholder(ctx: &Context, len: u64) -> Src {
    Src::Buffer(Arc::new(Storage::new(ctx, len as usize, ElemTy::F32.layout())))
}

/// The direct edge that makes access `access` of `consumer` read a producer that only `consumer` reads.
fn sole_producer(scope: &Scope, consumer: CompId, access: u8) -> Option<&SComp> {
    let edge = scope.edges.iter().find(|edge| edge.consumer == consumer && edge.access == access)?;
    if !(edge.direct || reads_leaf(&scope.comp(consumer)?.form.inputs[usize::from(access)])) || edge.planned || scope.edges.iter().filter(|other| other.producer == edge.producer).count() != 1 {
        return None;
    }
    scope.comp(edge.producer).filter(|comp| comp.role == ScopeRole::Producer)
}

/// Whether the access reads the leaf of a product, or the copy of it a reduction registers as its reader.
fn reads_leaf(access: &Access) -> bool {
    let Src::Node(node) = &access.src else { return false };
    node.deferred_value().is_some() || (node.op().kind == OpKind::Unary(UnOp::Id) && matches!(node.operands(), Some([Src::Node(inner), ..]) if inner.deferred_value().is_some()))
}

fn rewritten(form: &mut Comprehension, id: Rid) {
    form.origin = Origin::Rewritten(RULES[id as usize].id);
}

fn add_combiner() -> Option<Combiner> {
    Some(Combiner::of(CombineOp::Single(CombinePart::Add), ElemTy::F32))
}

fn operand(src: Src, shape: [u32; 4], map: AffineMap) -> Access {
    Access { src, shape, ty: ElemTy::F32, map: MapRef::owned(map), edge: Edge::None, fill: None, indirect: None, props: Props::NONE }
}

/// `map` with its dims renamed: dim `from` of the old space is dim `to` of the new one (`renames` lists them).
fn rename(map: &AffineMap, renames: &[(usize, usize)]) -> AffineMap {
    let mut out = AffineMap::ZERO;
    for (row, slot) in map.rows.iter().zip(&mut out.rows) {
        slot.offset = row.offset;
        for &(from, to) in renames {
            slot.coeff[to] = row.coeff[from];
        }
    }
    out
}

fn form_of(space: SmallVec<[Dim; 4]>, inputs: SmallVec<[Access; 2]>, instrs: Arc<Vec<crate::jit::program::Instr>>, output: OutMap, rule: Rid) -> Comprehension {
    let mut form = Comprehension {
        space,
        inputs,
        body: crate::jit::comprehension::Body { instrs, scalars: smallvec![], outs: smallvec![0] },
        combiner: add_combiner(),
        epilogue: Epilogue::None,
        output,
        props: Props::NONE,
        origin: Origin::Rewritten(RULES[rule as usize].id),
        ty: ElemTy::F32,
    };
    if !form.space.iter().any(Dim::is_reduce) {
        form.combiner = None;
    }
    form
}

// -- rule 7 --

/// The sums of one operand of the product `p` over its free dim: `Σ_m a(k, m)` (`lhs`) or `Σ_n b(n, k)`, as a vector over `k`.
fn operand_sum(p: &Comprehension, lhs: bool) -> Comprehension {
    let access = &p.inputs[usize::from(!lhs)];
    // The product's dims are `n` 0, `m` 1, `k` 2; the sum's are `k` 0 (parallel) and the free dim 1 (reduce).
    let renames: &[(usize, usize)] = if lhs { &[(2, 0), (1, 1)] } else { &[(2, 0), (0, 1)] };
    let free = if lhs { p.space[1].extent } else { p.space[0].extent };
    form_of(
        smallvec![Dim::parallel(p.space[2].extent), Dim::reduce(free)],
        smallvec![operand(access.src.clone(), access.shape, rename(&access.map, renames))],
        Bodies::x0(ElemTy::F32),
        OutMap::new([Some(0), None, None, None]),
        Rid::SumOfProduct,
    )
}

/// `sum(a·b)` and `sum_axis(a·b, j)` as sums of the operands and a vector product.
fn sum_of_product(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(root_id) = site else { return None };
    let root = scope.comp(root_id).filter(|comp| comp.role == ScopeRole::Root)?;
    let (form, product) = (&root.form, sole_producer(scope, root_id, 0)?);
    if form.inputs.len() != 1 || form.ty != ElemTy::F32 || !matches!(form.epilogue, Epilogue::None) || form.inputs[0].edge != Edge::None || form.inputs[0].indirect.is_some() {
        return None;
    }
    let p = &product.form;
    if !api_product(p) || matches!(p.origin, Origin::Api(OpTag::Matmul { tf32: Some(true) })) {
        return None;
    }
    let (n, m, k) = (p.space[0].extent, p.space[1].extent, p.space[2].extent);
    let ctx = context_of(scope)?;
    let (first, second) = (scope.next_id(), CompId(scope.next_id().0 + 1));
    let vector = || Some((placeholder(&ctx, k), [u32::try_from(k).ok()?, 1, 1, 1]));
    let feed = |access, producer| Feed { consumer: root_id, access, producer };
    let (replacement, add, feeds): (Comprehension, SmallVec<[Comprehension; 2]>, SmallVec<[Feed; 2]>) = match &form.origin {
        // The whole sum: `Σ_k (Σ_m a)(Σ_n b)`.
        Origin::Api(OpTag::Reduce(ops)) if ops.as_slice() == [ReduceOp::Sum] && form.space.len() == 1 => {
            let ((s1, sh1), (s2, sh2)) = (vector()?, vector()?);
            let root = form_of(
                smallvec![Dim::reduce(k)],
                smallvec![operand(s1, sh1, AffineMap::following([0])), operand(s2, sh2, AffineMap::following([0]))],
                Bodies::x0_times_x1(ElemTy::F32),
                OutMap::new([None; 4]),
                Rid::SumOfProduct,
            );
            (root, smallvec![operand_sum(p, true), operand_sum(p, false)], smallvec![feed(0, first), feed(1, second)])
        }
        // A sum along an axis of the product (`[N, M]`): axis 0 adds over `n` (`out(m) = Σ_k a(k, m)·(Σ_n b(n, k))`), axis 1 over `m`.
        Origin::Api(OpTag::ReduceAxis(ops)) if ops.as_slice() == [ReduceOp::Sum] && form.space.len() == 3 => {
            let (stride, lines) = (form.space[0].extent, form.space[2].extent);
            let (s, shape) = vector()?;
            let output = OutMap::new([Some(0), Some(1), None, None]);
            if stride == 1 && form.space[1].extent == n && lines == m {
                // out[0, m] over (a dummy dim of extent 1, m, k reduce): the dims are the product's own, so its maps are kept.
                let a = &p.inputs[0];
                let root = form_of(
                    smallvec![Dim::parallel(1), Dim::parallel(m), Dim::reduce(k)],
                    smallvec![operand(a.src.clone(), a.shape, *a.map), operand(s, shape, AffineMap::following([2]))],
                    Bodies::x0_times_x1(ElemTy::F32),
                    output,
                    Rid::SumOfProduct,
                );
                (root, smallvec![operand_sum(p, false)], smallvec![feed(1, first)])
            } else if lines == 1 && stride == n && form.space[1].extent == m {
                // out[n, 0] over (n, a dummy dim of extent 1, k reduce).
                let b = &p.inputs[1];
                let root = form_of(
                    smallvec![Dim::parallel(n), Dim::parallel(1), Dim::reduce(k)],
                    smallvec![operand(s, shape, AffineMap::following([2])), operand(b.src.clone(), b.shape, *b.map)],
                    Bodies::x0_times_x1(ElemTy::F32),
                    output,
                    Rid::SumOfProduct,
                );
                (root, smallvec![operand_sum(p, true)], smallvec![feed(0, first)])
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(Rewrite { replace: smallvec![(root_id, replacement)], add, drop: smallvec![product.id], feeds })
}

// -- rule 9 --

/// What a link of the chain multiplies: a matrix of `rows x cols` read as the access said (an operand that is not a product).
struct Leaf {
    src: Src,
    rows: u32,
    cols: u32,
}

/// The costs rule 9 proposes orders with; the engine's measure decides whether the proposal is taken (it knows the device).
const PROPOSAL: LinkCosts = LinkCosts { flops_per_ns: 5_000.0, bytes_per_ns: 500.0, launch_ns: 5_000.0 };

/// The chain `comp` is the product of, as the tree it is written in; `leaves` and `producers` are what it reads and what it consumes.
fn collect(scope: &Scope, comp: &SComp, leaves: &mut Vec<Leaf>, producers: &mut Vec<CompId>) -> Option<Tree> {
    let form = &comp.form;
    if !api_product(form) {
        return None;
    }
    let (m, n, k) = (u32::try_from(form.space[1].extent).ok()?, u32::try_from(form.space[0].extent).ok()?, u32::try_from(form.space[2].extent).ok()?);
    let mut parts: SmallVec<[Tree; 2]> = SmallVec::new();
    for (j, (rows, cols)) in [(m, k), (k, n)].into_iter().enumerate() {
        match sole_producer(scope, comp.id, j as u8) {
            Some(producer) => {
                producers.push(producer.id);
                parts.push(collect(scope, producer, leaves, producers)?);
            }
            None if scope.edges.iter().any(|edge| edge.consumer == comp.id && edge.access == j as u8) => return None,
            None => {
                let access = &form.inputs[j];
                leaves.push(Leaf { src: access.src.clone(), rows, cols });
                parts.push(Tree::Leaf(leaves.len() - 1));
            }
        }
    }
    let right = parts.pop()?;
    let left = parts.pop()?;
    Some(Tree::Join(Box::new(left), Box::new(right)))
}

/// What a built sub-chain is: a leaf of the chain, or a comprehension of the plan that makes a matrix of `rows x cols`.
enum Built {
    Leaf(usize),
    Made { id: CompId, rows: u32, cols: u32 },
}

struct Builder<'a> {
    ctx: &'a Context,
    leaves: &'a [Leaf],
    attrs: ContractionAttrs,
    next: u32,
    add: SmallVec<[Comprehension; 2]>,
    feeds: SmallVec<[Feed; 2]>,
}

impl Builder<'_> {
    fn dims(&self, built: &Built) -> (u32, u32) {
        match built {
            Built::Leaf(i) => (self.leaves[*i].rows, self.leaves[*i].cols),
            Built::Made { rows, cols, .. } => (*rows, *cols),
        }
    }

    fn src(&self, built: &Built) -> Src {
        match built {
            Built::Leaf(i) => self.leaves[*i].src.clone(),
            Built::Made { rows, cols, .. } => placeholder(self.ctx, u64::from(*rows) * u64::from(*cols)),
        }
    }

    /// The comprehension of `tree`: for the root the form that replaces it (`root` names its id), for the others a new one.
    fn build(&mut self, tree: &Tree, root: Option<CompId>) -> Option<(Built, Option<Comprehension>)> {
        let Tree::Join(l, r) = tree else { return match tree {
            Tree::Leaf(i) => Some((Built::Leaf(*i), None)),
            Tree::Join(..) => None,
        } };
        let left = self.build(l, None)?.0;
        let right = self.build(r, None)?.0;
        let ((m, k), (_, n)) = (self.dims(&left), self.dims(&right));
        let mut form = Comprehension::contract(self.src(&left), self.src(&right), Geometry { m, n, k }, self.attrs);
        rewritten(&mut form, Rid::ChainOrder);
        let id = root.unwrap_or(CompId(self.next + self.add.len() as u32));
        for (access, child) in [(0u8, &left), (1u8, &right)] {
            if let Built::Made { id: producer, .. } = child {
                self.feeds.push(Feed { consumer: id, access, producer: *producer });
            }
        }
        let built = Built::Made { id, rows: m, cols: n };
        if root.is_some() {
            return Some((built, Some(form)));
        }
        self.add.push(form);
        Some((built, None))
    }
}

/// A chain of products multiplied in the cheapest order.
fn chain_order(scope: &Scope, site: Site, _: &Numerics) -> Option<Rewrite> {
    let Site::Comp(root_id) = site else { return None };
    let root = scope.comp(root_id).filter(|comp| comp.role == ScopeRole::Root)?;
    let attrs = match root.form.origin {
        Origin::Api(OpTag::Matmul { tf32: None }) => Tf32Request::Default,
        Origin::Api(OpTag::Matmul { tf32: Some(false) }) => Tf32Request::Exact,
        _ => return None,
    };
    let (mut leaves, mut producers) = (Vec::new(), Vec::new());
    let current = collect(scope, root, &mut leaves, &mut producers)?;
    if leaves.len() < 3 || leaves.windows(2).any(|pair| pair[0].cols != pair[1].rows) {
        return None;
    }
    let dims: Vec<u64> = std::iter::once(u64::from(leaves[0].rows)).chain(leaves.iter().map(|leaf| u64::from(leaf.cols))).collect();
    let (order, _) = best(&dims, &PROPOSAL);
    if order == current {
        return None;
    }
    let ctx = context_of(scope)?;
    let mut builder = Builder { ctx: &ctx, leaves: &leaves, attrs: ContractionAttrs { tf32: attrs }, next: scope.next_id().0, add: SmallVec::new(), feeds: SmallVec::new() };
    let (_, form) = builder.build(&order, Some(root_id))?;
    Some(Rewrite { replace: smallvec![(root_id, form?)], add: builder.add, drop: producers.into_iter().collect(), feeds: builder.feeds })
}
