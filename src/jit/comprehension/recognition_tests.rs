//! Role inference, `classify` and look-through (W2-R1 §1-2).

use luma_core::{BackendKind, DeviceKind};
use smallvec::{SmallVec, smallvec};

use super::build::Bodies;
use super::inline::{Budget, inline_sources};
use super::map::{AffineMap, AffineRow, MapRef};
use super::roles::{DimMask, Roles, classify, infer_contraction_roles, infer_roles, separate};
use super::*;
use crate::array::Array;
use crate::builtin::{BinOp, ReduceList, ReduceOp};
use crate::context::{Context, ContextOptions};
use crate::jit::program::{Instr, Ref};
use crate::jit::structured::contraction::{ContractionAttrs, Geometry, Tf32Request};
use crate::jit::{Op, OpKind, Scalar};
use crate::reduce::AxisGeom;
use crate::view::PadMode;

fn cpu() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| tuning.reduce_multi_after = 0).unwrap()
}

fn data(ctx: &Context, len: usize, seed: usize) -> Array<f32> {
    shaped(ctx, len, seed)
}

/// A stored array of `shape` (small integers, so that sums are exact).
fn shaped(ctx: &Context, shape: impl Into<crate::shape::Shape> + Copy, seed: usize) -> Array<f32> {
    let shape: crate::shape::Shape = shape.into();
    let len = shape.len();
    let values: Vec<f32> = (0..len).map(|i| ((i * 7 + seed * 3) % 11) as f32 - 4.0).collect();
    Array::from_slice_on(ctx, &values, shape).unwrap()
}

fn tag(c: &Comprehension) -> &OpTag {
    match &c.origin {
        Origin::Api(tag) => tag,
        other => panic!("not an API form: {other:?}"),
    }
}

fn mask(dims: &[usize]) -> DimMask {
    dims.iter().fold(DimMask::EMPTY, |m, &d| m.union(DimMask::of(d)))
}

fn buffer(ctx: &Context, len: usize) -> Src {
    data(ctx, len, 0).src().ok().unwrap()
}

/// `matmul` of `[K, M]` by `[N, K]`.
fn matmul(ctx: &Context, m: u32, n: u32, k: u32) -> Comprehension {
    Comprehension::contract(buffer(ctx, (k * m) as usize), buffer(ctx, (n * k) as usize), Geometry { m, n, k }, ContractionAttrs { tf32: Tf32Request::Default })
}

fn unit(op: OpKind, ty: ElemTy, args: [Ref; 3]) -> Instr {
    Instr { op: Op::new(op, ty), args }
}

fn mul(a: Ref, b: Ref) -> Instr {
    unit(OpKind::Binary(BinOp::Mul), ElemTy::F32, [a, b, Ref::Unused])
}

/// A form over `dims` (extent, reduce) with accesses `(shape, dims the rows follow)`, the output following the parallel dims.
fn form(ctx: &Context, dims: &[(u64, bool)], accesses: &[&[usize]], instrs: Vec<Instr>, part: CombinePart) -> Comprehension {
    let space: SmallVec<[Dim; 4]> = dims.iter().map(|&(extent, reduce)| if reduce { Dim::reduce(extent) } else { Dim::parallel(extent) }).collect();
    let inputs: SmallVec<[Access; 2]> = accesses
        .iter()
        .map(|rows| {
            let mut shape = [1u32; 4];
            let mut map = AffineMap::ZERO;
            for (a, &d) in rows.iter().enumerate() {
                shape[a] = dims[d].0 as u32;
                map.rows[a] = AffineRow::dim(d);
            }
            let len: u32 = shape.iter().product();
            Access { src: buffer(ctx, len as usize), shape, ty: ElemTy::F32, map: MapRef::owned(map), edge: Edge::None, fill: None, indirect: None }
        })
        .collect();
    let parallel: Vec<usize> = (0..dims.len()).filter(|&d| !dims[d].1).collect();
    let mut axes = [None; 4];
    for (axis, &d) in parallel.iter().enumerate() {
        axes[axis] = Some(d as u8);
    }
    let last = instrs.len() as u32 - 1;
    let c = Comprehension {
        space,
        inputs,
        body: Body { instrs: std::sync::Arc::new(instrs), scalars: SmallVec::new(), outs: smallvec![last] },
        combiner: Some(Combiner::of(CombineOp::Single(part), ElemTy::F32)),
        epilogue: Epilogue::None,
        output: OutMap::new(axes),
        props: Props::NONE,
        origin: Origin::Rewritten(RuleId(0)),
        ty: ElemTy::F32,
    };
    c.validate().unwrap();
    c
}

#[test]
fn roles_of_a_product_are_n_m_k() {
    let ctx = cpu();
    let c = matmul(&ctx, 3, 4, 5);
    let roles = infer_roles(&c).unwrap();
    // The space is (n, m, k); lhs reads (k, m), rhs reads (n, k): m belongs to the lhs.
    assert_eq!((roles.m, roles.n, roles.k), (mask(&[1]), mask(&[0]), mask(&[2])));
    assert_eq!((roles.batch, roles.free, roles.lone), (DimMask::EMPTY, DimMask::EMPTY, DimMask::EMPTY));
    assert_eq!((roles.lhs, roles.rhs), (0, 1));
    assert_eq!(classify(&c), Form::Gemm);
}

#[test]
fn roles_of_a_batched_product() {
    let ctx = cpu();
    let both = Comprehension::contract_batched(buffer(&ctx, 30), buffer(&ctx, 40), [5, 3, 4, 2], true, true);
    let roles = infer_roles(&both).unwrap();
    assert_eq!((roles.batch, roles.m, roles.n, roles.k), (mask(&[2]), mask(&[1]), mask(&[0]), mask(&[3])));
    assert_eq!(classify(&both), Form::Gemm);
    // A shared rhs: the batch of the lhs is one more `m` dim (the rhs does not see it).
    let shared = Comprehension::contract_batched(buffer(&ctx, 30), buffer(&ctx, 20), [5, 3, 4, 2], true, false);
    let roles = infer_roles(&shared).unwrap();
    assert_eq!((roles.batch, roles.m, roles.n, roles.k), (DimMask::EMPTY, mask(&[1, 2]), mask(&[0]), mask(&[3])));
}

#[test]
fn degenerate_products_are_simpler_classes() {
    let ctx = cpu();
    for (m, n, k, expected) in [(3, 4, 5, Form::Gemm), (1, 4, 5, Form::Gemv), (3, 1, 5, Form::Gemv), (1, 1, 5, Form::Dot), (3, 4, 1, Form::Outer), (1, 1, 1, Form::Elementwise)] {
        let c = matmul(&ctx, m, n, k);
        assert_eq!(classify(&c), expected, "m={m} n={n} k={k}");
        assert!(tag(&c).expected_forms().contains(&expected));
    }
    // Every shape of the kernel tests is a class the tag expects.
    for m in [1, 2, 7, 33] {
        for n in [1, 3, 40] {
            for k in [1, 4, 17] {
                let c = matmul(&ctx, m, n, k);
                assert!(tag(&c).expected_forms().contains(&classify(&c)), "m={m} n={n} k={k}");
            }
        }
    }
}

#[test]
fn reductions_and_windows_are_not_contractions() {
    let ctx = cpu();
    let src = buffer(&ctx, 24);
    for c in [
        Comprehension::reduce(src.clone(), 24, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32),
        Comprehension::reduce_axis(src.clone(), AxisGeom { n: 4, stride: 3, lines: 6 }, ReduceOp::Max, ElemTy::F32),
        Comprehension::arg(src.clone(), 24, Some(AxisGeom { n: 4, stride: 3, lines: 6 }), true, ElemTy::F32),
    ] {
        assert_eq!(infer_roles(&c), None);
        assert_eq!(classify(&c), Form::Reduce);
        assert!(tag(&c).expected_forms().contains(&Form::Reduce));
    }
    let offsets = Array::from_slice_on(&ctx, &[0u32, 2, 6], 3).unwrap().src().ok().unwrap();
    let segments = Comprehension::segments(ReduceOp::Sum, buffer(&ctx, 6), 6, offsets, 2, ElemTy::F32);
    assert_eq!((infer_roles(&segments), classify(&segments)), (None, Form::Reduce));
    let conv = Comprehension::conv1d(buffer(&ctx, 20), buffer(&ctx, 5), 20, 5, 11, 2, 3).unwrap();
    assert_eq!((infer_roles(&conv), classify(&conv)), (None, Form::Window));
    // The dot of the product node is one access: a reduction.
    let dot = Comprehension::dot(buffer(&ctx, 12), 12, ElemTy::F32);
    assert_eq!(classify(&dot), Form::Reduce);
    assert!(tag(&dot).expected_forms().contains(&Form::Reduce));
}

#[test]
fn a_reduce_dim_of_one_side_is_not_a_contraction() {
    let ctx = cpu();
    // sum_k x(i, k) * y(i): the N-body shape, `k` is lone, and there is no m or n.
    let c = form(&ctx, &[(4, false), (5, true)], &[&[0, 1], &[0]], vec![mul(Ref::Input(0), Ref::Input(1))], CombinePart::Add);
    assert_eq!(infer_roles(&c), None);
    assert_eq!(classify(&c), Form::Reduce);
    // Both operands over (i, j): `i` is a batch dim, there is no m and n: a batched dot.
    let c = form(&ctx, &[(4, false), (5, true)], &[&[0, 1], &[0, 1]], vec![mul(Ref::Input(0), Ref::Input(1))], CombinePart::Add);
    let roles = infer_roles(&c).unwrap();
    assert_eq!((roles.batch, roles.m, roles.n, roles.k), (mask(&[0]), DimMask::EMPTY, DimMask::EMPTY, mask(&[1])));
    assert_eq!(classify(&c), Form::Dot);
}

#[test]
fn only_a_float_sum_is_the_roles_of_the_gemm_family() {
    let ctx = cpu();
    let dims = [(4, false), (3, false), (5, true)];
    let min_plus = form(&ctx, &dims, &[&[2, 1], &[0, 2]], vec![unit(OpKind::Binary(BinOp::Add), ElemTy::F32, [Ref::Input(0), Ref::Input(1), Ref::Unused])], CombinePart::Min);
    assert_eq!(infer_roles(&min_plus), None, "the GEMM family sums");
    assert!(infer_contraction_roles(&min_plus).is_some(), "the generic contraction takes any combiner");
    assert_eq!(classify(&min_plus), Form::Reduce, "identity maps and a combiner the reduction tail has");
    let mut ints = matmul(&ctx, 3, 4, 5);
    ints.ty = ElemTy::U32;
    assert_eq!(infer_roles(&ints), None);
}

#[test]
fn extent_one_dims_carry_no_role() {
    let ctx = cpu();
    // A reduce dim of extent 1 on the lhs only: dropped, so the form is an outer product.
    let c = form(&ctx, &[(4, false), (3, false), (1, true)], &[&[2, 1], &[0, 2]], vec![mul(Ref::Input(0), Ref::Input(1))], CombinePart::Add);
    assert_eq!(infer_roles(&c), None);
    assert_eq!(classify(&c), Form::Outer);
    // A free parallel dim no input uses.
    let c = form(&ctx, &[(4, false), (3, false), (5, true), (2, false)], &[&[2, 1], &[0, 2]], vec![mul(Ref::Input(0), Ref::Input(1))], CombinePart::Add);
    assert_eq!(infer_roles(&c).unwrap().free, mask(&[3]));
}

/// `c` with the dims in the order `perm` (new dim `i` is old dim `perm[i]`) and the inputs swapped when `swap`.
fn permuted(c: &Comprehension, perm: &[usize], swap: bool) -> Comprehension {
    let mut out = c.clone();
    out.space = perm.iter().map(|&d| c.space[d].clone()).collect();
    for access in &mut out.inputs {
        let mut map = *access.map;
        for row in &mut map.rows {
            let old = row.coeff;
            for (i, &d) in perm.iter().enumerate() {
                row.coeff[i] = old[d];
            }
        }
        access.map = MapRef::owned(map);
    }
    let inverse = |d: u8| perm.iter().position(|&p| p == usize::from(d)).unwrap() as u8;
    out.output.dim_of_axis = c.output.dim_of_axis.map(|d| d.map(inverse));
    if swap {
        out.inputs.swap(0, 1);
        let mut instrs = c.body.instrs.to_vec();
        for instr in &mut instrs {
            for arg in &mut instr.args {
                if let Ref::Input(j) = arg {
                    *j = 1 - *j;
                }
            }
        }
        out.body.instrs = std::sync::Arc::new(instrs);
    }
    out.validate().unwrap();
    out
}

#[test]
fn roles_do_not_depend_on_the_order_of_dims_or_inputs() {
    let ctx = cpu();
    // batch b, m, n, k.
    let base = form(&ctx, &[(2, false), (3, false), (4, false), (5, true)], &[&[3, 1, 0], &[2, 3, 0]], vec![mul(Ref::Input(0), Ref::Input(1))], CombinePart::Add);
    let roles = infer_roles(&base).unwrap();
    let perms: [[usize; 4]; 6] = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1], [0, 3, 1, 2], [3, 0, 2, 1]];
    for perm in perms {
        for swap in [false, true] {
            let c = permuted(&base, &perm, swap);
            let r = infer_roles(&c).unwrap();
            let back = |m: DimMask| m.dims().map(|d| perm[d]).collect::<Vec<_>>();
            let sorted = |m: DimMask| {
                let mut v: Vec<usize> = m.dims().collect();
                v.sort_unstable();
                v
            };
            assert_eq!(back(r.batch), sorted(roles.batch), "{perm:?} {swap}");
            assert_eq!(back(r.k), sorted(roles.k));
            // m and n exchange with the inputs: the side with its own dims is `lhs`, so as sets they agree up to the swap.
            let (rm, rn) = (back(r.m), back(r.n));
            let (bm, bn) = (sorted(roles.m), sorted(roles.n));
            assert!((rm == bm && rn == bn) || (rm == bn && rn == bm), "{perm:?} {swap}");
            assert_eq!(classify(&c), Form::Gemm);
        }
    }
}

#[test]
fn a_body_of_three_inputs_separates_in_two_operand_expressions() {
    let ctx = cpu();
    // (a(k, m) * s(k)) * b(n, k) over (n, m, k), and the same grouped the other way round: the scale goes to the side with fewer
    // inputs, and the roles are the product's.
    let dims = [(4, false), (3, false), (5, true)];
    let accesses: [&[usize]; 3] = [&[2, 1], &[2], &[0, 2]];
    let left = form(&ctx, &dims, &accesses, vec![mul(Ref::Input(0), Ref::Input(1)), mul(Ref::Tmp(0), Ref::Input(2))], CombinePart::Add);
    let right = form(&ctx, &dims, &accesses, vec![mul(Ref::Input(1), Ref::Input(2)), mul(Ref::Input(0), Ref::Tmp(0))], CombinePart::Add);
    for c in [&left, &right] {
        let separation = separate(c).unwrap();
        let roles = separation.roles;
        assert_eq!((roles.m, roles.n, roles.k), (mask(&[1]), mask(&[0]), mask(&[2])));
        let sizes = [separation.sides[0].len(), separation.sides[1].len()];
        assert_eq!(sizes[0] + sizes[1], 3);
        assert_eq!(infer_roles(c), Some(roles));
        assert_eq!(classify(c), Form::Gemm);
    }
    // Both orders put `s` with one operand: it is never alone on a side.
    for c in [&left, &right] {
        assert!(separate(c).unwrap().sides.iter().all(|side| !side.is_empty()));
    }
    // Not separable: `a * b + c`, or inputs that both factors read.
    let sum = form(&ctx, &dims, &accesses, vec![mul(Ref::Input(0), Ref::Input(2)), unit(OpKind::Binary(BinOp::Add), ElemTy::F32, [Ref::Tmp(0), Ref::Input(1), Ref::Unused])], CombinePart::Add);
    assert!(separate(&sum).is_none());
    let shared = form(&ctx, &dims, &accesses, vec![mul(Ref::Input(0), Ref::Input(1)), mul(Ref::Tmp(0), Ref::Input(0))], CombinePart::Add);
    assert!(separate(&shared).is_none());
}

// -- look-through --

/// The identity form over a source of shape `shape`: one dim per axis, the one `reduce` (if any) added.
fn identity(src: Src, shape: [u32; 4], reduce: Option<usize>) -> Comprehension {
    let space: SmallVec<[Dim; 4]> = (0..4).map(|d| if Some(d) == reduce { Dim::reduce(u64::from(shape[d])) } else { Dim::parallel(u64::from(shape[d])) }).collect();
    let mut map = AffineMap::ZERO;
    for (a, row) in map.rows.iter_mut().enumerate() {
        *row = AffineRow::dim(a);
    }
    let mut axes = [None; 4];
    for (slot, d) in (0..4).filter(|&d| Some(d) != reduce).enumerate() {
        axes[slot] = Some(d as u8);
    }
    Comprehension {
        space,
        inputs: smallvec![Access { src, shape, ty: ElemTy::F32, map: MapRef::owned(map), edge: Edge::None, fill: None, indirect: None }],
        body: Body { instrs: Bodies::x0(ElemTy::F32), scalars: SmallVec::new(), outs: smallvec![0] },
        combiner: reduce.map(|_| Combiner::of(CombineOp::Single(CombinePart::Add), ElemTy::F32)),
        epilogue: Epilogue::None,
        output: OutMap::new(axes),
        props: Props::NONE,
        origin: Origin::Rewritten(RuleId(0)),
        ty: ElemTy::F32,
    }
}

/// The values of the buffers the leaves of an inlined form read.
fn flat(leaves: &[(Src, Vec<f64>)], c: &Comprehension) -> Vec<Vec<f64>> {
    c.inputs
        .iter()
        .map(|access| {
            let found = leaves.iter().find(|(src, _)| match (src, &access.src) {
                (Src::Buffer(a), Src::Buffer(b)) => a.id() == b.id(),
                _ => false,
            });
            found.unwrap_or_else(|| panic!("an inlined form reads the buffers of the leaves: {:?}", access.src)).1.clone()
        })
        .collect()
}

fn values(a: &Array<f32>) -> Vec<f64> {
    a.to_vec().unwrap().into_iter().map(f64::from).collect()
}

fn leaf(a: &Array<f32>) -> (Src, Vec<f64>) {
    (a.src().ok().unwrap(), values(a))
}

/// Checks that inlining the node `x` (of shape `shape`) leaves its sum over `reduce` unchanged, and returns the inlined form.
fn round_trip(make: impl Fn() -> Array<f32>, shape: [u32; 4], reduce: usize, leaves: &[(Src, Vec<f64>)]) -> Comprehension {
    // Reading the values evaluates the node: a fresh one for each of the two.
    let x = &make();
    let expected_values = values(&make());
    let original = identity(x.src().ok().unwrap(), shape, Some(reduce));
    let inlined = inline_sources(&original, &mut Budget::default()).expect("within the budget");
    inlined.validate().unwrap();
    let expected = original.eval_reference(&[&expected_values]);
    let inputs = flat(leaves, &inlined);
    let refs: Vec<&[f64]> = inputs.iter().map(Vec::as_slice).collect();
    assert_eq!(inlined.eval_reference(&refs), expected);
    inlined
}

#[test]
fn views_compose_into_the_maps() {
    let ctx = cpu();
    let a = Array::<f32>::from_slice_on(&ctx, &(0..60).map(|i| (i % 9) as f32).collect::<Vec<_>>(), [5, 4, 3]).unwrap();
    let leaves = [leaf(&a)];
    type Make<'a> = Box<dyn Fn() -> Array<f32> + 'a>;
    let cases: Vec<(Make, [u32; 4])> = vec![
        (Box::new(|| a.flip(0)), [5, 4, 3, 1]),
        (Box::new(|| a.slice(1, 1..3)), [5, 2, 3, 1]),
        (Box::new(|| a.transpose()), [4, 5, 3, 1]),
        (Box::new(|| a.flip(0).slice(2, 1..)), [5, 4, 2, 1]),
        (Box::new(|| a.slice(0, 2..5).transpose().flip(1)), [4, 3, 3, 1]),
    ];
    for (make, shape) in &cases {
        for reduce in 0..3 {
            let inlined = round_trip(make, *shape, reduce, &leaves);
            assert_eq!(inlined.inputs.len(), 1);
            assert!(matches!(inlined.inputs[0].src, Src::Buffer(_)), "the view nodes are gone");
        }
    }
}

#[test]
fn a_broadcast_product_is_two_accesses_with_zero_coefficients() {
    let ctx = cpu();
    // The hand-written product-sum: a [K=4, M=3] and b [N=5, K=4] broadcast to [K, M, N].
    let a = shaped(&ctx, [4, 3, 1], 1);
    let bt = shaped(&ctx, [4, 1, 5], 2);
    let (a_buf, b_buf) = (a.clone(), bt.clone());
    let product = &a.broadcast_to([4, 3, 5]) * &bt.broadcast_to([4, 3, 5]);
    let leaves = [leaf(&a_buf), leaf(&b_buf)];
    let inlined = {
        let original = identity(product.src().ok().unwrap(), [4, 3, 5, 1], Some(0));
        let inlined = inline_sources(&original, &mut Budget::default()).unwrap();
        let inputs = flat(&leaves, &inlined);
        let refs: Vec<&[f64]> = inputs.iter().map(Vec::as_slice).collect();
        assert_eq!(inlined.eval_reference(&refs), original.eval_reference(&[&values(&product)]));
        inlined
    };
    // The broadcasts and the product dissolve into two accesses.
    assert_eq!(inlined.inputs.len(), 2);
    assert!(is_product(&inlined));
    let roles = infer_roles(&inlined).unwrap();
    assert_eq!((roles.m, roles.n, roles.k), (mask(&[1]), mask(&[2]), mask(&[0])));
    assert_eq!(classify(&inlined), Form::Gemm);
}

fn is_product(c: &Comprehension) -> bool {
    c.body.instrs.iter().any(|instr| instr.op.kind == OpKind::Binary(BinOp::Mul))
}

#[test]
fn elementwise_nodes_move_into_the_body() {
    let ctx = cpu();
    let (a, b) = (shaped(&ctx, [4, 3, 2], 1), shaped(&ctx, [4, 3, 2], 2));
    let leaves = [leaf(&a), leaf(&b)];
    let inlined = round_trip(|| (&(&a * &b.flip(0)) * 3.0f32).abs(), [4, 3, 2, 1], 1, &leaves);
    assert_eq!(inlined.inputs.len(), 2, "two buffers, merged where equal");
    assert_eq!(inlined.body.scalars.len(), 1, "the scalar operand became a body scalar");
    // The same buffer through the same map is one access.
    let inlined = round_trip(|| &a * &a, [4, 3, 2, 1], 2, &leaves);
    assert_eq!(inlined.inputs.len(), 1, "x * x loads x once");
    // `select` and comparisons.
    let inlined = round_trip(|| crate::ops::select(&a.gt(&b), &a, &b), [4, 3, 2, 1], 0, &leaves);
    assert_eq!(inlined.inputs.len(), 2);
}

#[test]
fn an_outer_edge_blocks_composition_and_a_fill_is_the_value_of_the_node() {
    let ctx = cpu();
    let a = shaped(&ctx, [4, 3], 3);
    let padded = || a.pad([(1, 1), (0, 0)], PadMode::Zero);
    let leaves = [leaf(&a)];
    // A padded view under a plain access: composed, the edge is the view's.
    let inlined = round_trip(padded, [6, 3, 1, 1], 0, &leaves);
    assert_eq!(inlined.inputs[0].edge, Edge::Fill);
    assert_eq!(inlined.inputs[0].fill.map(|fill| fill.bits), Some(0));
    // The same form read through an outer edge of its own stays one opaque access to the node.
    let mut outer = identity(padded().src().ok().unwrap(), [6, 3, 1, 1], Some(0));
    outer.inputs[0].edge = Edge::Clamp;
    let kept = inline_sources(&outer, &mut Budget::default()).unwrap();
    assert!(matches!(kept.inputs[0].src, Src::Node(_)));
    // An elementwise node under a fill edge is not moved either: its fill is the value of the node, not f(fill).
    let doubled = &a * 2.0f32;
    let mut filled = identity(doubled.src().ok().unwrap(), [4, 3, 1, 1], Some(0));
    filled.inputs[0].edge = Edge::Fill;
    filled.inputs[0].fill = Some(Scalar::f32(1.0));
    let kept = inline_sources(&filled, &mut Budget::default()).unwrap();
    assert!(matches!(kept.inputs[0].src, Src::Node(_)));
}

#[test]
fn the_budget_bounds_the_work() {
    let ctx = cpu();
    let a = shaped(&ctx, [4, 3], 1);
    let mut x = a.clone();
    for _ in 0..4 {
        x = &x + 1.0f32;
    }
    let original = identity(x.src().ok().unwrap(), [4, 3, 1, 1], Some(0));
    assert!(inline_sources(&original, &mut Budget { nodes: 2, inputs: 4 }).is_none());
    assert!(inline_sources(&original, &mut Budget::default()).is_some());
    // Too many inputs: a * b over four distinct leaves.
    let leaves: Vec<Array<f32>> = (0..5).map(|s| shaped(&ctx, [4, 3], s)).collect();
    let big = &(&(&leaves[0] * &leaves[1]) * &(&leaves[2] * &leaves[3])) * &leaves[4];
    let original = identity(big.src().ok().unwrap(), [4, 3, 1, 1], Some(0));
    assert!(inline_sources(&original, &mut Budget::default()).is_none());
}

#[test]
fn the_original_form_and_its_nodes_are_untouched() {
    let ctx = cpu();
    let a = shaped(&ctx, [4, 3], 1);
    let x = &a.flip(0) * 2.0f32;
    let original = identity(x.src().ok().unwrap(), [4, 3, 1, 1], Some(1));
    let hash = original.structure_hash();
    let _ = inline_sources(&original, &mut Budget::default()).unwrap();
    assert_eq!(original.structure_hash(), hash);
    assert!(matches!(original.inputs[0].src, Src::Node(_)));
    assert_eq!(x.to_vec().unwrap().len(), 12);
    let _: Roles = infer_roles(&matmul(&ctx, 3, 4, 5)).unwrap();
}
