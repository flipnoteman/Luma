//! The form: affine maps against views, validation, structure hashes, and the form of each API operation (the form-per-op tests of
//! W2-R0: they live here because the form is crate-private). The reductions' forms are read off the values `Array::sum` and friends
//! make; the other operations' forms off the pure builders the eager families' wrappers call.

use luma_core::{BackendKind, DeviceKind};

use super::build::{Bodies, WindowGeom};
use super::map::AffineRow;
use super::*;
use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::image::{Border, ConvSpec};
use crate::jit::structured::contraction::{ContractionAttrs, Geometry, Tf32Request};
use crate::builtin::{BinOp, UnOp};
use crate::jit::{AxisMap, OpKind, ViewMap};
use crate::reduce::{AxisGeom, Reduction};

fn cpu() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| tuning.reduce_multi_after = 0).unwrap()
}

fn array(ctx: &Context, len: usize) -> Array<f32> {
    let data: Vec<f32> = (0..len).map(|i| (i % 7) as f32).collect();
    Array::from_slice_on(ctx, &data, len).unwrap()
}

fn src(array: &Array<f32>) -> Src {
    array.src().ok().unwrap()
}

fn extents(form: &Comprehension) -> Vec<(u64, &'static str)> {
    form.space
        .iter()
        .map(|dim| {
            (
                dim.extent,
                match dim.kind {
                    DimKind::Parallel => "P",
                    DimKind::Reduce => "R",
                    DimKind::Segmented { .. } => "S",
                },
            )
        })
        .collect()
}

fn rows(access: &Access) -> Vec<(Vec<i32>, i64)> {
    access.map.rows.iter().map(|row| (row.coeff.to_vec(), row.offset)).collect()
}

/// The coefficients of the first `dims` dims of row `a` of access `j`.
fn coeff(form: &Comprehension, j: usize, a: usize) -> Vec<i32> {
    form.inputs[j].map.rows[a].coeff[..form.space.len()].to_vec()
}

fn parts(form: &Comprehension) -> Vec<CombinePart> {
    form.combiner.as_ref().unwrap().op.parts().to_vec()
}

// -- maps --

/// Views of a four-axis array: the identity, a broadcast, a reversed and a sliced axis, a permutation, a diagonal.
fn views() -> Vec<ViewMap> {
    let ax = |from: u8, start: i32, step: i32| AxisMap { from, start, step };
    let identity = ViewMap::identity([4, 5, 6, 7]);
    let broadcast = ViewMap { src: [4, 1, 6, 1], out: [4, 5, 6, 7], axes: [ax(0, 0, 1), ax(1, 0, 0), ax(2, 0, 1), ax(3, 0, 0)] };
    let reversed = ViewMap { src: [4, 5, 6, 7], out: [4, 5, 6, 7], axes: [ax(0, 3, -1), ax(1, 0, 1), ax(2, 5, -1), ax(3, 0, 1)] };
    let sliced = ViewMap { src: [9, 5, 12, 7], out: [4, 5, 5, 7], axes: [ax(0, 1, 2), ax(1, 0, 1), ax(2, 2, 2), ax(3, 0, 1)] };
    let permuted = ViewMap { src: [4, 5, 6, 7], out: [5, 4, 7, 6], axes: [ax(1, 0, 1), ax(0, 0, 1), ax(3, 0, 1), ax(2, 0, 1)] };
    // A diagonal: two source rows follow one output axis.
    let diagonal = ViewMap { src: [4, 4, 6, 7], out: [4, 1, 6, 7], axes: [ax(0, 0, 1), ax(0, 0, 1), ax(2, 0, 1), ax(3, 0, 1)] };
    vec![identity, broadcast, reversed, sliced, permuted, diagonal]
}

#[test]
fn views_round_trip_through_affine_maps() {
    for view in views() {
        let map = AffineMap::from_view(&view, [0, 1, 2, 3]);
        let back = map.to_view([Some(0), Some(1), Some(2), Some(3)], view.src, view.out);
        assert_eq!(back, Some(view), "{view:?}");
    }
}

#[test]
fn a_map_that_is_not_a_view_has_none() {
    let axes = [Some(0), Some(1), Some(2), Some(3)];
    // A window reads `2 * o + t`: two coefficients in one row.
    let mut window = AffineMap::ZERO;
    window.rows[0].coeff[0] = 2;
    window.rows[0].coeff[4] = 1;
    assert_eq!(window.to_view([Some(0), Some(1), Some(2), Some(4)], [9, 1, 1, 1], [4, 1, 1, 1]), None);
    // A coefficient on a dim no output axis stands for.
    let mut stray = AffineMap::ZERO;
    stray.rows[0].coeff[5] = 1;
    assert_eq!(stray.to_view(axes, [4, 1, 1, 1], [4, 1, 1, 1]), None);
    // An offset that does not fit `i32`.
    let mut far = AffineMap::ZERO;
    far.rows[0].offset = i64::from(i32::MAX) + 1;
    assert_eq!(far.to_view(axes, [4, 1, 1, 1], [4, 1, 1, 1]), None);
}

#[test]
fn composition_equals_view_composition() {
    let ax = |from: u8, start: i32, step: i32| AxisMap { from, start, step };
    // `inner` reads a [10, 6, 3, 2] array as [8, 3, 3, 2]; `outer` reads that as [4, 3, 3, 2] (every second element, reversed on axis 1).
    let inner = ViewMap { src: [10, 6, 3, 2], out: [8, 3, 3, 2], axes: [ax(0, 1, 1), ax(1, 1, 2), ax(2, 0, 1), ax(3, 0, 1)] };
    let outer = ViewMap { src: [8, 3, 3, 2], out: [4, 3, 3, 2], axes: [ax(0, 0, 2), ax(1, 2, -1), ax(2, 0, 1), ax(3, 0, 1)] };
    let identity = [0, 1, 2, 3];
    let composed = AffineMap::from_view(&inner, identity).then(&AffineMap::from_view(&outer, identity)).unwrap();
    assert_eq!(composed, AffineMap::from_view(&inner.then(&outer), identity));
    // A dim past the rows of `inner` has no source row.
    let mut wide = AffineMap::ZERO;
    wide.rows[0].coeff[4] = 1;
    assert_eq!(wide.then(&composed), None);
}

#[test]
fn strides_match_a_brute_force_flat_index() {
    let shape = [4u32, 5, 3, 2];
    let mut map = AffineMap::ZERO;
    // Dims: 0..3 with extents 2, 3, 2.
    map.rows[0] = AffineRow { coeff: [2, 0, 1, 0, 0, 0], offset: 1 };
    map.rows[1] = AffineRow { coeff: [0, 1, 0, 0, 0, 0], offset: 2 };
    map.rows[2] = AffineRow { coeff: [0, 0, -1, 0, 0, 0], offset: 2 };
    map.rows[3] = AffineRow { coeff: [0, 0, 0, 0, 0, 0], offset: 1 };
    let (strides, base) = map.strides(shape);
    for p in [[0i64, 0, 0], [1, 2, 1], [0, 1, 1], [1, 0, 0]] {
        let coordinate = |a: usize| map.rows[a].offset + (0..3).map(|d| i64::from(map.rows[a].coeff[d]) * p[d]).sum::<i64>();
        let flat = coordinate(0) + i64::from(shape[0]) * (coordinate(1) + i64::from(shape[1]) * (coordinate(2) + i64::from(shape[2]) * coordinate(3)));
        assert_eq!(base + (0..3).map(|d| strides[d] * p[d]).sum::<i64>(), flat, "{p:?}");
    }
}

// -- validation --

fn valid(ctx: &Context) -> Comprehension {
    let x = array(ctx, 12);
    Comprehension::reduce(src(&x), 12, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32)
}

fn lines(ctx: &Context) -> Comprehension {
    let x = array(ctx, 24);
    Comprehension::reduce_axis(src(&x), AxisGeom { n: 4, stride: 3, lines: 6 }, ReduceOp::Max, ElemTy::F32)
}

#[test]
fn validate_rejects_each_malformed_form() {
    let ctx = cpu();
    assert_eq!(valid(&ctx).validate(), Ok(()));
    assert_eq!(lines(&ctx).validate(), Ok(()));

    let mut form = valid(&ctx);
    form.combiner = None;
    assert_eq!(form.validate(), Err(FormError::Combiner), "a reduce dim without a combiner");

    let mut form = valid(&ctx);
    form.space[0].kind = DimKind::Parallel;
    assert_eq!(form.validate(), Err(FormError::Combiner), "a combiner without a reduce dim");

    let mut form = valid(&ctx);
    form.body.outs.push(0);
    assert!(matches!(form.validate(), Err(FormError::Body(_))), "an out per part");

    let mut form = valid(&ctx);
    form.body.outs[0] = 9;
    assert!(matches!(form.validate(), Err(FormError::Body(_))), "an out that is not an instruction");

    let mut form = valid(&ctx);
    let mut bad = (*form.body.instrs).clone();
    bad[0].args[0] = Ref::Input(3);
    form.body.instrs = Arc::new(bad);
    assert!(matches!(form.validate(), Err(FormError::Body(_))), "an input that is not an access");

    let mut form = lines(&ctx);
    form.output.dim_of_axis[0] = None;
    assert_eq!(form.validate(), Err(FormError::Output), "a parallel dim no output axis follows");

    let mut form = lines(&ctx);
    form.output.dim_of_axis[1] = Some(1);
    assert_eq!(form.validate(), Err(FormError::Output), "an output axis that follows the reduce dim");

    let mut form = valid(&ctx);
    let mut map = *form.inputs[0].map;
    map.rows[0].coeff[3] = 1;
    form.inputs[0].map = MapRef::owned(map);
    assert_eq!(form.validate(), Err(FormError::MapDim { input: 0, row: 0 }), "a coefficient on a dim the space does not have");

    let mut form = valid(&ctx);
    let mut map = *form.inputs[0].map;
    map.rows[0].offset = 1;
    form.inputs[0].map = MapRef::owned(map);
    assert_eq!(form.validate(), Err(FormError::OutOfRange { input: 0, row: 0 }), "an Edge::None map that leaves the shape at the extremes");

    let mut form = valid(&ctx);
    form.inputs[0].edge = Edge::Fill;
    assert_eq!(form.validate(), Err(FormError::Fill(0)), "a Fill edge without a fill");

    let mut form = valid(&ctx);
    form.inputs[0].shape = [11, 1, 1, 1];
    assert_eq!(form.validate(), Err(FormError::ShapeLength(0)), "a shape that is not the source's length");

    let x = array(&ctx, 6);
    let mut form = Comprehension::segments(ReduceOp::Sum, src(&x), 6, src(&Array::from_slice_on(&ctx, &[0.0f32, 2.0, 6.0], 3).unwrap()), 2, ElemTy::F32);
    assert_eq!(form.validate(), Ok(()));
    form.space[1].kind = DimKind::Segmented { offsets: Box::new(Src::unused()), selector: 1 };
    assert_eq!(form.validate(), Err(FormError::Selector), "a selector that is not a parallel dim");
}

#[test]
fn structure_hash_ignores_buffers_and_values() {
    let ctx = cpu();
    let (a, b) = (array(&ctx, 12), array(&ctx, 12));
    let sum = |x: &Array<f32>| Comprehension::reduce(src(x), 12, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32);
    assert_eq!(sum(&a).structure_hash(), sum(&b).structure_hash(), "two buffers of one structure");
    let max = Comprehension::reduce(src(&a), 12, ReduceList::new(&[ReduceOp::Max]), ElemTy::F32);
    assert_ne!(sum(&a).structure_hash(), max.structure_hash(), "another combiner");
    let longer = array(&ctx, 13);
    assert_ne!(sum(&a).structure_hash(), Comprehension::reduce(src(&longer), 13, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32).structure_hash(), "another extent");
    // The value of a fill is not structure, that there is one is.
    let x = array(&ctx, 8);
    let w = array(&ctx, 3);
    let conv = |fill: f32| {
        let mut form = Comprehension::conv1d(src(&x), src(&w), 8, 3, 8, 1, 1).unwrap();
        form.inputs[0].fill = Some(Scalar::f32(fill));
        form
    };
    assert_eq!(conv(0.0).structure_hash(), conv(5.0).structure_hash());
    let mut plain = conv(0.0);
    plain.inputs[0].edge = Edge::None;
    plain.inputs[0].fill = None;
    assert_ne!(conv(0.0).structure_hash(), plain.structure_hash());
    assert_ne!(sum(&a).structure_hash(), Comprehension::dot(src(&a), 12, ElemTy::F32).structure_hash(), "the origin tag");
}

#[test]
fn props_meet_and_join_are_the_lattice_of_the_facts() {
    let diagonal = Props { flags: Props::DIAGONAL | Props::SYMMETRIC | Props::LOWER | Props::UPPER, band: Some((0, 0)), block: Some(4), rank: None };
    let lower = Props { flags: Props::LOWER, band: Some((7, 0)), block: Some(6), rank: Some(3) };
    let met = diagonal.meet(lower);
    assert_eq!(met, Props { flags: Props::LOWER, band: Some((7, 0)), block: Some(12), rank: None }, "what holds for either");
    let joined = diagonal.join(lower);
    assert_eq!(joined.flags, diagonal.flags);
    assert_eq!((joined.band, joined.block, joined.rank), (Some((0, 0)), Some(2), Some(3)), "what both say");
    assert_eq!(Props::NONE.meet(diagonal), Props::NONE);
    assert_eq!(Props::NONE.join(diagonal), diagonal);
}

#[test]
fn algebra_switch_parses() {
    assert_eq!(Algebra::parse("off"), Some(Algebra::Off));
    assert_eq!(Algebra::parse(" API "), Some(Algebra::Api));
    assert_eq!(Algebra::parse("t1"), Some(Algebra::T1));
    assert_eq!(Algebra::parse("on"), Some(Algebra::On));
    assert!(Algebra::Off < Algebra::Api && Algebra::Api < Algebra::T1 && Algebra::T1 < Algebra::On);
    assert_eq!(Algebra::parse("maybe"), None);
}

// -- the form of each operation --

fn is_unary_id(instr: &Instr) -> bool {
    instr.op.kind == OpKind::Unary(UnOp::Id)
}

fn is_mul(instr: &Instr) -> bool {
    instr.op.kind == OpKind::Binary(BinOp::Mul)
}

#[test]
fn reductions_are_a_reduce_dim_over_the_operand() {
    let ctx = cpu();
    let a = array(&ctx, 12);
    for (op, part) in [(ReduceOp::Sum, CombinePart::Add), (ReduceOp::Min, CombinePart::Min), (ReduceOp::Max, CombinePart::Max), (ReduceOp::Prod, CombinePart::Mul), (ReduceOp::SumSquares, CombinePart::Add)] {
        let ops = ReduceList::new(&[op]);
        let form = Comprehension::reduce(src(&a), 12, ops, ElemTy::F32);
        assert_eq!(extents(&form), [(12, "R")]);
        assert_eq!(form.inputs.len(), 1);
        assert_eq!(form.inputs[0].shape, [12, 1, 1, 1]);
        assert_eq!(coeff(&form, 0, 0), [1]);
        assert_eq!(parts(&form), [part]);
        assert_eq!(form.output.dim_of_axis, [None; 4]);
        assert_eq!(form.origin, Origin::Api(OpTag::Reduce(ops)));
        assert_eq!(form.ty, ElemTy::F32);
        assert_eq!(form.body.outs.as_slice(), [0]);
        assert_eq!(is_mul(&form.body.instrs[0]), op == ReduceOp::SumSquares, "{op:?}: the body of a sum of squares is x0 * x0, the others x0");
        assert!(matches!(form.epilogue, Epilogue::None));
        form.validate().unwrap();
    }
}

#[test]
fn several_ops_are_one_out_per_op() {
    let ctx = cpu();
    let a = array(&ctx, 12);
    let ops = ReduceList::new(&[ReduceOp::Min, ReduceOp::SumSquares, ReduceOp::Max]);
    let form = Comprehension::reduce(src(&a), 12, ops, ElemTy::F32);
    assert_eq!(parts(&form), [CombinePart::Min, CombinePart::Add, CombinePart::Max]);
    assert!(matches!(form.combiner.as_ref().unwrap().op, CombineOp::Tuple(_)));
    assert_eq!(form.body.outs.as_slice(), [0, 1, 0], "the sum of squares reads x0 * x0, the others x0");
    assert!(is_unary_id(&form.body.instrs[0]) && is_mul(&form.body.instrs[1]));
    assert_eq!(form.combiner.as_ref().unwrap().identity.iter().map(|s| f32::from_bits(s.bits as u32)).collect::<Vec<_>>(), [f32::INFINITY, 0.0, f32::NEG_INFINITY]);
    form.validate().unwrap();
    // Bodies are interned: the same instructions are one allocation.
    let again = Comprehension::reduce(src(&a), 12, ReduceList::new(&[ReduceOp::Sum]), ElemTy::F32);
    assert!(Arc::ptr_eq(&again.body.instrs, &Bodies::x0(ElemTy::F32)));
    assert!(!Arc::ptr_eq(&again.body.instrs, &Bodies::x0(ElemTy::U32)), "an instruction carries its type");
}

#[test]
fn identities_and_exactness_follow_the_type() {
    let combiner = |op: ReduceOp, ty: ElemTy| Combiner::of_ops(&[op], ty);
    assert!(combiner(ReduceOp::Min, ElemTy::F32).reorder_exact);
    assert!(combiner(ReduceOp::Sum, ElemTy::I32).reorder_exact);
    assert!(!combiner(ReduceOp::Sum, ElemTy::F32).reorder_exact);
    assert!(!combiner(ReduceOp::Prod, ElemTy::Df32).reorder_exact);
    assert_eq!(combiner(ReduceOp::Min, ElemTy::I32).identity[0].bits, i32::MAX as u32 as u64);
    assert_eq!(combiner(ReduceOp::Max, ElemTy::I8).identity[0].bits, (-128i32) as u32 as u64, "a small signed register is sign-extended");
    assert_eq!(combiner(ReduceOp::Max, ElemTy::U32).identity[0].bits, 0);
    assert_eq!(combiner(ReduceOp::Prod, ElemTy::F32).identity[0].bits, u64::from(1f32.to_bits()));
    assert_eq!(combiner(ReduceOp::Min, ElemTy::F64).identity[0].bits, f64::INFINITY.to_bits());
    assert_eq!(combiner(ReduceOp::Min, ElemTy::Bool).identity[0].bits, 1);
}

#[test]
fn axis_reductions_are_two_parallel_dims_and_a_reduce_dim() {
    let ctx = cpu();
    let a = array(&ctx, 24);
    let geom = AxisGeom { n: 4, stride: 3, lines: 6 };
    let form = Comprehension::reduce_axis(src(&a), geom, ReduceOp::Max, ElemTy::F32);
    assert_eq!(extents(&form), [(3, "P"), (4, "R"), (2, "P")], "s, k, o");
    assert_eq!(form.inputs[0].shape, [3, 4, 2, 1]);
    assert_eq!((coeff(&form, 0, 0), coeff(&form, 0, 1), coeff(&form, 0, 2)), (vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]));
    assert_eq!(form.output.dim_of_axis, [Some(0), Some(2), None, None], "line l = s + stride * o");
    assert_eq!(parts(&form), [CombinePart::Max]);
    assert_eq!(form.origin, Origin::Api(OpTag::ReduceAxis(ReduceList::new(&[ReduceOp::Max]))));
    form.validate().unwrap();
}

#[test]
fn arg_reductions_return_the_index() {
    let ctx = cpu();
    let a = array(&ctx, 24);
    let whole = Comprehension::arg(src(&a), 24, None, true, ElemTy::F32);
    assert_eq!(extents(&whole), [(24, "R")]);
    assert_eq!((parts(&whole), whole.ty), (vec![CombinePart::ArgMax], ElemTy::U32));
    assert_eq!(whole.inputs[0].ty, ElemTy::F32);
    assert_eq!(whole.origin, Origin::Api(OpTag::Arg { max: true, axis: false }));
    whole.validate().unwrap();
    let axis = Comprehension::arg(src(&a), 24, Some(AxisGeom { n: 4, stride: 3, lines: 6 }), false, ElemTy::F32);
    assert_eq!(extents(&axis), [(3, "P"), (4, "R"), (2, "P")]);
    assert_eq!(parts(&axis), [CombinePart::ArgMin]);
    assert_eq!(axis.origin, Origin::Api(OpTag::Arg { max: false, axis: true }));
    axis.validate().unwrap();
}

#[test]
fn segments_are_a_segmented_reduce_dim_selected_by_the_rows() {
    let ctx = cpu();
    let (x, offsets) = (array(&ctx, 6), Array::from_slice_on(&ctx, &[0u32, 2, 6], 3).unwrap());
    let form = Comprehension::segments(ReduceOp::Sum, src(&x), 6, offsets.src().ok().unwrap(), 2, ElemTy::F32);
    assert_eq!(extents(&form), [(2, "P"), (6, "S")]);
    assert!(matches!(form.space[1].kind, DimKind::Segmented { selector: 0, .. }));
    assert_eq!(coeff(&form, 0, 0), [0, 1], "x[j]");
    assert_eq!(form.output.dim_of_axis, [Some(0), None, None, None]);
    assert_eq!(form.origin, Origin::Api(OpTag::Segments(ReduceOp::Sum)));
    form.validate().unwrap();
}

#[test]
fn dot_is_a_reduce_dim_over_one_access_of_the_product() {
    let ctx = cpu();
    let product = &array(&ctx, 12) * &array(&ctx, 12);
    let form = Comprehension::dot(src(&product), 12, ElemTy::F32);
    assert_eq!((extents(&form), form.inputs.len()), (vec![(12, "R")], 1));
    assert_eq!(parts(&form), [CombinePart::Add]);
    assert_eq!(form.origin, Origin::Api(OpTag::Dot));
    form.validate().unwrap();
}

#[test]
fn a_product_is_two_parallel_dims_and_a_reduce_dim() {
    let ctx = cpu();
    let (lhs, rhs) = (array(&ctx, 15), array(&ctx, 20));
    // lhs [K=5, M=3], rhs [N=4, K=5].
    let geom = Geometry { m: 3, n: 4, k: 5 };
    for (request, tf32) in [(Tf32Request::Default, None), (Tf32Request::Exact, Some(false)), (Tf32Request::Tf32, Some(true))] {
        let form = Comprehension::contract(src(&lhs), src(&rhs), geom, ContractionAttrs { tf32: request, half: false });
        assert_eq!(extents(&form), [(4, "P"), (3, "P"), (5, "R")], "n, m, k");
        assert_eq!(form.inputs[0].shape, [5, 3, 1, 1]);
        assert_eq!(form.inputs[1].shape, [4, 5, 1, 1]);
        assert_eq!((coeff(&form, 0, 0), coeff(&form, 0, 1)), (vec![0, 0, 1], vec![0, 1, 0]), "lhs: rows k, m");
        assert_eq!((coeff(&form, 1, 0), coeff(&form, 1, 1)), (vec![1, 0, 0], vec![0, 0, 1]), "rhs: rows n, k");
        assert!(is_mul(&form.body.instrs[0]));
        assert_eq!(parts(&form), [CombinePart::Add]);
        assert_eq!(form.output.dim_of_axis, [Some(0), Some(1), None, None], "[N, M]");
        assert_eq!(form.origin, Origin::Api(OpTag::Matmul { tf32 }));
        form.validate().unwrap();
    }
}

#[test]
fn a_transposed_operand_has_the_same_maps() {
    // matmul_nt hands the product `rhs.transpose()`, a view node: the form reads it as `[N, K]` like any operand.
    let ctx = cpu();
    let lhs = array(&ctx, 15);
    let rhs = Array::<f32>::from_slice_on(&ctx, &(0..20).map(|i| i as f32).collect::<Vec<_>>(), [5, 4]).unwrap().transpose();
    let form = Comprehension::contract(src(&lhs), src(&rhs), Geometry { m: 3, n: 4, k: 5 }, ContractionAttrs { tf32: Tf32Request::Default, half: false });
    let straight = Comprehension::contract(src(&lhs), src(&array(&ctx, 20)), Geometry { m: 3, n: 4, k: 5 }, ContractionAttrs { tf32: Tf32Request::Default, half: false });
    assert_eq!(form.structure_hash(), straight.structure_hash());
    form.validate().unwrap();
}

#[test]
fn a_batched_product_shares_an_operand_with_a_zero_batch_row() {
    let ctx = cpu();
    // [K=5, M=3, B=2] by a shared [N=4, K=5].
    let (lhs, rhs) = (array(&ctx, 30), array(&ctx, 20));
    let form = Comprehension::contract_batched(src(&lhs), src(&rhs), [5, 3, 4, 2], true, false);
    assert_eq!(extents(&form), [(4, "P"), (3, "P"), (2, "P"), (5, "R")], "n, m, b, k");
    assert_eq!(form.inputs[0].shape, [5, 3, 2, 1]);
    assert_eq!(form.inputs[1].shape, [4, 5, 1, 1]);
    assert_eq!(coeff(&form, 0, 2), [0, 0, 1, 0], "batched lhs follows b");
    assert_eq!(coeff(&form, 1, 2), [0, 0, 0, 0], "a shared rhs has a zero batch row");
    assert_eq!(form.output.dim_of_axis, [Some(0), Some(1), Some(2), None]);
    assert_eq!(form.origin, Origin::Api(OpTag::MatmulBatched));
    form.validate().unwrap();
}

#[test]
fn conv1d_reads_two_coefficients_and_zero_outside() {
    let ctx = cpu();
    let (x, w) = (array(&ctx, 20), array(&ctx, 5));
    // stride 2, padding 3: (20 - 5 + 6) / 2 + 1 = 11 outputs.
    let form = Comprehension::conv1d(src(&x), src(&w), 20, 5, 11, 2, 3).unwrap();
    assert_eq!(extents(&form), [(11, "P"), (5, "R")]);
    assert_eq!(coeff(&form, 0, 0), [2, 1], "stride * o + t");
    assert_eq!(form.inputs[0].map.rows[0].offset, -3);
    assert_eq!((form.inputs[0].edge, form.inputs[0].fill.map(|fill| fill.bits)), (Edge::Fill, Some(0)));
    assert_eq!(coeff(&form, 1, 0), [0, 1]);
    assert!(is_mul(&form.body.instrs[0]));
    assert_eq!(parts(&form), [CombinePart::Add]);
    assert_eq!(form.origin, Origin::Api(OpTag::Conv1d));
    form.validate().unwrap();
    assert!(form.inputs[0].map.to_view([Some(0), None, None, None], [20, 1, 1, 1], [11, 1, 1, 1]).is_none(), "a window is not a view");
}

/// A 2-d window geometry of a `[w, h, batch]` source.
fn geom2(w: u32, h: u32, batch: u32, k: [u32; 2], stride: [u32; 2], before: [u32; 2], out: [u32; 2]) -> WindowGeom {
    WindowGeom { src: [w, h, batch, 1], k: [k[0], k[1], 1], stride: [stride[0], stride[1], 1], before: [before[0], before[1], 0], out: [out[0], out[1], 1], three_d: false }
}

#[test]
fn conv2_is_output_dims_a_batch_and_tap_dims() {
    let ctx = cpu();
    let (x, w) = (array(&ctx, 8 * 6 * 2), array(&ctx, 9));
    // Same padding, stride 1 on axis 0 and 2 on axis 1: out 8 x 3.
    let geom = geom2(8, 6, 2, [3, 3], [1, 2], [1, 1], [8, 3]);
    let form = Comprehension::conv(ConvSpec::new(), &geom, Border::Zero, src(&x), src(&w)).unwrap();
    assert_eq!(extents(&form), [(8, "P"), (3, "P"), (2, "P"), (3, "R"), (3, "R")], "o0, o1, batch, t0, t1");
    assert_eq!(form.inputs[0].shape, [8, 6, 2, 1]);
    assert_eq!(coeff(&form, 0, 0), [1, 0, 0, 1, 0], "o0 + t0 - 1");
    assert_eq!(coeff(&form, 0, 1), [0, 2, 0, 0, 1], "2 * o1 + t1 - 1");
    assert_eq!(coeff(&form, 0, 2), [0, 0, 1, 0, 0], "the batch");
    assert_eq!(&form.inputs[0].map.rows.iter().map(|row| row.offset).collect::<Vec<_>>()[..3], [-1, -1, 0]);
    assert_eq!((form.inputs[0].edge, form.inputs[0].fill.map(|fill| fill.bits)), (Edge::Fill, Some(0)), "Border::Zero is a Fill of 0");
    assert_eq!(form.inputs[1].shape, [3, 3, 1, 1]);
    assert_eq!((coeff(&form, 1, 0), coeff(&form, 1, 1)), (vec![0, 0, 0, 1, 0], vec![0, 0, 0, 0, 1]));
    assert_eq!(form.output.dim_of_axis, [Some(0), Some(1), Some(2), None]);
    assert_eq!(parts(&form), [CombinePart::Add]);
    assert_eq!(form.origin, Origin::Api(OpTag::Conv(Box::new(ConvSpec::new()), false)));
    form.validate().unwrap();
    // The border of the spec is the access's edge.
    for (border, edge) in [(Border::Edge, Edge::Clamp), (Border::Mirror, Edge::Mirror), (Border::Wrap, Edge::Wrap)] {
        let form = Comprehension::conv(ConvSpec::new(), &geom, border, src(&x), src(&w)).unwrap();
        assert_eq!((form.inputs[0].edge, form.inputs[0].fill.is_some()), (edge, false));
    }
}

#[test]
fn conv3_has_three_tap_dims_and_no_room_for_a_batch() {
    let ctx = cpu();
    let (x, w) = (array(&ctx, 6 * 5 * 4), array(&ctx, 8));
    let geom = WindowGeom { src: [6, 5, 4, 1], k: [2, 2, 2], stride: [1, 1, 1], before: [0, 0, 0], out: [5, 4, 3], three_d: true };
    let form = Comprehension::conv(ConvSpec::new(), &geom, Border::Zero, src(&x), src(&w)).unwrap();
    assert_eq!(extents(&form), [(5, "P"), (4, "P"), (3, "P"), (2, "R"), (2, "R"), (2, "R")]);
    assert_eq!(coeff(&form, 0, 2), [0, 0, 1, 0, 0, 1]);
    assert_eq!(form.output.dim_of_axis, [Some(0), Some(1), Some(2), None]);
    assert_eq!(form.origin, Origin::Api(OpTag::Conv(Box::new(ConvSpec::new()), true)));
    form.validate().unwrap();
    // A batch makes seven dims, one more than a form has.
    let batched = WindowGeom { src: [6, 5, 4, 2], ..geom };
    assert_eq!(Comprehension::conv(ConvSpec::new(), &batched, Border::Zero, src(&x), src(&w)).err(), Some(FormError::TooMany("dims")));
}

#[test]
fn separable_convolution_multiplies_three_accesses() {
    let ctx = cpu();
    let (x, w0, w1) = (array(&ctx, 8 * 6), array(&ctx, 3), array(&ctx, 5));
    let geom = geom2(8, 6, 1, [3, 5], [1, 1], [1, 2], [8, 6]);
    let form = Comprehension::conv_sep(&geom, Border::Edge, src(&x), src(&w0), src(&w1)).unwrap();
    assert_eq!(extents(&form), [(8, "P"), (6, "P"), (3, "R"), (5, "R")], "no batch dim for a batch of 1");
    assert_eq!(form.inputs.len(), 3);
    assert_eq!((coeff(&form, 1, 0), coeff(&form, 2, 0)), (vec![0, 0, 1, 0], vec![0, 0, 0, 1]));
    assert!(is_mul(&form.body.instrs[0]) && is_mul(&form.body.instrs[1]));
    assert_eq!(form.body.outs.as_slice(), [1]);
    assert_eq!(form.origin, Origin::Api(OpTag::ConvSep));
    form.validate().unwrap();
}

#[test]
fn rank_filters_are_extrema_of_the_window() {
    let ctx = cpu();
    let x = array(&ctx, 8 * 6);
    let geom = geom2(8, 6, 1, [3, 3], [1, 1], [1, 1], [8, 6]);
    for (op, part) in [(WindowOp::Min, CombinePart::Min), (WindowOp::Max, CombinePart::Max)] {
        let form = Comprehension::window_extremum(op, &geom, Border::Ignore, src(&x), None, ElemTy::F32).unwrap();
        assert_eq!(parts(&form), [part]);
        assert_eq!(form.inputs.len(), 1);
        assert_eq!(form.inputs[0].edge, Edge::Fill);
        let identity = if part == CombinePart::Min { f32::INFINITY } else { f32::NEG_INFINITY };
        assert_eq!(form.inputs[0].fill.map(|fill| f32::from_bits(fill.bits as u32)), Some(identity), "an ignored border reads the identity of the extremum");
        assert_eq!(form.origin, Origin::Api(OpTag::Window(op)));
        form.validate().unwrap();
    }
    // A structuring element: `select(m != 0, x0, +inf)`.
    let mask = array(&ctx, 9);
    let form = Comprehension::window_extremum(WindowOp::MaskedMin, &geom, Border::Edge, src(&x), Some(src(&mask)), ElemTy::F32).unwrap();
    assert_eq!(form.inputs.len(), 2);
    assert_eq!(form.inputs[1].shape, [3, 3, 1, 1]);
    assert_eq!(form.body.instrs.iter().map(|instr| instr.op.kind).collect::<Vec<_>>(), [OpKind::Compare(crate::builtin::CmpOp::Ne), OpKind::Select]);
    assert_eq!(form.body.scalars[1].bits, u64::from(f32::INFINITY.to_bits()));
    form.validate().unwrap();
    assert_eq!(Comprehension::window_extremum(WindowOp::MaskedMin, &geom, Border::Edge, src(&x), None, ElemTy::F32).err(), Some(FormError::Window(WindowOp::MaskedMin)), "a masked filter needs its mask");
    assert_eq!(Comprehension::window_extremum(WindowOp::Median, &geom, Border::Edge, src(&x), None, ElemTy::F32).err(), Some(FormError::Window(WindowOp::Median)), "a median is not a form");
}

#[test]
fn sobel_and_gradient_have_two_outputs_sharing_the_access() {
    let ctx = cpu();
    let x = array(&ctx, 8 * 6);
    let geom = geom2(8, 6, 1, [3, 3], [1, 1], [1, 1], [8, 6]);
    for (op, border) in [(WindowOp::Sobel, Border::Mirror), (WindowOp::Gradient, Border::Edge)] {
        let forms = if op == WindowOp::Sobel { Comprehension::sobel(op, &geom, border, src(&x)) } else { Comprehension::gradient(&geom, src(&x)) }.unwrap();
        assert_eq!(forms.len(), 2, "{op:?}");
        assert_eq!(forms[0].inputs[0].map.rows, forms[1].inputs[0].map.rows, "the two derivatives read the same window");
        for form in &forms {
            assert_eq!(extents(form), [(8, "P"), (6, "P"), (3, "R"), (3, "R")]);
            assert_eq!(parts(form), [CombinePart::Add]);
            assert_eq!(form.origin, Origin::Api(OpTag::Window(op)));
            form.validate().unwrap();
        }
        assert_ne!(forms[0].structure_hash(), forms[1].structure_hash(), "dimension 0 weighs the taps differently from dimension 1");
    }
    for op in [WindowOp::SobelMagnitude, WindowOp::SobelFast] {
        let forms = Comprehension::sobel(op, &geom, Border::Edge, src(&x)).unwrap();
        assert_eq!(forms.len(), 1);
        assert_eq!(parts(&forms[0]), [CombinePart::Add, CombinePart::Add]);
        assert!(matches!(forms[0].epilogue, Epilogue::Program(_)));
        forms[0].validate().unwrap();
    }
    assert_eq!(Comprehension::sobel(WindowOp::Sobel, &geom2(8, 6, 1, [5, 5], [1, 1], [2, 2], [8, 6]), Border::Edge, src(&x)).err(), Some(FormError::Window(WindowOp::Sobel)), "a 3 x 3 window");
}

#[test]
fn bilateral_is_two_sums_and_a_division() {
    let ctx = cpu();
    let x = array(&ctx, 8 * 6);
    let geom = geom2(8, 6, 1, [5, 5], [1, 1], [2, 2], [8, 6]);
    let form = Comprehension::bilateral(&geom, Border::Mirror, src(&x), -0.5, -2.0).unwrap();
    assert_eq!(extents(&form), [(8, "P"), (6, "P"), (5, "R"), (5, "R")]);
    assert_eq!(form.inputs.len(), 2, "the centre x(o) and the window x(o + t)");
    assert_eq!(coeff(&form, 0, 0), [1, 0, 0, 0], "the centre reads no taps");
    assert_eq!(coeff(&form, 1, 0), [1, 0, 1, 0]);
    assert_eq!(parts(&form), [CombinePart::Add, CombinePart::Add]);
    let Epilogue::Program(program) = &form.epilogue else { panic!("the division is the epilogue") };
    assert_eq!(program.len(), 1);
    assert_eq!(program[0].op.kind, OpKind::Binary(BinOp::Div));
    assert_eq!(form.origin, Origin::Api(OpTag::Window(WindowOp::Bilateral)));
    form.validate().unwrap();
}

// -- the values the library makes --

fn form_of(ctx: &Context, array: &Array<f32>) -> Option<Comprehension> {
    array.structured().expect("a lazy reduction").form(ctx)
}

#[test]
fn a_reduction_holds_its_form() {
    let ctx = cpu();
    let a = array(&ctx, 1000);
    let lazy = &a * 2.0f32;
    let sum = lazy.sum();
    let value = sum.structured().expect("a reduction of a lazy array is a structured value");
    assert_eq!(value.comp().meta.tag, OpTag::Reduce(ReduceList::new(&[ReduceOp::Sum])));
    let Some(form) = form_of(&ctx, &sum) else {
        assert!(!Algebra::builds_forms(), "LUMA_ALGEBRA=off builds no form");
        return;
    };
    assert_eq!(extents(&form), [(1000, "R")]);
    assert_eq!(form.origin, Origin::Api(OpTag::Reduce(ReduceList::new(&[ReduceOp::Sum]))));
    // The form reads the registered reader node of the operand, which is the one the value reduces.
    let Src::Node(node) = &form.inputs[0].src else { panic!("an expression operand is a node") };
    assert!(Arc::ptr_eq(node, value.operand()));
    form.validate().unwrap();

    let several = lazy.reduce([Reduction::Min, Reduction::SumOfSquares]);
    let form = form_of(&ctx, &several).unwrap();
    assert_eq!(parts(&form), [CombinePart::Min, CombinePart::Add]);
    assert_eq!(form.body.outs.as_slice(), [0, 1]);

    // An evaluated operand is a buffer: the reduction is still lazy (steps on its result fuse), over the reader node of it.
    let buffer = array(&ctx, 1000).sum();
    assert_eq!(form_of(&ctx, &buffer).unwrap().inputs[0].shape, [1000, 1, 1, 1]);
}

#[test]
fn a_step_on_a_reduction_extends_its_epilogue() {
    let ctx = cpu();
    let a = array(&ctx, 1000);
    let mean = (&(&a * 2.0f32).sum() / 1000.0f32).sqrt();
    let Some(form) = form_of(&ctx, &mean) else {
        assert!(!Algebra::builds_forms());
        return;
    };
    let Epilogue::Steps(finish) = &form.epilogue else { panic!("steps on the result are the epilogue") };
    assert_eq!(finish.epilogue.steps().count(), 2);
    assert_eq!(mean.structured().unwrap().finish().epilogue, finish.epilogue, "the value and its form say the same steps");
    assert_eq!(form.origin, Origin::Api(OpTag::Reduce(ReduceList::new(&[ReduceOp::Sum]))));
    form.validate().unwrap();
}

#[test]
fn the_dot_product_is_a_reduction_tagged_dot() {
    let ctx = cpu();
    let (a, b) = (array(&ctx, 1000), array(&ctx, 1000));
    let dot = a.dot(&b);
    let value = dot.structured().expect("a dot product is a lazy reduction");
    assert_eq!(value.comp().meta.tag, OpTag::Dot);
    assert_eq!(value.ops(), ReduceList::new(&[ReduceOp::Sum]));
    assert!(value.fusable().is_some(), "it fuses steps on its result like any sum");
    if let Some(form) = form_of(&ctx, &dot) {
        assert_eq!(form.origin, Origin::Api(OpTag::Dot));
        assert_eq!(form.inputs.len(), 1);
    }
    assert_eq!(dot.item().unwrap(), (0..1000).map(|i| ((i % 7) * (i % 7)) as f32).sum::<f32>());
}

#[test]
fn a_deferred_axis_reduction_holds_the_form_of_its_lines() {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    let ctx = Context::with_tuning(options, |tuning| {
        tuning.reduce_multi_after = 0;
        tuning.axis_defer_cells = 0;
        tuning.axis_defer_weight = 0;
    })
    .unwrap();
    // [w=300, h=400, 2]: far above what the host answers at once.
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; 300 * 400 * 2], [300, 400, 2]).unwrap();
    let lines = x.max_axis(1);
    let value = lines.structured().expect("a long axis reduction stays pending");
    assert!(value.is_axis());
    assert_eq!(value.geom(), Some(AxisGeom { n: 400, stride: 300, lines: 600 }));
    let Some(form) = value.form(&ctx) else {
        assert!(!Algebra::builds_forms());
        return;
    };
    assert_eq!(extents(&form), [(300, "P"), (400, "R"), (2, "P")]);
    assert_eq!(form.origin, Origin::Api(OpTag::ReduceAxis(ReduceList::new(&[ReduceOp::Max]))));
    let Src::Node(node) = &form.inputs[0].src else { panic!("the reader node of the operand") };
    assert!(Arc::ptr_eq(node, value.operand()));
    form.validate().unwrap();
    assert_eq!(lines.to_vec().unwrap(), vec![1.0; 600]);
}

#[test]
fn an_interned_body_exists_for_every_element_type() {
    use ElemTy::*;
    for ty in [F32, U32, I32, Bool, U8, I8, U16, I16, F16, BF16, F64, Df32, C32] {
        assert!(Bodies::x0(ty)[0].op.ty == ty);
    }
    // A new type must be counted in `build::TYPES`: the match has no wildcard.
    let _ = |ty: ElemTy| match ty {
        F32 | U32 | I32 | Bool | U8 | I8 | U16 | I16 | F16 | BF16 | F64 | Df32 | C32 => (),
    };
}

// -- the lazy product (W2-LAZY: the Contraction row of 07 §2.1) --

use crate::jit::structured::contraction::ContractionView;
use crate::jit::structured::sites::LazyMode;
use crate::jit::structured::Launch;

fn lazy_cpu() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| tuning.lazy = LazyMode::Always).unwrap()
}

/// `lhs` is `[K=5, M=3]`, `rhs` `[N=4, K=5]`.
fn product_operands(ctx: &Context) -> (Array<f32>, Array<f32>) {
    (array(ctx, 15).reshape([5, 3]), array(ctx, 20).reshape([4, 5]))
}

#[test]
fn a_contraction_view_has_the_form_of_the_call() {
    let ctx = lazy_cpu();
    let (lhs, rhs) = product_operands(&ctx);
    for (request, tf32) in [(Tf32Request::Default, None), (Tf32Request::Exact, Some(false)), (Tf32Request::Tf32, Some(true))] {
        let attrs = ContractionAttrs { tf32: request, half: false };
        let view = ContractionView { lhs: &lhs, rhs: &rhs, attrs, location: std::panic::Location::caller() };
        let Some(form) = view.form() else {
            assert!(!Algebra::builds_forms(), "LUMA_ALGEBRA=off builds no form");
            return;
        };
        // The space: n and m parallel, k reduce, in that order, fastest first.
        assert_eq!(extents(&form), [(4, "P"), (3, "P"), (5, "R")], "n, m, k");
        // The maps: the left operand's rows are k and m, the right one's n and k.
        assert_eq!(form.inputs[0].shape, [5, 3, 1, 1]);
        assert_eq!(form.inputs[1].shape, [4, 5, 1, 1]);
        assert_eq!((coeff(&form, 0, 0), coeff(&form, 0, 1)), (vec![0, 0, 1], vec![0, 1, 0]), "lhs: rows k, m");
        assert_eq!((coeff(&form, 1, 0), coeff(&form, 1, 1)), (vec![1, 0, 0], vec![0, 0, 1]), "rhs: rows n, k");
        // The body multiplies the two reads, the combiner adds them, the output is `[N, M]`.
        assert_eq!(form.body.instrs.len(), 1);
        assert!(is_mul(&form.body.instrs[0]));
        assert_eq!(parts(&form), [CombinePart::Add]);
        assert!(matches!(form.epilogue, Epilogue::None));
        assert_eq!(form.output.dim_of_axis, [Some(0), Some(1), None, None], "[N, M]");
        // The origin: the call, with the precision it asked for.
        assert_eq!(form.origin, Origin::Api(OpTag::Matmul { tf32 }));
        form.validate().unwrap();
        // It is the builder's form of the same operands, which is what a recogniser or a rule reads.
        let geom = Geometry { m: 3, n: 4, k: 5 };
        let built = Comprehension::contract(src(&lhs), src(&rhs), geom, attrs);
        assert_eq!(form.structure_hash(), built.structure_hash());
    }
}

#[test]
fn a_contraction_view_of_operands_that_make_no_product_has_no_form() {
    let ctx = lazy_cpu();
    let (lhs, rhs) = (array(&ctx, 15).reshape([5, 3]), array(&ctx, 20).reshape([5, 4]));
    let view = ContractionView { lhs: &lhs, rhs: &rhs, attrs: ContractionAttrs { tf32: Tf32Request::Default, half: false }, location: std::panic::Location::caller() };
    assert!(view.form().is_none(), "the shapes do not match");
}

#[test]
fn a_lazy_product_holds_no_form_until_it_is_asked_and_then_remembers_it() {
    let ctx = lazy_cpu();
    let (lhs, rhs) = product_operands(&ctx);
    let c = lhs.matmul(&rhs);
    let value = c.structured().expect("a product at a lazy site is a structured value");
    assert!(value.is_unforced());
    assert_eq!(value.comp().meta.launch, Launch::AtForce);
    assert_eq!(value.comp().meta.tag, OpTag::Matmul { tf32: None });
    assert!(crate::gpu::lock(&value.comp().form).is_none(), "no form is built per product");
    let Some(form) = value.form(&ctx) else {
        assert!(!Algebra::builds_forms());
        return;
    };
    assert_eq!(extents(&form), [(4, "P"), (3, "P"), (5, "R")]);
    assert_eq!(form.origin, Origin::Api(OpTag::Matmul { tf32: None }));
    form.validate().unwrap();
    assert!(crate::gpu::lock(&value.comp().form).is_some(), "asked once, kept");
    assert_eq!(value.form(&ctx).unwrap().structure_hash(), form.structure_hash());
    // Used, the product holds neither its operands nor a form that holds them.
    c.to_vec().unwrap();
    assert!(crate::gpu::lock(&value.comp().form).is_none());
    assert!(value.form(&ctx).is_none(), "a forced product has no operands to say a form of");
    // The transposed forms are the same product over views.
    let transposed = Array::<f32>::from_slice_on(&ctx, &(0..20).map(|i| (i % 7) as f32).collect::<Vec<_>>(), [5, 4]).unwrap();
    let nt = lhs.matmul_nt(&transposed);
    // The view is composed into the map by rule 1 (`contract_view`) when the form is first made, so the maps differ but the form
    // is of the same product.
    let moved = nt.structured().unwrap().form(&ctx).unwrap();
    assert!(matches!(moved.origin, Origin::Rewritten(_)) && extents(&moved) == extents(&form), "a transposed operand is read through the map");
}

#[test]
fn a_lazy_batched_product_holds_the_batched_form() {
    let ctx = lazy_cpu();
    // [K=5, M=3, B=2] by a shared [N=4, K=5].
    let (lhs, rhs) = (array(&ctx, 30).reshape([5, 3, 2]), array(&ctx, 20).reshape([4, 5]));
    let c = lhs.matmul_batched(&rhs);
    let value = c.structured().expect("a batch of one launch at a lazy site is a structured value");
    assert_eq!(value.comp().meta.tag, OpTag::MatmulBatched);
    let Some(form) = value.form(&ctx) else {
        assert!(!Algebra::builds_forms());
        return;
    };
    assert_eq!(extents(&form), [(4, "P"), (3, "P"), (2, "P"), (5, "R")], "n, m, b, k");
    assert_eq!(coeff(&form, 0, 2), [0, 0, 1, 0], "the batched lhs follows b");
    assert_eq!(coeff(&form, 1, 2), [0, 0, 0, 0], "the shared rhs has a zero batch row");
    assert_eq!(form.origin, Origin::Api(OpTag::MatmulBatched));
    form.validate().unwrap();
    let built = Comprehension::contract_batched(src(&lhs), src(&rhs), [5, 3, 4, 2], true, false);
    assert_eq!(form.structure_hash(), built.structure_hash());
}
