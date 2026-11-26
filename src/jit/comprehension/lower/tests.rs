//! The generic lowering against the f64 reference of the form (W2-R0 §5.3-5.6, "fallback tests"). Forms are `Origin::Rewritten`, so
//! no recogniser can take them.

use luma_core::{BackendKind, DeviceKind};
use smallvec::{SmallVec, smallvec};

use super::{build, lowerable, prepare, tile_budget, TEST_BUDGET};
use crate::array::Array;
use crate::builtin::BinOp;
use crate::context::{Context, ContextOptions};
use crate::jit::comprehension::{AffineMap, AffineRow, Access, Body, CombineOp, CombinePart, Combiner, Comprehension, Dim, Epilogue, MapRef, OutMap, Props, Origin, RuleId};
use crate::jit::program::{Instr, Ref};
use crate::jit::{Edge, ElemTy, Op, OpKind, Scalar};

pub(super) fn cpu() -> Context {
    let options = ContextOptions { backend: BackendKind::Cpu, device: DeviceKind::Cpu, ..ContextOptions::from_env() }.calibrate(false);
    Context::with_tuning(options, |tuning| tuning.reduce_multi_after = 0).unwrap()
}

/// A source of `shape` of small integers (exact in `f32` and `f64` for any sum of these tests) and its values.
pub(super) fn source(ctx: &Context, shape: [u32; 4], seed: usize) -> (crate::jit::Src, Vec<f64>) {
    let len: usize = shape.iter().map(|&e| e as usize).product();
    let values: Vec<f32> = (0..len).map(|i| ((i * 7 + seed * 5 + i / 13) % 9) as f32 - 3.0).collect();
    let array = Array::<f32>::from_slice_on(ctx, &values, [shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize]).unwrap();
    (array.src().ok().unwrap(), values.into_iter().map(f64::from).collect())
}

pub(super) fn binary(op: BinOp, a: Ref, b: Ref) -> Instr {
    Instr { op: Op::new(OpKind::Binary(op), ElemTy::F32), args: [a, b, Ref::Unused] }
}

/// A row reading `terms` of dims (`(dim, coefficient)`) plus `offset`.
pub(super) fn row(terms: &[(usize, i32)], offset: i64) -> AffineRow {
    let mut row = AffineRow::ZERO;
    for &(d, c) in terms {
        row.coeff[d] = c;
    }
    row.offset = offset;
    row
}

/// One access: its source of `shape` (with its values) read through `rows`.
pub(super) struct Read {
    pub(super) shape: [u32; 4],
    pub(super) rows: Vec<AffineRow>,
    pub(super) edge: Edge,
    pub(super) fill: Option<f32>,
}

impl Read {
    pub(super) fn plain(shape: [u32; 4], rows: Vec<AffineRow>) -> Read {
        Read { shape, rows, edge: Edge::None, fill: None }
    }
}

/// A form over `dims` (extent, reduce) whose output follows the parallel dims in the order `out` (fastest first), with the data of
/// every access.
pub(super) fn form(ctx: &Context, dims: &[(u64, bool)], out: &[usize], reads: &[Read], instrs: Vec<Instr>, part: CombinePart) -> (Comprehension, Vec<Vec<f64>>) {
    let mut data = Vec::new();
    let mut inputs: SmallVec<[Access; 2]> = SmallVec::new();
    for (j, read) in reads.iter().enumerate() {
        let (src, values) = source(ctx, read.shape, j + 1);
        data.push(values);
        let mut map = AffineMap::ZERO;
        for (slot, row) in map.rows.iter_mut().zip(&read.rows) {
            *slot = *row;
        }
        inputs.push(Access { src, shape: read.shape, ty: ElemTy::F32, map: MapRef::owned(map), edge: read.edge, fill: read.fill.map(Scalar::f32), indirect: None });
    }
    let mut axes = [None; 4];
    for (slot, &d) in axes.iter_mut().zip(out) {
        *slot = Some(d as u8);
    }
    let last = instrs.len() as u32 - 1;
    let reduces = dims.iter().any(|d| d.1);
    let form = Comprehension {
        space: dims.iter().map(|&(extent, reduce)| if reduce { Dim::reduce(extent) } else { Dim::parallel(extent) }).collect(),
        inputs,
        body: Body { instrs: std::sync::Arc::new(instrs), scalars: smallvec![], outs: smallvec![last] },
        combiner: reduces.then(|| Combiner::of(CombineOp::Single(part), ElemTy::F32)),
        epilogue: Epilogue::None,
        output: OutMap::new(axes),
        props: Props::NONE,
        origin: Origin::Rewritten(RuleId(0)),
        ty: ElemTy::F32,
    };
    form.validate().expect("a valid form");
    (form, data)
}

pub(super) fn reference(form: &Comprehension, data: &[Vec<f64>]) -> Vec<f64> {
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    form.eval_reference(&refs)
}

pub(super) fn values(array: &Array<f32>) -> Vec<f64> {
    array.to_vec().expect("the form lowered").into_iter().map(f64::from).collect()
}

/// Builds the form and checks it against the reference, exactly (the data are small integers).
pub(super) fn check(ctx: &Context, form: &Comprehension, data: &[Vec<f64>], what: &str) {
    assert!(lowerable(form) || super::contract::prepare(form).is_some(), "{what}: the form has a lowering");
    let got = values(&build(ctx, form));
    let expected = reference(form, data);
    assert_eq!(got.len(), expected.len(), "{what}");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g, e, "{what}: element {i}");
    }
}

fn mul01() -> Vec<Instr> {
    vec![binary(BinOp::Mul, Ref::Input(0), Ref::Input(1))]
}

/// All permutations of 0..n.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 1 {
        return vec![vec![0]];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..n {
            let mut p = rest.clone();
            p.insert(at, n - 1);
            out.push(p);
        }
    }
    out
}

/// `sum_k lhs(.., ) * rhs(..)` over dims `n`, `m`, `k` in every order of the dims, every order of the axes of each operand and
/// both orders of the output, with a broadcast axis of extent 1 on the operands: G-B1 against the f64 reference.
#[test]
fn a_product_sum_over_every_permutation_of_axes() {
    let ctx = cpu();
    let (n, m) = (5u64, 3u64);
    for k in [1u64, 2, 31, 40] {
        let extents = [n, m, k];
        for space_order in permutations(3) {
            for lhs_axes in permutations(2) {
                for rhs_axes in permutations(2) {
                    for out_swap in [false, true] {
                        // The dims in the order `space_order`: dim `i` stands for the logical dim `space_order[i]` (0 = n, 1 = m, 2 = k).
                        let dims: Vec<(u64, bool)> = space_order.iter().map(|&l| (extents[l], l == 2)).collect();
                        let at = |logical: usize| space_order.iter().position(|&l| l == logical).unwrap();
                        // lhs reads (k, m) in the axes `lhs_axes`, rhs reads (n, k).
                        let side = |logicals: [usize; 2], axes: &[usize]| {
                            let ordered: Vec<usize> = axes.iter().map(|&a| logicals[a]).collect();
                            let mut shape = [1u32; 4];
                            let mut rows = Vec::new();
                            for (a, &l) in ordered.iter().enumerate() {
                                shape[a] = extents[l] as u32;
                                rows.push(row(&[(at(l), 1)], 0));
                            }
                            // A trailing axis of extent 1 that every point reads at 0.
                            shape[2] = 1;
                            rows.push(row(&[], 0));
                            rows.push(row(&[], 0));
                            Read::plain(shape, rows)
                        };
                        let reads = [side([2, 1], &lhs_axes), side([0, 2], &rhs_axes)];
                        let (n_dim, m_dim) = (at(0), at(1));
                        let out: Vec<usize> = if out_swap { vec![m_dim, n_dim] } else { vec![n_dim, m_dim] };
                        let (f, data) = form(&ctx, &dims, &out, &reads, mul01(), CombinePart::Add);
                        check(&ctx, &f, &data, &format!("k={k} space={space_order:?} lhs={lhs_axes:?} rhs={rhs_axes:?} swap={out_swap}"));
                    }
                }
            }
        }
    }
}

#[test]
fn reversed_and_sliced_maps_read_the_right_elements() {
    let ctx = cpu();
    // lhs is a slice of a wider source (offset 2, step 1), rhs reads k reversed.
    let (n, m, k) = (4u64, 3u64, 33u64);
    let dims = [(n, false), (m, false), (k, true)];
    let reads = [
        Read::plain([(k + 5) as u32, m as u32, 1, 1], vec![row(&[(2, 1)], 2), row(&[(1, 1)], 0)]),
        Read::plain([n as u32, k as u32, 1, 1], vec![row(&[(0, 1)], 0), row(&[(2, -1)], k as i64 - 1)]),
    ];
    let (f, data) = form(&ctx, &dims, &[0, 1], &reads, mul01(), CombinePart::Add);
    check(&ctx, &f, &data, "slice and reverse");
}

#[test]
fn a_k_of_4097_and_prime_extents() {
    let ctx = cpu();
    for (n, m, k) in [(7u64, 11u64, 4097u64), (1, 13, 4097), (2, 1, 17)] {
        let dims = [(n, false), (m, false), (k, true)];
        let reads = [
            Read::plain([k as u32, m as u32, 1, 1], vec![row(&[(2, 1)], 0), row(&[(1, 1)], 0)]),
            Read::plain([n as u32, k as u32, 1, 1], vec![row(&[(0, 1)], 0), row(&[(2, 1)], 0)]),
        ];
        let (f, data) = form(&ctx, &dims, &[0, 1], &reads, mul01(), CombinePart::Add);
        check(&ctx, &f, &data, &format!("n={n} m={m} k={k}"));
    }
}

#[test]
fn min_max_and_prod_combiners_and_bodies_of_one_access() {
    let ctx = cpu();
    let dims = [(5u64, false), (3, false), (9, true)];
    let reads = [
        Read::plain([9, 3, 1, 1], vec![row(&[(2, 1)], 0), row(&[(1, 1)], 0)]),
        Read::plain([5, 9, 1, 1], vec![row(&[(0, 1)], 0), row(&[(2, 1)], 0)]),
    ];
    for part in [CombinePart::Min, CombinePart::Max] {
        let (f, data) = form(&ctx, &dims, &[0, 1], &reads, vec![binary(BinOp::Add, Ref::Input(0), Ref::Input(1))], part);
        check(&ctx, &f, &data, &format!("{part:?}"));
    }
    // A squared distance `sum (a - b)^2`.
    let (f, data) = form(&ctx, &dims, &[0, 1], &reads, vec![binary(BinOp::Sub, Ref::Input(0), Ref::Input(1)), binary(BinOp::Mul, Ref::Tmp(0), Ref::Tmp(0))], CombinePart::Add);
    check(&ctx, &f, &data, "distance");
    // Nothing reduces: the body per point.
    let dims = [(5u64, false), (3, false)];
    let reads = [Read::plain([5, 3, 1, 1], vec![row(&[(0, 1)], 0), row(&[(1, 1)], 0)]), Read::plain([3, 5, 1, 1], vec![row(&[(1, 1)], 0), row(&[(0, 1)], 0)])];
    let (f, data) = form(&ctx, &dims, &[0, 1], &reads, mul01(), CombinePart::Add);
    check(&ctx, &f, &data, "elementwise");
}

#[test]
fn a_budget_smaller_than_the_space_tiles_it() {
    let ctx = cpu();
    let (n, m, k) = (6u64, 5u64, 40u64);
    let dims = [(n, false), (m, false), (k, true)];
    let reads = [
        Read::plain([k as u32, m as u32, 1, 1], vec![row(&[(2, 1)], 0), row(&[(1, 1)], 0)]),
        Read::plain([n as u32, k as u32, 1, 1], vec![row(&[(0, 1)], 0), row(&[(2, 1)], 0)]),
    ];
    let (f, data) = form(&ctx, &dims, &[0, 1], &reads, mul01(), CombinePart::Add);
    let whole = values(&build(&ctx, &f));
    // Slabs of the output only (the reduce axis fits), then chunks of the reduce axis too (partial results combine in order).
    for budget in [200u64, 60, 41, 40, 39, 17, 7] {
        TEST_BUDGET.with(|b| b.set(Some(budget)));
        let plan = prepare(&f).unwrap();
        assert!(plan.tiles() > 1 || budget >= n * m * k, "budget {budget}");
        let first = values(&build(&ctx, &f));
        let second = values(&build(&ctx, &f));
        TEST_BUDGET.with(|b| b.set(None));
        assert_eq!(first, whole, "budget {budget}");
        assert_eq!(first, second, "chunk-order determinism at budget {budget}");
        assert_eq!(first, reference(&f, &data), "budget {budget}");
    }
    assert!(tile_budget() > 1 << 20);
}

// -- G-B2: the taps of a window --

use crate::error::LumaError;
use crate::image::{Border, ConvSpec};
use crate::jit::comprehension::build::WindowGeom;
use crate::jit::structured::strategy::FamilyTag;
use crate::profile::DecisionSite;

/// The form of an API op as a form no tag claims.
fn untagged(mut form: Comprehension) -> Comprehension {
    form.origin = Origin::Rewritten(RuleId(1));
    form
}

fn stored(ctx: &Context, len: usize, seed: usize) -> Array<f32> {
    let values: Vec<f32> = (0..len).map(|i| ((i * 5 + seed * 3 + i / 7) % 9) as f32 - 3.0).collect();
    Array::from_slice_on(ctx, &values, len).unwrap()
}

#[test]
fn a_correlation_with_few_taps_is_unrolled() {
    let ctx = cpu();
    for taps in [3usize, 9, 64] {
        for (stride, padding) in [(1usize, 0usize), (1, taps / 2), (2, 1)] {
            let len = 100usize;
            let (x, w) = (stored(&ctx, len, 1), stored(&ctx, taps, 2));
            let out = (len + 2 * padding - taps) / stride + 1;
            let form = untagged(Comprehension::conv1d(x.src().ok().unwrap(), w.src().ok().unwrap(), len, taps, out, stride, padding).unwrap());
            let plan = prepare(&form).expect("a lowering");
            assert!(plan.unrolls() && plan.taps() == taps as u64, "taps={taps}");
            let expected = x.conv1d(&w, stride, padding).to_vec().unwrap();
            let got = build(&ctx, &form).to_vec().unwrap();
            assert_eq!(got, expected, "taps={taps} stride={stride} padding={padding}");
        }
    }
}

fn geom2(src: [u32; 2], k: [u32; 2], before: [u32; 2], out: [u32; 2]) -> WindowGeom {
    WindowGeom { src: [src[0], src[1], 1, 1], k: [k[0], k[1], 1], stride: [1, 1, 1], before: [before[0], before[1], 0], out: [out[0], out[1], 1], three_d: false }
}

#[test]
fn a_2d_correlation_matches_conv2_for_every_border() {
    let ctx = cpu();
    let (w, h, kw, kh) = (13u32, 9u32, 3u32, 5u32);
    let x = stored(&ctx, (w * h) as usize, 3).reshape([w as usize, h as usize]);
    let kernel = stored(&ctx, (kw * kh) as usize, 4).reshape([kw as usize, kh as usize]);
    for border in [Border::Zero, Border::Edge, Border::Mirror, Border::Wrap] {
        let geom = geom2([w, h], [kw, kh], [kw / 2, kh / 2], [w, h]);
        let (xs, ks) = (x.src().ok().unwrap(), kernel.src().ok().unwrap());
        let form = untagged(Comprehension::conv(ConvSpec::new().border(border), &geom, border, xs, ks).unwrap());
        let plan = prepare(&form).expect("a lowering");
        assert!(plan.unrolls() && plan.taps() == u64::from(kw * kh));
        let expected = x.conv2(&kernel, ConvSpec::new().border(border)).to_vec().unwrap();
        assert_eq!(build(&ctx, &form).to_vec().unwrap(), expected, "{border:?}");
    }
}

#[test]
fn more_than_the_unroll_limit_has_no_lowering() {
    let ctx = cpu();
    let taps = 65usize;
    let len = 200usize;
    let (x, w) = (stored(&ctx, len, 1), stored(&ctx, taps, 2));
    let form = untagged(Comprehension::conv1d(x.src().ok().unwrap(), w.src().ok().unwrap(), len, taps, len - taps + 1, 1, 0).unwrap());
    assert!(!lowerable(&form));
    let built = build(&ctx, &form);
    match built.to_vec() {
        Err(error) => assert!(format!("{error:?}").contains("InvalidShape") || format!("{error}").contains("no lowering"), "{error}"),
        Ok(_) => panic!("a form without a lowering poisons its array"),
    }
    let _ = LumaError::InvalidShape(String::new());
}

// -- The registry --

#[test]
fn the_generic_strategies_record_their_decision() {
    let ctx = cpu();
    ctx.record_decisions(true);
    let dims = [(5u64, false), (3, false), (9, true)];
    let reads = [
        Read::plain([9, 3, 1, 1], vec![row(&[(2, 1)], 0), row(&[(1, 1)], 0)]),
        Read::plain([5, 9, 1, 1], vec![row(&[(0, 1)], 0), row(&[(2, 1)], 0)]),
    ];
    let (f, data) = form(&ctx, &dims, &[0, 1], &reads, mul01(), CombinePart::Add);
    let got = values(&build(&ctx, &f));
    assert_eq!(got, reference(&f, &data));
    let decisions: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::Comprehension).collect();
    assert_eq!(decisions.len(), 1, "{decisions:?}");
    assert!(decisions[0].chosen.starts_with("generic-"), "{}", decisions[0].chosen);
    assert_eq!(decisions[0].chosen, "generic-reduce");
    // A taps form is the unrolled strategy.
    let (x, w) = (stored(&ctx, 40, 1), stored(&ctx, 5, 2));
    let conv = untagged(Comprehension::conv1d(x.src().ok().unwrap(), w.src().ok().unwrap(), 40, 5, 36, 1, 0).unwrap());
    let _ = build(&ctx, &conv).to_vec().unwrap();
    let decisions: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == DecisionSite::Comprehension).collect();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].chosen, "generic-unrolled");
    assert_eq!(crate::jit::structured::strategy::StrategyId::GENERIC_REDUCE.family(), FamilyTag::Generic);
}

#[test]
fn an_api_form_never_reaches_the_generic_strategies() {
    let ctx = cpu();
    let (x, w) = (stored(&ctx, 40, 1), stored(&ctx, 5, 2));
    let api = Comprehension::conv1d(x.src().ok().unwrap(), w.src().ok().unwrap(), 40, 5, 36, 1, 0).unwrap();
    assert!(build(&ctx, &api).to_vec().is_err(), "an API form runs through its own strategy");
}
