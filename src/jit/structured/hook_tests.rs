//! Tests of the shader kernels' epilogue hook (W3-U4 part B): a product of a forced shader strategy that applies an epilogue program at
//! its stores gives, bit for bit, what the same product gives plain followed by the program's operations as an elementwise pass. The
//! operands hold arbitrary floats (not integers), so that a different order of operations or a contracted multiply-add would show.

use std::panic::Location;
use std::sync::Arc;

use smallvec::smallvec;

use super::contraction::{ContractionAttrs, ContractionView, Tf32Request, CONTRACTION_STRATEGIES};
use super::epilogue::{EpiInput, EpiMap, EpiProgram};
use super::select::execute_in;
use super::strategy::{Output, StrategyId, Subject};
use crate::array::Array;
use crate::builtin::{BinOp, UnOp};
use crate::context::{Context, ContextOptions};
use crate::jit::program::{Instr, Ref};
use crate::jit::{ElemTy, Op, OpKind, Scalar, Src};

type A = Array<f32>;

fn context() -> Context {
    Context::with_tuning(ContextOptions::from_env().calibrate(false), |_| {}).unwrap()
}

/// Deterministic floats in about `[-1, 1)`.
fn floats(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn un(op: UnOp, x: Ref) -> Instr {
    Instr { op: Op::new(OpKind::Unary(op), ElemTy::F32), args: [x, Ref::Unused, Ref::Unused] }
}

fn bin(op: BinOp, x: Ref, y: Ref) -> Instr {
    Instr { op: Op::new(OpKind::Binary(op), ElemTy::F32), args: [x, y, Ref::Unused] }
}

fn program(instrs: Vec<Instr>, inputs: Vec<(&A, EpiMap)>, scalars: Vec<f32>) -> EpiProgram {
    let last = instrs.len() as u32 - 1;
    EpiProgram {
        instrs: Arc::new(instrs),
        inputs: inputs.into_iter().map(|(array, map)| EpiInput { src: array.src().ok().unwrap(), map, ty: ElemTy::F32 }).collect(),
        scalars: scalars.into_iter().map(Scalar::f32).collect(),
        outputs: smallvec![last],
    }
}

/// The product through the strategy `by`, with `epilogue`; `None` for the output if the strategy did not apply it.
fn product(ctx: &Context, lhs: &A, rhs: &A, by: StrategyId, epilogue: Option<&EpiProgram>) -> (Vec<f32>, bool) {
    let view = ContractionView { lhs, rhs, attrs: ContractionAttrs { tf32: Tf32Request::Exact, half: false }, location: Location::caller(), epilogue, partners: &[] };
    let output = execute_in(&CONTRACTION_STRATEGIES, ctx, &Subject::Contraction(&view), Some(by)).ok().unwrap();
    let applied = matches!(output, Output::Epilogued(_));
    let shape = [rhs.shape().dims()[0], lhs.shape().dims()[1]];
    (A::from_storage(ctx, shape.into(), output.into_storage()).to_vec().unwrap(), applied)
}

/// Whether two vectors hold the same bits.
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// `lhs` is `[K, M]`, `rhs` `[N, K]`: the epilogue `program` through `by` against the plain product of `by` followed by `reference`, which
/// maps the product (an array of the output's shape) the way the program does. Returns whether the strategy applied the program.
fn check(ctx: &Context, (k, m, n): (usize, usize, usize), by: StrategyId, program: &EpiProgram, reference: impl Fn(&A) -> A) -> bool {
    let (a, b) = (A::from_slice_on(ctx, &floats(k * m, 1), [k, m]).unwrap(), A::from_slice_on(ctx, &floats(n * k, 2), [n, k]).unwrap());
    check_on(ctx, (&a, &b), by, program, reference)
}

fn check_on(ctx: &Context, (a, b): (&A, &A), by: StrategyId, program: &EpiProgram, reference: impl Fn(&A) -> A) -> bool {
    let (plain, applied_plain) = product(ctx, a, b, by, None);
    assert!(!applied_plain);
    let (fused, applied) = product(ctx, a, b, by, Some(program));
    let shape = [b.shape().dims()[0], a.shape().dims()[1]];
    let want = reference(&A::from_slice_on(ctx, &plain, shape).unwrap()).to_vec().unwrap();
    if applied {
        assert!(same_bits(&fused, &want), "the fused epilogue differs from the separate pass ({:?} elements, first at {:?})", want.len(), fused.iter().zip(&want).position(|(x, y)| x.to_bits() != y.to_bits()));
    } else {
        // Not applied: the plain product, for the caller to apply the program.
        assert!(same_bits(&fused, &plain));
    }
    applied
}

/// A `[N, M]` array that holds `values` (`N` of them) along its columns, `M` times.
fn along_cols(ctx: &Context, values: &[f32], m: usize) -> A {
    let n = values.len();
    A::from_slice_on(ctx, &(0..n * m).map(|i| values[i % n]).collect::<Vec<_>>(), [n, m]).unwrap()
}

/// A `[N, M]` array that holds `values` (`M` of them) along its rows, `N` times each.
fn along_rows(ctx: &Context, values: &[f32], n: usize) -> A {
    let m = values.len();
    A::from_slice_on(ctx, &(0..n * m).map(|i| values[i / n]).collect::<Vec<_>>(), [n, m]).unwrap()
}

/// `relu(acc + bias)`, the bias along the columns.
fn bias_relu(ctx: &Context, n: usize, m: usize) -> (EpiProgram, impl Fn(&A) -> A) {
    let bias = A::from_slice_on(ctx, &floats(n, 3), [n]).unwrap();
    let full = along_cols(ctx, &floats(n, 3), m);
    let relu = program(vec![bin(BinOp::Add, Ref::Acc, Ref::Input(0)), bin(BinOp::Max, Ref::Tmp(0), Ref::Scalar(0))], vec![(&bias, EpiMap::AlongCols)], vec![0.0]);
    (relu, move |c: &A| (c + &full).maximum(0.0))
}

const TILES: [(&str, (usize, usize, usize)); 6] = [
    ("T128 aligned", (64, 128, 128)),
    ("T128 masked", (37, 130, 131)),
    ("T128x32", (33, 130, 20)),
    ("T32x128", (33, 20, 130)),
    ("T32 aligned", (32, 32, 32)),
    ("T32 masked", (5, 17, 19)),
];

#[test]
fn relu_of_the_product_and_a_bias_is_the_separate_pass_on_every_tile() {
    let ctx = context();
    for (name, (k, m, n)) in TILES {
        let (relu, reference) = bias_relu(&ctx, n, m);
        assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_TILED, &relu, reference), "{name}: the tiled kernel applies the epilogue");
    }
}

#[test]
fn the_identity_epilogue_is_the_plain_product() {
    let ctx = context();
    for (name, (k, m, n)) in TILES {
        let identity = program(vec![un(UnOp::Id, Ref::Acc)], vec![], vec![]);
        assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_TILED, &identity, |c| c.clone()), "{name}");
    }
}

#[test]
fn a_tanh_and_a_chain_of_scalar_steps_are_the_separate_pass_and_a_gelu_is_over_the_limit() {
    let ctx = context();
    let shape = (40, 37, 45);
    let gelu_program = {
        let t = Ref::Tmp;
        program(
            vec![
                bin(BinOp::Mul, Ref::Acc, Ref::Acc),
                bin(BinOp::Mul, t(0), Ref::Acc),
                bin(BinOp::Mul, t(1), Ref::Scalar(0)),
                bin(BinOp::Add, Ref::Acc, t(2)),
                bin(BinOp::Mul, t(3), Ref::Scalar(1)),
                un(UnOp::Tanh, t(4)),
                bin(BinOp::Add, t(5), Ref::Scalar(2)),
                bin(BinOp::Mul, Ref::Acc, t(6)),
                bin(BinOp::Mul, t(7), Ref::Scalar(3)),
            ],
            vec![],
            vec![0.044715, 0.797_884_6, 1.0, 0.5],
        )
    };
    let gelu = |c: &A| {
        let cube = &(c * c) * c;
        let inner = (c + &(&cube * 0.044715)) * 0.797_884_6;
        c * &(&inner.tanh() + 1.0) * 0.5
    };
    // Nine instructions: over `SHADER_EPILOGUE_INSTRS`, the plain product for the caller to apply the program to.
    assert!(!check(&ctx, shape, StrategyId::CONTRACTION_TILED, &gelu_program, gelu));
    let tanh = program(vec![un(UnOp::Tanh, Ref::Acc)], vec![], vec![]);
    assert!(check(&ctx, shape, StrategyId::CONTRACTION_TILED, &tanh, |c| c.tanh()));
    // A chain of six scalar steps.
    let chain = program(
        (0..6).map(|k| if k % 2 == 0 { bin(BinOp::Mul, if k == 0 { Ref::Acc } else { Ref::Tmp(k - 1) }, Ref::Scalar(k)) } else { bin(BinOp::Add, Ref::Tmp(k - 1), Ref::Scalar(k)) }).collect(),
        vec![],
        vec![0.5, 0.25, 1.5, -0.125, 0.75, 0.0625],
    );
    let steps = |c: &A| (&(&(&(c * 0.5) + 0.25) * 1.5) + -0.125) * 0.75 + 0.0625;
    assert!(check(&ctx, shape, StrategyId::CONTRACTION_TILED, &chain, steps));
}

#[test]
fn point_row_column_and_scalar_inputs_are_read_where_their_map_says() {
    let ctx = context();
    let (k, m, n) = (33, 29, 31);
    let residual = A::from_slice_on(&ctx, &floats(n * m, 4), [n, m]).unwrap();
    let rows = floats(m, 5);
    let cols = floats(n, 6);
    let (row_array, col_array) = (A::from_slice_on(&ctx, &rows, [m]).unwrap(), A::from_slice_on(&ctx, &cols, [n]).unwrap());
    let one = A::from_slice_on(&ctx, &[0.75], [1]).unwrap();
    // acc * 0.5 + residual
    let point = program(vec![bin(BinOp::Mul, Ref::Acc, Ref::Scalar(0)), bin(BinOp::Add, Ref::Tmp(0), Ref::Input(0))], vec![(&residual, EpiMap::Point)], vec![0.5]);
    assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_TILED, &point, |c| &(c * 0.5) + &residual));
    // (acc + row) * col - one
    let mixed = program(
        vec![bin(BinOp::Add, Ref::Acc, Ref::Input(0)), bin(BinOp::Mul, Ref::Tmp(0), Ref::Input(1)), bin(BinOp::Sub, Ref::Tmp(1), Ref::Input(2))],
        vec![(&row_array, EpiMap::AlongRows), (&col_array, EpiMap::AlongCols), (&one, EpiMap::Scalar)],
        vec![],
    );
    let (row_full, col_full) = (along_rows(&ctx, &rows, n), along_cols(&ctx, &cols, m));
    assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_TILED, &mixed, |c| &(&(c + &row_full) * &col_full) - 0.75));
}

#[test]
fn operands_that_are_expressions_and_operands_that_are_buffers_both_apply_the_epilogue() {
    let ctx = context();
    let (k, m, n) = (35, 40, 33);
    let (a, b) = (A::from_slice_on(&ctx, &floats(k * m, 7), [k, m]).unwrap(), A::from_slice_on(&ctx, &floats(n * k, 8), [n, k]).unwrap());
    let (relu, reference) = bias_relu(&ctx, n, m);
    // One operand an expression (fused into the loads), then both.
    let scaled = &a * 2.0;
    assert!(scaled.pending().is_some());
    assert!(check_on(&ctx, (&scaled, &b), StrategyId::CONTRACTION_TILED, &relu, &reference));
    let scaled_b = &b + 1.0;
    assert!(check_on(&ctx, (&scaled, &scaled_b), StrategyId::CONTRACTION_TILED, &relu, &reference));
    assert!(check_on(&ctx, (&a, &b), StrategyId::CONTRACTION_TILED, &relu, &reference));
}

#[test]
fn a_long_program_and_an_unfit_input_run_plain_and_a_split_product_applies_the_epilogue_in_its_sum() {
    let ctx = context();
    // 64 x 64 over a long K is split: the kernel that sums the slices applies the epilogue (W3-U4 B2), bit for bit.
    let (relu, reference) = bias_relu(&ctx, 64, 64);
    assert!(check(&ctx, (8192, 64, 64), StrategyId::CONTRACTION_TILED, &relu, &reference), "a split product applies the epilogue in its sum");
    // Seventeen instructions are over the limit.
    let long = program((0..17).map(|k| bin(BinOp::Add, if k == 0 { Ref::Acc } else { Ref::Tmp(k - 1) }, Ref::Scalar(0))).collect(), vec![], vec![1.0]);
    assert!(!check(&ctx, (16, 32, 32), StrategyId::CONTRACTION_TILED, &long, |c| {
        let mut x = c.clone();
        for _ in 0..17 {
            x = &x + 1.0;
        }
        x
    }));
    // A bias with too few elements is not read.
    let short = A::from_slice_on(&ctx, &[1.0, 2.0], [2]).unwrap();
    let narrow = program(vec![bin(BinOp::Add, Ref::Acc, Ref::Input(0))], vec![(&short, EpiMap::AlongCols)], vec![]);
    assert!(!check(&ctx, (16, 32, 32), StrategyId::CONTRACTION_TILED, &narrow, |c| c.clone()));
    // Two stored values (a base and its derived value) are the caller's too.
    let mut two = program(vec![un(UnOp::Id, Ref::Acc), un(UnOp::Tanh, Ref::Tmp(0))], vec![], vec![]);
    two.outputs = smallvec![0, 1];
    assert!(!check(&ctx, (16, 32, 32), StrategyId::CONTRACTION_TILED, &two, |c| c.clone()));
}

#[test]
fn a_skinny_product_applies_the_epilogue_split_or_not() {
    let ctx = context();
    // A tall output over a long K, left operand along K: one slice of K per workgroup tile, no sum after.
    let (k, m, n) = (2048, 8192, 8);
    let (relu, reference) = bias_relu(&ctx, n, m);
    assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_SKINNY, &relu, reference), "the skinny kernel applies the epilogue");
    // The common skinny plan is split over K: the sum kernel applies it.
    let (relu, reference) = bias_relu(&ctx, 8, 8);
    assert!(check(&ctx, (8192, 8, 8), StrategyId::CONTRACTION_SKINNY, &relu, reference));
}

#[test]
fn an_input_is_a_buffer_or_an_expression() {
    let ctx = context();
    let (k, m, n) = (20, 33, 34);
    let source = A::from_slice_on(&ctx, &floats(n, 9), [n]).unwrap();
    let pending = &source * 3.0;
    assert!(pending.pending().is_some());
    let epilogue = program(vec![bin(BinOp::Add, Ref::Acc, Ref::Input(0))], vec![(&pending, EpiMap::AlongCols)], vec![]);
    let full = along_cols(&ctx, &floats(n, 9).iter().map(|x| x * 3.0).collect::<Vec<_>>(), m);
    assert!(check(&ctx, (k, m, n), StrategyId::CONTRACTION_TILED, &epilogue, |c| c + &full));
    assert!(matches!(epilogue.inputs[0].src, Src::Node(_) | Src::Buffer(_)));
}

// -- the CPU's native product: the epilogue runs over each tile after its sum (W3-U4 B3) --

/// [`check_on`] for the native CPU product, whose hook compiles in the background: the plain product and the pass stay until it did.
fn check_native(ctx: &Context, dims: (usize, usize, usize), program: &EpiProgram, reference: impl Fn(&A) -> A) {
    let (k, m, n) = dims;
    let (a, b) = (A::from_slice_on(ctx, &floats(k * m, 1), [k, m]).unwrap(), A::from_slice_on(ctx, &floats(n * k, 2), [n, k]).unwrap());
    let start = std::time::Instant::now();
    while !product(ctx, &a, &b, StrategyId::CONTRACTION_NATIVE_CPU, Some(program)).1 {
        assert!(start.elapsed() < std::time::Duration::from_secs(20), "the native product did not apply its epilogue");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(check_on(ctx, (&a, &b), StrategyId::CONTRACTION_NATIVE_CPU, program, reference));
}

#[test]
fn the_native_cpu_product_applies_the_epilogue_over_each_tile_bit_for_bit() {
    let ctx = context();
    if ctx.hal().native_matmul().is_none() {
        return;
    }
    // Shapes that are whole tiles, ragged ones and a few that are tiny; the tile edge is the backend's (32, 64 or 128).
    for (k, m, n) in [(64, 128, 128), (37, 130, 131), (33, 20, 300), (5, 17, 19), (200, 257, 129)] {
        let (relu, reference) = bias_relu(&ctx, n, m);
        check_native(&ctx, (k, m, n), &relu, reference);
    }
    let (k, m, n) = (33, 70, 71);
    let residual = A::from_slice_on(&ctx, &floats(n * m, 4), [n, m]).unwrap();
    let rows = floats(m, 5);
    let cols = floats(n, 6);
    let (row_array, col_array) = (A::from_slice_on(&ctx, &rows, [m]).unwrap(), A::from_slice_on(&ctx, &cols, [n]).unwrap());
    let one = A::from_slice_on(&ctx, &[0.75], [1]).unwrap();
    let point = program(vec![bin(BinOp::Mul, Ref::Acc, Ref::Scalar(0)), bin(BinOp::Add, Ref::Tmp(0), Ref::Input(0))], vec![(&residual, EpiMap::Point)], vec![0.5]);
    check_native(&ctx, (k, m, n), &point, |c| &(c * 0.5) + &residual);
    let mixed = program(
        vec![bin(BinOp::Add, Ref::Acc, Ref::Input(0)), bin(BinOp::Mul, Ref::Tmp(0), Ref::Input(1)), bin(BinOp::Sub, Ref::Tmp(1), Ref::Input(2)), un(UnOp::Tanh, Ref::Tmp(2))],
        vec![(&row_array, EpiMap::AlongRows), (&col_array, EpiMap::AlongCols), (&one, EpiMap::Scalar)],
        vec![],
    );
    let (row_full, col_full) = (along_rows(&ctx, &rows, n), along_cols(&ctx, &cols, m));
    check_native(&ctx, (k, m, n), &mixed, |c| (&(&(c + &row_full) * &col_full) - 0.75).tanh());
}
