//! Shrunk failures of the fuzzer, as literal programs. Each test names the finding it
//! reproduces (see `fuzz-findings.md`); a test is `#[ignore]`d while the bug is open, and the
//! attribute is removed with the fix.
//!
//! Paste the program a failing run prints (`Program { .. }`) into a new test; the glob imports
//! below make the printed variant names resolve.

#![allow(unused_imports)]

use super::generator::{CostChoice::*, Config, InjView, InjView::*, LoopKind::*, Opd::*, Program, Stmt::*, ViewOp::*};
use super::oracle::{BinOp::*, CmpOp::*, RedOp::*, ScanOp::*, Ty::*, UnOp::*};
use super::run;
use crate::PadMode::{Edge, Mirror, Wrap, Zero};

/// The engine's defaults with the given switches.
fn engine(simplify: bool, fast_math: bool) -> Config {
    Config { simplify, fast_math, ..Config::default_engine() }
}

fn check(program: Program) {
    if let Err(failure) = run(&program) {
        panic!("{failure}\n{program:?}");
    }
}

/// Finding 1: `x -= x` with `fast_math`: `Sub(x, x)` simplifies to the constant 0, the node no
/// longer reads the buffer it is evaluated into, and the in-place evaluation refuses it.
#[test]
#[ignore = "finding 1: in-place compound assignment of an expression that no longer reads its target"]
fn fast_math_self_subtraction_does_not_run_in_place() {
    check(Program {
        config: engine(true, true),
        math: false,
        stmts: vec![Input { dims: [12, 1, 1, 1], ty: F32, seed: 1 }, Compound { target: 0, op: Sub, rhs: Var(0) }],
    });
}

/// Finding 1 without `fast_math`: `x *= 0` on integers (a buffer another handle shares) simplifies to the constant 0 (exact rule).
#[test]
#[ignore = "finding 1: in-place compound assignment of an expression that no longer reads its target"]
fn integer_multiplication_by_zero_does_not_run_in_place() {
    check(Program {
        config: engine(true, false),
        math: false,
        stmts: vec![Input { dims: [12, 1, 1, 1], ty: U32, seed: 1 }, Alias { src: 0 }, Compound { target: 0, op: Mul, rhs: Const(0.0) }],
    });
}

/// Finding 2: a lazy expression read at two circular offsets by one kernel, with the fusion
/// limit cutting the batch: the launch binds three buffers, the kernel layout has two.
#[test]
#[ignore = "finding 2: bind group has more bindings than the kernel layout (max_nodes = 2, circular shift of an expression)"]
fn shifted_expression_with_a_tight_node_limit_binds_too_many_buffers() {
    check(Program {
        config: Config { max_nodes: 2, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Unary { src: 0, op: Neg },
            View { src: 1, view: ShiftCirc { offs: vec![-4] } },
            Binary { lhs: Var(1), rhs: Var(2), op: Add },
        ],
    });
}

/// Finding 3: a lazy reduction recorded before a write into the array it reduces must keep the
/// old sum; here the reduced array is a shift (not a view), so the write goes into its own buffer.
#[test]
fn lazy_reduction_sees_a_later_write_to_its_operand() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 7 },
            View { src: 0, view: ShiftCirc { offs: vec![-4] } },
            Reduce { src: 1, op: Sum },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 2, hi: 10 }], rhs: Const(1.0) },
        ],
    });
}

#[test]
fn lazy_reduction_sees_a_later_write_to_its_elementwise_operand() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 7 },
            Unary { src: 0, op: Neg },
            Reduce { src: 1, op: Sum },
            Assign { target: 1, path: vec![], rhs: Const(1.0) },
        ],
    });
}

/// Finding 4: a flipped whole-array write into a lazy array, then a write into a slice of it: the
/// second write is lost (`r[0]` is `x[11]`, not `-2`). The same with an elementwise expression
/// in place of the range, and with two flips; not with a slice-only first write or a second
/// write of the whole array.
#[test]
#[ignore = "finding 4: write into a slice of a lazy array whose expression a flipped write replaced"]
fn slice_write_after_a_flipped_write_into_a_lazy_array_is_lost() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Range { dims: [12, 1, 1, 1], axis: 0, ty: F32 },
            Assign { target: 1, path: vec![InjFlip { axis: 0 }], rhs: Var(0) },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 0, hi: 1 }], rhs: Const(-2.0) },
        ],
    });
}

#[test]
#[ignore = "finding 4: write into a slice of a lazy array whose expression a flipped write replaced"]
fn slice_write_after_a_flipped_write_into_an_expression_is_lost() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Unary { src: 0, op: Neg },
            Assign { target: 1, path: vec![InjFlip { axis: 0 }], rhs: Var(0) },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 0, hi: 1 }], rhs: Const(-2.0) },
        ],
    });
}

/// Finding 5: a view taken before a write into its base, then assigned through: the base is
/// written but the view itself keeps showing the data it had (the old element), not the
/// assigned data ("the receiver shows the new data", `tests/assign.rs`).
#[test]
#[ignore = "finding 5: the receiver of a write through a view taken before an earlier write into the base shows old data"]
fn view_assigned_after_its_base_was_written_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            Assign { target: 0, path: vec![], rhs: Const(-2.0) },
            Assign { target: 1, path: vec![], rhs: Const(4.0) },
        ],
    });
}

#[test]
#[ignore = "finding 5: the receiver of a write through a view taken before an earlier write into the base shows old data"]
fn view_of_a_view_assigned_after_its_base_was_written_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: Df32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            Assign { target: 0, path: vec![InjFlip { axis: 1 }, InjSlice { axis: 0, lo: 3, hi: 5 }], rhs: Const(-2.0) },
            View { src: 1, view: Flip { axis: 1 } },
            Assign { target: 2, path: vec![], rhs: Const(4.0) },
        ],
    });
}

/// Finding 5, simplest form: an evaluated view keeps its own buffer, which a write through it
/// does not update (the base is written, the view still shows the old values).
#[test]
#[ignore = "finding 5: the receiver of a write through an evaluated view shows old data"]
fn evaluated_view_assigned_through_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            EvalVars { vars: vec![1] },
            Assign { target: 1, path: vec![], rhs: Const(4.0) },
        ],
    });
}

/// Finding 6: an `iterate_all` of two steps whose body takes from an array outside the loop,
/// with the fusion limit at 3 nodes or fewer: afterwards the outside array holds wrong
/// elements (`concat(x, x)[13]` is 1, not `x[1]`).
#[test]
#[ignore = "finding 6: iterate_all with a gather from an outer array corrupts it when max_nodes <= 3"]
fn iterate_all_with_a_gather_and_a_tight_node_limit_corrupts_the_outer_array() {
    check(Program {
        config: Config { max_nodes: 2, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 0 },
            View { src: 0, view: Concat { axis: 0, others: vec![0] } },
            Loop {
                kind: IterateAll,
                steps: 2,
                states: vec![0],
                body: vec![Full { dims: [12, 1, 1, 1], ty: U32, value: 0.0 }, View { src: 1, view: Take { idx: 0 } }],
                next: vec![3],
            },
        ],
    });
}

/// Finding 7: `x *= -2` is a lazy expression that reuses `x`'s buffer; a traced loop that reads
/// `x` forces it inside the trace, so the recorded launches include `x *= -2` and every replayed
/// step multiplies `x` again (`x[0]` ends at `-32` instead of `8`).
#[test]
#[ignore = "finding 7: a lazy in-place compound assignment captured by a traced loop is replayed with the loop"]
fn captured_lazy_compound_assignment_is_replayed_with_the_loop() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 1 },
            Compound { target: 0, op: Mul, rhs: Const(-2.0) },
            Loop { kind: IterateAll, steps: 3, states: vec![1], body: vec![Scan { src: 0, axis: 0, op: CumSum }], next: vec![3] },
        ],
    });
}

/// Finding 3, indirect form (found on `LUMA_DEVICE=cpu`): the lazy reduction reduces an
/// expression of `x`, and a later write goes into `x` (through an alias and a slice).
#[test]
fn lazy_reduction_of_an_expression_sees_a_later_write_to_its_input() {
    check(Program {
        config: Config::default_engine(),
        math: true,
        stmts: vec![
            Input { dims: [9, 6, 1, 1], ty: F32, seed: 0 },
            Compound { target: 0, op: Mul, rhs: Const(0.5) },
            Binary { lhs: Var(0), rhs: Var(0), op: Add },
            Alias { src: 1 },
            Reduce { src: 2, op: Min },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 0, hi: 1 }, InjSlice { axis: 1, lo: 2, hi: 5 }], rhs: Const(0.25) },
        ],
    });
}

/// Finding 12 (found by FixF3 while fixing finding 3): two equal expressions share one node, so
/// `cache` of one evaluates the base of a flipped view of the other; the write through that view
/// is then not shown by the view (`v[6]` is `-1`, not `-2`). Same family as finding 5.
#[test]
#[ignore = "finding 12: the receiver of a write through a view whose shared base node a cache evaluated shows old data"]
fn view_shows_old_data_after_a_cache_evaluated_the_shared_base_node() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: U32, seed: 0 },
            Select { mask: 0, a: Const(-2.0), b: Const(-1.0), ty: F32 },
            Unary { src: 1, op: ToDf32 },
            Unary { src: 1, op: ToDf32 },
            View { src: 3, view: Flip { axis: 0 } },
            Cache { src: 2 },
            Assign { target: 4, path: vec![], rhs: Const(-2.0) },
        ],
    });
}

/// Finding 3, hash-consed form (fuzz seed 101): `sum_where` builds `select(mask, x, 0)`, and the
/// later `select` of the same operands is the same node; the lazy reduction recorded before the
/// write through a flipped transpose of it must keep the old sum.
#[test]
fn sum_where_sees_a_later_write_to_the_equal_select() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [9, 6, 1, 1], ty: F32, seed: 0 },
            Range { dims: [9, 6, 1, 1], axis: 1, ty: U32 },
            SumWhere { src: 0, mask: 1 },
            Select { mask: 1, a: Var(0), b: Const(0.0), ty: F32 },
            Assign { target: 3, path: vec![InjFlip { axis: 1 }, InjTranspose], rhs: Const(-2.0) },
        ],
    });
}
