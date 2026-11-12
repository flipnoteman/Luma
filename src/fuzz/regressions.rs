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
fn fast_math_self_subtraction_does_not_run_in_place() {
    check(Program {
        config: engine(true, true),
        math: false,
        stmts: vec![Input { dims: [12, 1, 1, 1], ty: F32, seed: 1 }, Compound { target: 0, op: Sub, rhs: Var(0) }],
    });
}

/// Finding 1 without `fast_math`: `x *= 0` on integers (a buffer another handle shares) simplifies to the constant 0 (exact rule).
#[test]
fn integer_multiplication_by_zero_does_not_run_in_place() {
    check(Program {
        config: engine(true, false),
        math: false,
        stmts: vec![Input { dims: [12, 1, 1, 1], ty: U32, seed: 1 }, Alias { src: 0 }, Compound { target: 0, op: Mul, rhs: Const(0.0) }],
    });
}

/// Finding 2: an expression read directly and through a circular shift, with the fusion limit
/// cutting the batch: the shift of the sub-program is staged with its amount baked into the
/// source, so the kernel never reads its uniform and has no binding for it, but the launch
/// bound the uniform.
#[test]
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
/// with the fusion limit at 3 nodes or fewer: the pending `concat` read by the `take` ran inside
/// the recording and took over its operand's buffer, so the replays overwrote it
/// (`concat(x, x)[13]` was 1, not `x[1]`). Same cause as finding 7.
#[test]
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
/// `x` forces it inside the trace, and the recorded launch (which overwrote `x`'s old buffer) was
/// replayed at every step (`x[0]` ended at `-32` instead of `8`). Fixed with finding 6: a
/// recording never overwrites a buffer that is older than the recording.
#[test]
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

/// Finding 11 (found while fixing finding 2, independent of it): a lazy `Range` of `u32`
/// created after a slice write into a lazy expression loses its own slice write.
#[test]
fn slice_write_into_a_range_created_after_another_slice_write_is_lost() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: F32, seed: 0 },
            Unary { src: 0, op: Neg },
            Assign { target: 1, path: vec![InjSlice { axis: 1, lo: 1, hi: 2 }], rhs: Const(-1.0) },
            Range { dims: [7, 5, 1, 1], axis: 1, ty: U32 },
            Assign { target: 2, path: vec![InjSlice { axis: 1, lo: 3, hi: 4 }], rhs: Const(0.0) },
        ],
    });
}

/// Finding 10 (found while fixing findings 6 and 7; no loop involved): a whole-array write
/// through two flips replaces the lazy array's expression, a view of it is taken afterwards and
/// written through; the write into the view does not reach the array (`element 40` keeps the
/// old value). Same family as finding 4, through a view variable.
#[test]
fn view_of_a_lazy_array_after_a_flipped_write_loses_a_write_through_it() {
    check(Program {
        config: Config { staging: false, ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            Binary { lhs: Var(0), rhs: Const(0.5), op: Minimum },
            Assign { target: 1, path: vec![InjFlip { axis: 0 }, InjFlip { axis: 0 }], rhs: Var(0) },
            View { src: 1, view: Slice { axis: 1, lo: 4, hi: 7 } },
            Assign { target: 2, path: vec![InjSlice { axis: 1, lo: 1, hi: 3 }, InjFlip { axis: 0 }], rhs: Const(1.0) },
        ],
    });
}

/// Finding 9: `v = x.slice(143..299); s = v.shift(3); v.assign(&s)`: the right-hand side reads
/// the region being written at other elements than the one it stores, so the view is evaluated
/// first (it is read before the write); the write was correct but the receiver `v` kept the
/// evaluated old data instead of showing the assigned one (`v[0]` is 1.717 instead of 1.307).
#[test]
fn assign_of_a_shift_of_the_written_region_shows_the_assigned_data() {
    check(Program {
        config: Config { staging: false, ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 143, hi: 299 } },
            View { src: 1, view: ShiftCirc { offs: vec![3] } },
            Assign { target: 1, path: vec![], rhs: Var(2) },
        ],
    });
}

/// Finding 12 (found while fixing 4, 5 and 9): `v1 = x.slice(..); x.transpose().assign(0.0);
/// v2 = v1.slice(..); select(r, &v2, -2.0)`. `v1` was evaluated by the write (it keeps the old
/// element) but `v2`, a view of it, was composed with `v1`'s view of `x` instead of reading
/// `v1`'s buffer, so it showed the written data (`select(r, v2, -2)[1]` is 0, not the old 1).
#[test]
fn view_of_a_view_recorded_before_a_transposed_write_keeps_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: F32, seed: 32202 },
            View { src: 0, view: Slice { axis: 0, lo: 293, hi: 294 } },
            Assign { target: 0, path: vec![InjTranspose], rhs: Const(0.0) },
            View { src: 1, view: Slice { axis: 0, lo: 0, hi: 1 } },
            Range { dims: [12, 1, 1, 1], axis: 0, ty: U32 },
            Select { mask: 3, a: Var(2), b: Const(-2.0), ty: F32 },
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

/// Finding 14 (found while fixing finding 3; same family as finding 5): `cache` of `a` evaluates
/// the arrays sharing work with it, among them the flipped view `v` of the equal expression `b`.
/// The write through `v` then replaced `b`'s lazy expression but left `v`, already evaluated to
/// the old data, alone (`v[6]` is `-1`, not `-2`).
#[test]
fn view_shows_the_written_data_after_a_cache_evaluated_the_shared_base_node() {
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

/// Finding 15 (found while fixing finding 3; same cause as finding 14): a view that was
/// evaluated (here by `eval`) and is then written through with an expression that reads it shows
/// the data from before the write (`v[1]` is `1`, not `0`). Minimized from a program with a traced
/// `iterate_all` (see the next test): no loop is needed.
#[test]
fn evaluated_view_written_with_an_expression_of_itself_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            View { src: 0, view: Flip { axis: 1 } },
            EvalVars { vars: vec![1] },
            View { src: 1, view: ShiftClamp { offs: vec![3, 0] } },
            Binary { lhs: Var(2), rhs: Const(2.0), op: Rem },
            Assign { target: 1, path: vec![], rhs: Var(3) },
        ],
    });
}

/// Finding 15, the original form: the view is evaluated because a traced `iterate_all` reads it.
#[test]
fn view_read_by_a_traced_loop_shows_the_old_data_after_a_write_through_it() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            View { src: 0, view: Flip { axis: 1 } },
            View { src: 0, view: Flip { axis: 1 } },
            Loop { kind: IterateAll, steps: 1, states: vec![1], body: vec![View { src: 2, view: ShiftClamp { offs: vec![3, 0] } }], next: vec![4] },
            Binary { lhs: Var(3), rhs: Const(2.0), op: Rem },
            Assign { target: 2, path: vec![], rhs: Var(4) },
        ],
    });
}

/// Finding 8 (found by `fuzz::threads`: wrong values, and `inf`, only when programs ran at once): the
/// last step of an `eval` loop is postponed while another thread keeps the GPU busy, and the
/// expression after it (`copysign`) computes that step inside its own kernel. The array of the
/// loop is then still lazy when `iterate_until_all` traces its step, which reads it: the recording
/// evaluated it in place over its operand's buffer and every replayed step squared it again. Run by
/// itself the program never fails, so two copies run at once, many times.
#[test]
fn a_lazy_array_fused_into_another_kernel_is_not_evaluated_by_a_traced_loop() {
    let program = Program {
        config: Config::default_engine(),
        math: true,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 4036467767672712798 },
            Input { dims: [7, 5, 1, 1], ty: F32, seed: 1466409304873250584 },
            Compound { target: 1, op: Sub, rhs: Const(-1.0) },
            Loop { kind: PerStepEval, steps: 3, states: vec![1], body: vec![Scan { src: 0, axis: 0, op: ScanMax }, Binary { lhs: Var(2), rhs: Var(2), op: Mul }, Binary { lhs: Var(4), rhs: Const(3.0), op: Maximum }], next: vec![5] },
            Binary { lhs: Var(2), rhs: Const(0.5), op: Copysign },
            Loop { kind: IterateUntilAll, steps: 10, states: vec![1], body: vec![View { src: 2, view: ShiftFill { offs: vec![-1, -2], value: 0.25 } }, Assign { target: 5, path: vec![], rhs: Var(3) }, View { src: 0, view: Take { idx: 0 } }, Alias { src: 6 }, Range { dims: [9, 6, 1, 1], axis: 0, ty: F32 }], next: vec![5] },
        ],
    };
    check(program.clone());
    let ctx = crate::default_context().unwrap();
    let programs = [program.clone(), program];
    for round in 0..60 {
        let failures = super::run_concurrently(&ctx, &programs);
        assert!(failures.is_empty(), "round {round}: {failures:?}");
    }
}

/// Finding 16 (GPU fuzz seed 204; same cause as finding 8): `c = concat(x, x); c /= 2` is a lazy
/// expression that reuses its operand's buffer, and `select(c, ..)` fuses it into its own kernel
/// without materializing it (it stays pending, flagged as fused). Tracing skipped fused nodes when
/// it evaluated the live lazy arrays, so the traced step (which reads `c` through `max_axis`)
/// forced `c` inside the recording: the division ran again at every replayed step (`c[0]` was `2`,
/// not `4`, after 2 steps). Needs a fusion limit of 2 or 3 nodes.
#[test]
fn lazy_array_fused_into_the_loop_state_is_replayed_with_the_traced_step() {
    check(Program {
        config: Config { max_nodes: 2, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 0 },
            View { src: 0, view: Concat { axis: 0, others: vec![0] } },
            Compound { target: 1, op: Div, rhs: Const(2.0) },
            Select { mask: 1, a: Const(0.0), b: Const(2.0), ty: F32 },
            Loop { kind: IterateAll, steps: 2, states: vec![2], body: vec![Unary { src: 3, op: Neg }, AxisReduce { src: 1, axis: 0, op: Max }], next: vec![4] },
        ],
    });
}
