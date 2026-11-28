//! Shrunk failures of the fuzzer, as literal programs. Each test names the finding it
//! reproduces (see `fuzz-findings.md`); a test is `#[ignore]`d while the bug is open, and the
//! attribute is removed with the fix.
//!
//! Paste the program a failing run prints (`Program { .. }`) into a new test; the glob imports
//! below make the printed variant names resolve.

#![allow(unused_imports)]

use super::generator::{CostChoice::*, Config, InjView, InjView::*, LoopKind::*, Opd::*, Program, Stmt::*, ViewOp::*};
use super::oracle::{BinOp::*, Class::*, CmpOp::*, RedOp::*, ScanOp::*, Ty::*, UnOp::*};
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

/// Finding 5: a view taken before a write into its base keeps showing the data it had (the old
/// element) when a region of the base is written afterwards (through `slice_mut`: a view is
/// read-only, `tests/assign.rs`).
#[test]
fn view_taken_before_a_write_into_its_base_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            Assign { target: 0, path: vec![], rhs: Const(-2.0) },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 3, hi: 4 }], rhs: Const(4.0) },
        ],
    });
}

#[test]
fn view_of_a_view_taken_before_a_write_into_its_base_shows_the_old_data() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: Df32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            Assign { target: 0, path: vec![InjFlip { axis: 1 }, InjSlice { axis: 0, lo: 3, hi: 5 }], rhs: Const(-2.0) },
            View { src: 1, view: Flip { axis: 1 } },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 3, hi: 4 }, InjFlip { axis: 1 }], rhs: Const(4.0) },
        ],
    });
}

/// Finding 5, simplest form: an evaluated view keeps its own buffer, which a write into its base
/// does not update (the base region is written, the view still shows the old values).
#[test]
fn evaluated_view_keeps_the_old_data_after_a_write_into_its_base() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 4 } },
            EvalVars { vars: vec![1] },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 3, hi: 4 }], rhs: Const(4.0) },
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
/// through two flips replaces the lazy array's expression, a view of it is taken afterwards, and
/// a region of the array (the view's, then a smaller one) is written; the write does not reach
/// the array (`element 40` keeps the old value). Same family as finding 4.
#[test]
fn slice_write_after_a_flipped_write_and_a_view_of_the_lazy_array_is_lost() {
    check(Program {
        config: Config { staging: false, ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 0 },
            Binary { lhs: Var(0), rhs: Const(0.5), op: Minimum },
            Assign { target: 1, path: vec![InjFlip { axis: 0 }, InjFlip { axis: 0 }], rhs: Var(0) },
            View { src: 1, view: Slice { axis: 1, lo: 4, hi: 7 } },
            Assign { target: 1, path: vec![InjSlice { axis: 1, lo: 4, hi: 7 }, InjSlice { axis: 1, lo: 1, hi: 3 }, InjFlip { axis: 0 }], rhs: Const(1.0) },
        ],
    });
}

/// Finding 9: `v = x.slice(143..299); s = v.shift(3); x.slice_mut(143..299).assign(&s)`: the
/// right-hand side reads the region being written at other elements than the one it stores, so
/// the view is evaluated first (it is read before the write). The region must show the assigned
/// data (formerly the receiver `v` of the write kept the evaluated old data: `v[0]` was 1.717
/// instead of 1.307).
#[test]
fn assign_of_a_shift_of_the_written_region_reads_the_old_data() {
    check(Program {
        config: Config { staging: false, ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 143, hi: 299 } },
            View { src: 1, view: ShiftCirc { offs: vec![3] } },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 143, hi: 299 }], rhs: Var(2) },
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
            Compare { lhs: 3, rhs: Const(0.0), op: Gt },
            Select { mask: 4, a: Var(2), b: Const(-2.0), ty: F32 },
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
            Compare { lhs: 1, rhs: Const(0.0), op: Gt },
            SumWhere { src: 0, mask: 2 },
            Select { mask: 2, a: Var(0), b: Const(0.0), ty: F32 },
            Assign { target: 4, path: vec![InjFlip { axis: 1 }, InjTranspose], rhs: Const(-2.0) },
        ],
    });
}

/// Finding 14 (found while fixing finding 3; same family as finding 5): `cache` of `a` evaluates
/// the arrays sharing work with it, among them the flipped view `v` of the equal expression `b`.
/// The write through `b.flip_mut(0)` then replaced `b`'s lazy expression and must leave `v`,
/// already evaluated to the old data, alone (`v[6]` is `-1`).
#[test]
fn view_keeps_the_old_data_after_a_cache_evaluated_the_shared_base_node() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: U32, seed: 0 },
            Compare { lhs: 0, rhs: Const(0.0), op: Gt },
            Select { mask: 1, a: Const(-2.0), b: Const(-1.0), ty: F32 },
            Unary { src: 2, op: ToDf32 },
            Unary { src: 2, op: ToDf32 },
            View { src: 4, view: Flip { axis: 0 } },
            Cache { src: 3 },
            Assign { target: 4, path: vec![InjFlip { axis: 0 }], rhs: Const(-2.0) },
        ],
    });
}

/// Finding 15 (found while fixing finding 3; same cause as finding 14): a view that was
/// evaluated (here by `eval`), and an expression that reads it, written back through the same
/// flipped region of the array: the view keeps the data from before the write (`v[1]` is `1`).
/// Minimized from a program with a traced `iterate_all` (see the next test): no loop is needed.
#[test]
fn evaluated_view_keeps_the_old_data_when_its_array_is_written_with_an_expression_of_it() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            View { src: 0, view: Flip { axis: 1 } },
            EvalVars { vars: vec![1] },
            View { src: 1, view: ShiftClamp { offs: vec![3, 0] } },
            Binary { lhs: Var(2), rhs: Const(2.0), op: Rem },
            Assign { target: 0, path: vec![InjFlip { axis: 1 }], rhs: Var(3) },
        ],
    });
}

/// Finding 15, the original form: the view is evaluated because a traced `iterate_all` reads it.
#[test]
fn view_read_by_a_traced_loop_keeps_the_old_data_after_a_write_into_its_array() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            View { src: 0, view: Flip { axis: 1 } },
            View { src: 0, view: Flip { axis: 1 } },
            Loop { kind: IterateAll, steps: 1, states: vec![1], body: vec![View { src: 2, view: ShiftClamp { offs: vec![3, 0] } }], next: vec![4] },
            Binary { lhs: Var(3), rhs: Const(2.0), op: Rem },
            Assign { target: 0, path: vec![InjFlip { axis: 1 }], rhs: Var(4) },
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
            Compare { lhs: 1, rhs: Const(0.0), op: Gt },
            Select { mask: 2, a: Const(0.0), b: Const(2.0), ty: F32 },
            Loop { kind: IterateAll, steps: 2, states: vec![3], body: vec![Unary { src: 4, op: Neg }, AxisReduce { src: 1, axis: 0, op: Max }], next: vec![5] },
        ],
    });
}

/// Finding 18 (GPU fuzz seed 1002): `s = select(c, 5, c)` is a lazy array, `a = s.clone()`, a
/// lazy slice write into `s` replaces the expression both denote, and the expression is
/// evaluated; then `s %= 3`. `sole_owner` counted the references of the replacing node (one,
/// held by the link the arrays share) instead of the claims of the arrays, so `s` was rebound
/// to a new expression instead of changing the shared buffer and the clone kept the old data
/// (`a[1]` is `5`, not `2`).
#[test]
fn clone_of_a_lazy_array_misses_a_compound_assignment_after_a_lazy_slice_write() {
    check(Program {
        config: Config {
            max_nodes: 2,
            max_inlined: None,
            max_live: None,
            max_inputs: None,
            assume_busy: false,
            speculate: true,
            simplify: true,
            staging: false,
            fast_math: false,
            df_split: false,
            chunk_bytes: None,
            memory_budget: None,
            slab_bytes: None,
            split: false,
            file_leaves: false,
            cost: Test,
        },
        math: true,
        stmts: vec![
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            View { src: 0, view: Concat { axis: 1, others: vec![0, 0] } },
            Compare { lhs: 1, rhs: Const(0.0), op: Gt },
            Select { mask: 2, a: Const(5.0), b: Var(1), ty: U32 },
            Alias { src: 3 },
            Assign { target: 3, path: vec![InjSlice { axis: 1, lo: 6, hi: 13 }], rhs: Const(3.0) },
            Compound { target: 3, op: Rem, rhs: Const(3.0) },
        ],
    });
}

/// Finding 19 (`LUMA_DEVICE=cpu`, seeds 2001 and 2003): `m = minimum(x, x)` simplifies to a copy
/// of `x`, and the store of `x.flip(0).assign(&m)`, `Id(m)`, was simplified to `Id(x)` too
/// (`Id(Id(x))`), so the write no longer saw the pending copy `m` that reads the target and
/// had it evaluated first: it read the buffer it overwrites at other elements (`x[0]` is `0.5`,
/// not the old `x[299]`). Only an execution order that lets a write overtake a read shows it
/// (llvmpipe); `simplify: false` always passed.
#[test]
fn flipped_assign_of_a_simplified_self_minimum_reads_overwritten_data() {
    check(Program {
        config: Config::default_engine(),
        math: true,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: F32, seed: 0 },
            Binary { lhs: Var(0), rhs: Var(0), op: Minimum },
            Assign { target: 0, path: vec![InjFlip { axis: 0 }], rhs: Var(1) },
        ],
    });
}

/// Finding 19, second form: a `Df32` square, written through a transpose.
#[test]
fn transposed_assign_of_a_simplified_self_maximum_reads_overwritten_data() {
    check(Program {
        config: Config::default_engine(),
        math: true,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: Df32, seed: 0 },
            Binary { lhs: Var(0), rhs: Var(0), op: Maximum },
            Assign { target: 0, path: vec![InjTranspose], rhs: Var(1) },
        ],
    });
}

/// Finding 20 (GPU fuzz seed 5007): `v = lin.slice(3..6).flip(0)` is evaluated by `c = v.cache()`
/// (`c` shares `v`'s buffer, like a clone), then `lin.slice_mut(3..6).flip_mut(0).assign(&d)`
/// where the base `lin` is a lazy expression: the write replaces the lazy base, and the buffer
/// `v` had been evaluated into, and `c` with it, must keep the old data (`c[0]` is `5.875`).
#[test]
fn cache_of_an_evaluated_view_keeps_the_old_data_after_a_write_into_its_lazy_base() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Linspace { start: 3.125, end: 2.625, n: 9 },
            View { src: 0, view: Slice { axis: 0, lo: 3, hi: 6 } },
            View { src: 1, view: Flip { axis: 0 } },
            Cache { src: 2 },
            Binary { lhs: Var(1), rhs: Const(0.5), op: Div },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 3, hi: 6 }, InjFlip { axis: 0 }], rhs: Var(4) },
        ],
    });
}

/// Finding 21 (`LUMA_DEVICE=cpu`, seed 7002; also fails on the GPU): `m = minimum(x, 0)` is a lazy
/// expression and a lazy slice write goes into it (the expression is replaced); `v = m.slice(..)`
/// is a view of the replacement, then `m /= 1`. The compound assignment saw no sharing (it
/// looked at the claims on the node `m` was created with, not on the replacement `v` holds), so
/// it rebound `m` to a new expression reading the replacement, and the later slice write into
/// `m` went into a node `m` no longer denoted (`m[224]` is `0`, not `7`).
#[test]
fn slice_write_after_a_compound_assignment_of_a_lazy_array_is_lost() {
    check(Program {
        config: Config::default_engine(),
        math: true,
        stmts: vec![
            Input { dims: [24, 20, 1, 1], ty: U32, seed: 0 },
            Binary { lhs: Var(0), rhs: Const(0.0), op: Minimum },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 17, hi: 23 }, InjSlice { axis: 1, lo: 15, hi: 20 }], rhs: Const(1.0) },
            View { src: 1, view: Slice { axis: 0, lo: 1, hi: 13 } },
            Compound { target: 1, op: Div, rhs: Const(1.0) },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 1, hi: 13 }, InjSlice { axis: 0, lo: 7, hi: 10 }, InjSlice { axis: 1, lo: 9, hi: 14 }], rhs: Const(7.0) },
        ],
    });
}

/// Finding 21 without simplification (`m /= 1` stays a division): the same failure, so the
/// cause is not a simplification.
#[test]
fn slice_write_after_a_compound_assignment_of_a_lazy_array_is_lost_without_simplify() {
    check(Program {
        config: Config { simplify: false, ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [24, 20, 1, 1], ty: U32, seed: 0 },
            Binary { lhs: Var(0), rhs: Const(0.0), op: Minimum },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 17, hi: 23 }, InjSlice { axis: 1, lo: 15, hi: 20 }], rhs: Const(1.0) },
            View { src: 1, view: Slice { axis: 0, lo: 1, hi: 13 } },
            Compound { target: 1, op: Div, rhs: Const(1.0) },
            Assign { target: 1, path: vec![InjSlice { axis: 0, lo: 1, hi: 13 }, InjSlice { axis: 0, lo: 7, hi: 10 }, InjSlice { axis: 1, lo: 9, hi: 14 }], rhs: Const(7.0) },
        ],
    });
}

/// Epilogue fusion (item 5): `a = abs(sum(x))` is a step on a lazy reduction; the reduction is
/// then written in place (`s -= s`, which records it into its own buffer). `a` is the value the
/// sum had when `a` was written, so it must not read the buffer afterwards.
#[test]
fn a_step_on_a_reduction_does_not_read_the_buffer_the_reduction_is_later_written_through() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Reduce { src: 0, op: Sum },
            Unary { src: 1, op: Abs },
            Compound { target: 1, op: Sub, rhs: Var(1) },
        ],
    });
}

/// A stack of generated values (`Range`): the kernel that writes the stacked layout through
/// workgroup memory must define the global index the generators read, like the plain one.
#[test]
fn stack_of_a_generated_range_defines_the_global_index() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [9, 6, 1, 1], ty: F32, seed: 0 },
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            Compare { lhs: 1, rhs: Const(0.0), op: Gt },
            Select { mask: 2, a: Var(1), b: Const(2.0), ty: U32 },
            View { src: 3, view: Stack { axis: 0, others: vec![1] } },
        ],
    });
}

/// A register loop (`iterate_all`) whose step tests the class of a value: the loop kernel
/// defines the guard library the test reads its bits through.
#[test]
fn iterate_all_with_a_class_test_defines_the_guard_library() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Compare { lhs: 0, rhs: Const(1.0), op: Le },
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 1 },
            Loop { kind: IterateAll, steps: 11, states: vec![1], body: vec![Classify { src: 2, class: Finite }], next: vec![3] },
        ],
    });
}

/// Operands broadcast along their extents of 1: `[7, 1] + [1, 5]` is `[7, 5]`, in binary
/// operations and `select`; `+=` repeats its right-hand side to the target's shape.
#[test]
fn extents_of_one_broadcast_along_each_axis() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: F32, seed: 0 },
            AxisReduce { src: 0, axis: 1, op: Max },
            AxisReduce { src: 0, axis: 0, op: Max },
            Binary { lhs: Var(1), rhs: Var(2), op: Add },
            Range { dims: [7, 5, 1, 1], axis: 0, ty: U32 },
            Compare { lhs: 4, rhs: Const(0.0), op: Gt },
            Select { mask: 5, a: Var(1), b: Var(2), ty: F32 },
            Compound { target: 0, op: Add, rhs: Var(2) },
        ],
    });
}

/// A slice, flip or transpose, and a reshape of one, is read-only: `assign`, `put` and `write`
/// into it fail with `InvalidShape` (a clone of the array is written through `slice_mut`).
#[test]
fn writes_into_read_only_views_fail() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: F32, seed: 0 },
            View { src: 0, view: Slice { axis: 0, lo: 1, hi: 3 } },
            Assign { target: 1, path: vec![], rhs: Const(1.0) },
            Assign { target: 1, path: vec![InjFlip { axis: 0 }], rhs: Const(1.0) },
            Put { target: 1, idx: vec![0], vals: Const(1.0) },
            Write { target: 1, seed: 3 },
            Reshape { src: 1, dims: [10, 1, 1, 1] },
            Assign { target: 2, path: vec![], rhs: Const(1.0) },
            Assign { target: 0, path: vec![InjSlice { axis: 0, lo: 1, hi: 3 }], rhs: Const(1.0) },
        ],
    });
}

/// An `i32` constant that rounds to `2^31` as an `f32` saturates to `i32::MAX` (it does not wrap
/// to `i32::MIN`) on both sides.
#[test]
fn i32_constants_saturate() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![Full { dims: [12, 1, 1, 1], ty: I32, value: 2147483600.0 }],
    });
}

/// The edges of `i32` division, remainder, absolute value and shifts, as measured on WGSL:
/// `MIN / -1 = MIN`, `x / 0 = x`, `MIN % -1 = 0`, `x % 0 = 0`, `abs(MIN) = MIN`, arithmetic `>>`
/// by an amount modulo 32.
#[test]
fn i32_division_and_shift_edges() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: I32, seed: 7 },
            Full { dims: [12, 1, 1, 1], ty: I32, value: -2147483648.0 },
            Binary { lhs: Var(1), rhs: Const(-1.0), op: Div },
            Binary { lhs: Var(1), rhs: Const(-1.0), op: Rem },
            Binary { lhs: Var(0), rhs: Const(0.0), op: Div },
            Binary { lhs: Var(0), rhs: Const(0.0), op: Rem },
            Binary { lhs: Var(0), rhs: Var(1), op: Div },
            Unary { src: 1, op: Abs },
            Unary { src: 0, op: Neg },
            Unary { src: 0, op: Sign },
            Binary { lhs: Var(0), rhs: Const(33.0), op: Shr },
            Binary { lhs: Var(0), rhs: Const(31.0), op: Shl },
            Binary { lhs: Var(0), rhs: Var(0), op: Mul },
        ],
    });
}

/// Casts among `f32`, `u32`, `i32` and `bool`, and the `bool` operators, `count`, `any` and `all`.
#[test]
fn casts_and_bool_operations() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 5 },
            Unary { src: 0, op: ToU32 },
            Unary { src: 0, op: ToI32 },
            Unary { src: 1, op: ToI32 },
            Unary { src: 2, op: ToU32 },
            Unary { src: 2, op: ToF32 },
            Compare { lhs: 0, rhs: Const(0.5), op: Gt },
            Compare { lhs: 2, rhs: Const(-1.0), op: Lt },
            Binary { lhs: Var(6), rhs: Var(7), op: BitXor },
            Unary { src: 8, op: Not },
            Binary { lhs: Var(6), rhs: Var(9), op: BitAnd },
            Count { src: 10 },
            Reduce { src: 8, op: Max },
            Reduce { src: 9, op: Min },
            Unary { src: 9, op: ToU32 },
            Unary { src: 9, op: ToF32 },
            Select { mask: 9, a: Var(2), b: Const(-5.0), ty: I32 },
            SumWhere { src: 2, mask: 9 },
        ],
    });
}

/// The promotions of a `bool` array: `select` for a product, a cast and the operation for sums and
/// differences, with the boolean on either side, an array or a constant, and broadcasting.
#[test]
fn bool_promotions() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 7 },
            Input { dims: [12, 4, 1, 1], ty: I32, seed: 9 },
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 11 },
            Compare { lhs: 0, rhs: Const(0.5), op: Gt },
            Promote { mask: 3, other: Var(0), ty: F32, op: Mul, mask_left: true },
            Promote { mask: 3, other: Var(1), ty: I32, op: Sub, mask_left: false },
            Promote { mask: 3, other: Var(2), ty: U32, op: Add, mask_left: true },
            Promote { mask: 3, other: Const(2.0), ty: F32, op: Mul, mask_left: false },
            Promote { mask: 3, other: Const(3.0), ty: U32, op: Sub, mask_left: false },
        ],
    });
}

/// `x & -1` is `x` but `x & i32::MAX` is not: an `i32` constant that saturates to `i32::MAX`
/// (here `2^32`) was taken for all ones by the simplifier.
#[test]
fn i32_and_with_a_saturating_constant_is_not_the_identity() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: I32, seed: 18312717770150919551 },
            Binary { lhs: Var(0), rhs: Const(4294967300.0), op: BitAnd },
            Binary { lhs: Var(0), rhs: Const(-1.0), op: BitAnd },
        ],
    });
}

/// A kernel whose bindings the shader does not all read: the launch bound all of them, and
/// `create_bind_group` failed with "Number of bindings ... does not match" (repro 1).
#[test]
fn bool_xor_reduce_promote_df32() {
    check(Program {
        config: Config { max_nodes: 2, max_inlined: None, max_live: None, max_inputs: None, assume_busy: false, speculate: true, simplify: true, staging: false, fast_math: false, df_split: false, chunk_bytes: None, memory_budget: None, slab_bytes: None, split: false, file_leaves: false, cost: Test },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: Df32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: Bool, seed: 0 },
            Binary { lhs: Var(1), rhs: Const(31.0), op: BitXor },
            Reduce { src: 2, op: Max },
            Promote { mask: 3, other: Var(0), ty: Df32, op: Add, mask_left: false },
        ],
    });
}

/// The same class of failure through a per-step evaluated loop with a `bool` state (repro 2).
#[test]
fn bool_loop_state_promote_df32() {
    check(Program {
        config: Config { max_nodes: 2, max_inlined: None, max_live: None, max_inputs: None, assume_busy: false, speculate: true, simplify: true, staging: false, fast_math: false, df_split: false, chunk_bytes: None, memory_budget: None, slab_bytes: None, split: false, file_leaves: false, cost: Test },
        math: false,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: Bool, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: I32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: Df32, seed: 0 },
            Compare { lhs: 2, rhs: Var(2), op: Equal },
            Input { dims: [7, 5, 1, 1], ty: U32, seed: 766574886846494210 },
            Unary { src: 2, op: Neg },
            Binary { lhs: Var(4), rhs: Const(5.0), op: Minimum },
            Loop { kind: PerStepEval, steps: 11, states: vec![4], body: vec![Unary { src: 1, op: Abs }, Unary { src: 6, op: Not }, Promote { mask: 3, other: Var(5), ty: Df32, op: Sub, mask_left: true }], next: vec![9] },
        ],
    });
}

/// Finding 17 (found by `fuzz::threads`, only when programs ran at once; deterministic once
/// minimized): `iterate_all` whose step is lazy and reads a reduction of the state (`sum` of a
/// one-element state added to the max). Flattening the step for a register loop evaluated the
/// reduction on the placeholder, which is not the recording and holds unspecified data (a pooled
/// buffer: whatever other threads had in it), and the step then used that value at every step.
#[test]
fn iterate_all_step_reading_a_reduction_of_its_state_computes_it_at_every_step() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 7274293018610354030 },
            AxisReduce { src: 0, axis: 0, op: Max },
            Unary { src: 1, op: ToF32 },
            Loop { kind: IterateAll, steps: 6, states: vec![2], body: vec![Reduce { src: 3, op: Sum }, Binary { lhs: Var(4), rhs: Var(2), op: Minimum }, Range { dims: [8, 8, 1, 1], axis: 0, ty: F32 }], next: vec![5] },
        ],
    });
}

/// Finding 18 (fuzz seed `41ebf540`): `iterate_all` over a state of several buffers whose step
/// returns the state unchanged (here `-(-(-(-x)))`) returned the placeholder, which borrows the
/// state's buffers; `state *= result` then bound one buffer twice.
#[test]
fn iterate_all_over_a_chunked_state_does_not_return_its_placeholder() {
    check(Program {
        config: Config { chunk_bytes: Some(1024), ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Input { dims: [300, 1, 1, 1], ty: Df32, seed: 0 },
            Binary { lhs: Var(0), rhs: Const(-2.0), op: Minimum },
            Loop { kind: IterateAll, steps: 4, states: vec![1], body: vec![Binary { lhs: Const(0.5), rhs: Var(2), op: Mul }, Range { dims: [300, 1, 1, 1], axis: 0, ty: F32 }, Compare { lhs: 4, rhs: Const(2.0), op: Le }, Unary { src: 3, op: Neg }, Binary { lhs: Var(4), rhs: Var(4), op: Minimum }], next: vec![7] },
            Compound { target: 1, op: Mul, rhs: Var(3) },
        ],
    });
}

/// Finding 19 (found by `fuzz::threads`): a kernel that fuses a slice with `trunc` got a slot of
/// the parameter uniform for a view its lowering never applies (the slice of a one-element array
/// is the identity), so the shader declared the uniform without reading it and the launch bound
/// it anyway: `create_bind_group` failed with "Number of bindings (3) does not match ... (2)".
/// Like finding 7f7fa18 (a `bool` to `Df32` cast) it was fatal for every program that shares the
/// context: the error invalidated the batch command buffer, and the other threads' kernels in it
/// never ran. The launch now binds what the built pipeline reads (see `KernelCore`).
#[test]
fn slice_of_a_difference_then_trunc_then_assign_binds_an_unread_uniform() {
    check(Program {
        config: Config::default_engine(),
        math: false,
        stmts: vec![
            Input { dims: [8, 8, 1, 1], ty: F32, seed: 9481416801333318266 },
            View { src: 0, view: Slice { axis: 0, lo: 6, hi: 7 } },
            Loop { kind: PerStepEval, steps: 6, states: vec![1], body: vec![View { src: 1, view: ShiftMirror { offs: vec![1, 2] } }], next: vec![3] },
            View { src: 2, view: Slice { axis: 0, lo: 0, hi: 1 } },
            Binary { lhs: Var(2), rhs: Var(3), op: Sub },
            View { src: 4, view: Slice { axis: 0, lo: 0, hi: 1 } },
            Unary { src: 5, op: Trunc },
            Assign { target: 6, path: vec![], rhs: Const(-2.0) },
        ],
    });
}

/// Fuzz finding of the file-backed leaves: a scan loop over a leaf that is a file array: its segment was written by a pin
/// that `abort_recording` ran inside the pin of the same segment, which then found the segment in memory.
#[test]
fn a_loop_over_a_file_leaf() {
    check(Program {
        config: Config { file_leaves: true, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Loop { kind: IterateUntilAll, steps: 1, states: vec![0], body: vec![Scan { src: 0, axis: 0, op: CumSum }], next: vec![2] },
        ],
    });
}

/// The file-backed leaves are writable: a program's `assign` into its input goes to the file.
#[test]
fn assigning_into_a_file_leaf() {
    check(Program {
        config: Config { file_leaves: true, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 0 },
            Assign { target: 0, path: vec![], rhs: Const(0.0) },
        ],
    });
}

/// `slab_bytes` (W3-P4 fuzz dimension): a kernel that reads a view of a short array, on an output that fits one segment, restarted its
/// grid at every slab of 64 elements (the plain variant has no `base`), so the second slab repeated the first.
#[test]
fn a_broadcast_view_over_slabs_of_one_segment_keeps_its_index() {
    check(Program {
        config: Config { chunk_bytes: Some(1024), slab_bytes: Some(256), file_leaves: true, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            View { src: 1, view: Permute { perm: [2, 1, 3, 0] } },
            Binary { lhs: Var(0), rhs: Const(0.0), op: Minimum },
            Binary { lhs: Var(2), rhs: Var(3), op: Maximum },
        ],
    });
}

#[test]
fn an_iterate_all_stencil_with_pinned_slabs() {
    check(Program {
        config: Config { slab_bytes: Some(256), ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [7, 5, 1, 1], ty: F32, seed: 0 },
            Input { dims: [7, 5, 1, 1], ty: I32, seed: 0 },
            View { src: 0, view: Pad { widths: vec![(0, 0), (0, 0)], mode: Edge } },
            Loop { kind: IterateAll, steps: 4, states: vec![1], body: vec![Compare { lhs: 1, rhs: Const(2.0), op: Equal }, View { src: 3, view: ShiftClamp { offs: vec![0, -4] } }, Scan { src: 2, axis: 0, op: CumSum }], next: vec![5] },
            Put { target: 3, idx: vec![9, 6, 1, 26, 12, 27, 2, 11, 7, 17, 37, 23, 20, 29, 16, 18, 8, 33, 3, 31, 4, 24, 36, 0, 30, 5, 32, 13, 15, 28, 34, 22, 35, 10, 21], vals: Var(3) },
        ],
    });
}

/// Fuzz finding of the split dimension: `iterate_all` over an array of several segments (steps run on an alias, the probe), then `write` to the array: the loop's
/// result read the new data, because `absorb` passed the probe's reader nodes on to the owner but not the schedules holding its deferred steps.
#[test]
fn a_write_after_an_iterate_over_segments_commits_the_schedule_of_its_probe() {
    check(Program {
        config: Config { split: true, chunk_bytes: Some(1024), ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [300, 1, 1, 1], ty: F32, seed: 0 },
            Loop { kind: IterateAll, steps: 3, states: vec![0], body: vec![Unary { src: 1, op: Sqrt }], next: vec![2] },
            Write { target: 0, seed: 15014408748156108059 },
        ],
    });
}

/// Fuzz finding of the injected I/O failures (W3 end-of-wave gate, `split_io_failures`; it needs no failure, only file arrays whose segments are away):
/// `iterate_until_all` whose step gathers with an index array that is still in its file, and makes an array it drops. The recording of the step
/// aborted at the upload of the index (the step cannot be recorded); the probe of the state had taken over the buffer of the dropped array, and
/// the seed copy into it came before the aborted commands, which wrote the dropped array's value over it.
#[test]
fn an_iterate_until_whose_recording_aborts_keeps_the_state_a_released_buffer_was_handed_to() {
    check(Program {
        config: Config { max_nodes: 2, max_inputs: Some(8), file_leaves: true, ..Config::default_engine() },
        math: false,
        stmts: vec![
            Full { dims: [9, 6, 1, 1], ty: F32, value: 3.0 },
            Input { dims: [9, 6, 1, 1], ty: U32, seed: 0 },
            Loop { kind: IterateUntilAll, steps: 1, states: vec![0], body: vec![View { src: 2, view: Take { idx: 1 } }, Full { dims: [7, 5, 1, 1], ty: U32, value: 5.0 }], next: vec![3] },
        ],
    });
}

/// Fuzz finding of the injected I/O failures (W3 end-of-wave gate, `split_io_failures`): `iterate_until_all` over a file array whose first read fails. The
/// recording of the step aborts at the first launch, which copies the seed (the read that fails); the fused launch of the step and its record then
/// failed, the error was taken for a limit of the fusion, and the separate launches ran (the failed read was spent) on a probe that held nothing.
#[test]
fn an_iterate_until_over_a_file_array_whose_first_read_fails_does_not_run_on_an_unseeded_probe() {
    use luma_io::{FailPlan, Fault};
    let program = Program {
        config: Config { max_nodes: 2, max_inlined: Some(16), max_live: Some(8), max_inputs: Some(4), ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: I32, seed: 6158030500229569234 },
            Input { dims: [12, 1, 1, 1], ty: Df32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Loop { kind: IterateUntilAll, steps: 1, states: vec![2], body: vec![Binary { lhs: Var(2), rhs: Var(3), op: Minimum }], next: vec![4] },
        ],
    };
    let plan = FailPlan { read: Some((0, Fault::Fail(std::io::ErrorKind::Other))), write: None };
    if let Err(failure) = super::run_with_failure(&program, plan) {
        panic!("{failure}\n{program:?}");
    }
}

/// OPEN (found by `split_io_failures`, seed 13, in a run without an injected failure; no file arrays, no split): with `chunk_bytes` and `slab_bytes` set, variable 4 (the
/// `Concat` of a `Df32` array with itself six times, after `Assign`s of a `Df32` array to another and a `Compound` on it) has element 140 -4 on luma, -3.375 on the oracle.
#[test]
#[ignore = "open: slab and chunk configuration, Df32 concat"]
fn a_concat_assign_and_compound_of_df32_arrays_with_slabs_and_chunks_is_wrong() {
    let program = Program {
        config: Config { max_nodes: 64, max_inlined: Some(16), max_live: Some(8), max_inputs: None, chunk_bytes: Some(1024), slab_bytes: Some(4096), ..Config::default_engine() },
        math: false,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: Df32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: Df32, seed: 0 },
            Input { dims: [7, 5, 1, 1], ty: Df32, seed: 0 },
            View { src: 0, view: Concat { axis: 0, others: vec![0] } },
            Assign { target: 0, path: vec![], rhs: Var(2) },
            View { src: 2, view: Concat { axis: 0, others: vec![2, 2, 2, 2, 2] } },
            Assign { target: 0, path: vec![], rhs: Var(2) },
            Compound { target: 0, op: Sub, rhs: Const(-0.5) },
            Input { dims: [9, 6, 1, 1], ty: Bool, seed: 6187493582823668260 },
            Scan { src: 1, axis: 0, op: CumSum },
            Select { mask: 5, a: Const(2147483600.0), b: Const(1.0), ty: I32 },
            Assign { target: 7, path: vec![], rhs: Const(-2147483600.0) },
            Unary { src: 2, op: Neg },
        ],
    };
    check(program);
}

/// OPEN (found by `split_io_failures`, seed 20): the same cause as the test above, by another path. `iterate_until_all` over two states whose step gathers with a file array;
/// the abort of the step's recording fails at its seed copy (the read that fails), the node being evaluated is left pending (`run_store_planned` under `force_stepped`),
/// and its retry computes on a probe that was never seeded (0 against 0.405). A failed seed copy of an aborted recording needs to poison the probes (or the
/// recording keep its seeds to retry), not only the fused path (see `force_with_record`).
#[test]
#[ignore = "open: failed seed copy of an aborted recording, evaluate_group retry"]
fn an_iterate_until_over_two_states_whose_seed_copy_fails_does_not_run_on_unseeded_probes() {
    use luma_io::{FailPlan, Fault};
    let program = Program {
        config: Config { max_nodes: 8, max_inputs: Some(4), ..Config::default_engine() },
        math: true,
        stmts: vec![
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: U32, seed: 0 },
            Input { dims: [12, 1, 1, 1], ty: F32, seed: 0 },
            Loop { kind: IterateUntilAll, steps: 1, states: vec![0, 2], body: vec![View { src: 2, view: Take { idx: 1 } }, Unary { src: 3, op: Log1p }], next: vec![6, 5] },
        ],
    };
    let plan = FailPlan { read: Some((0, Fault::Short { keep: 8 })), write: None };
    if let Err(failure) = super::run_with_failure(&program, plan) {
        panic!("{failure}\n{program:?}");
    }
}

/// Finding of the end-of-wave 3 gate (a wgpu fuzz run hung): the thread's open schedule of a split context over a file leaf was committed when the last
/// outside handle of the context dropped only if the handles left were its segments' (it did not count the file's store, which holds one too). It never
/// committed otherwise, and the thread kept the context, its arrays and, on wgpu, its device until it ended: 64 such contexts used up the driver's
/// devices and every later context waited for one that never came back.
const FILE_LEAF_SCHEDULE: fn() -> Program = || Program {
    config: Config { split: true, file_leaves: true, ..Config::default_engine() },
    math: false,
    stmts: vec![
        Input { dims: [300, 1, 1, 1], ty: F32, seed: 16878225131511299228 },
        Binary { lhs: Var(0), rhs: Var(0), op: Minimum },
        Binary { lhs: Var(1), rhs: Var(1), op: Sub },
        Assign { target: 2, path: vec![], rhs: Var(2) },
    ],
};

/// `FILE_LEAF_SCHEDULE` on a fresh context: whether the context is gone once the program is.
fn context_is_released_after(program: &Program) -> bool {
    let ctx = super::engine(&program.config).expect("a context opens (on wgpu: a device came back to the pool)");
    let weak = ctx.downgrade();
    if let Err(failure) = super::run_on(ctx, program, super::Filter::Off) {
        panic!("{failure}\n{program:?}");
    }
    weak.upgrade().is_none()
}

#[test]
fn a_schedule_over_a_file_leaf_does_not_keep_its_context_alive() {
    assert!(context_is_released_after(&FILE_LEAF_SCHEDULE()));
}

/// More contexts than the driver has devices (63 or 64 on NVIDIA's Linux driver), made and dropped by several threads at once.
#[test]
fn many_threads_making_and_dropping_contexts_over_file_leaves_keep_getting_devices() {
    const THREADS: usize = 4;
    const PER_THREAD: usize = 24;
    let program = FILE_LEAF_SCHEDULE();
    let leaked: usize = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS).map(|_| scope.spawn(|| (0..PER_THREAD).filter(|_| !context_is_released_after(&program)).count())).collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).sum()
    });
    assert_eq!(leaked, 0, "contexts stayed alive after their program ended");
}
