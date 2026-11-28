//! Epilogues on axis reductions (W3-U4 D1): the elementwise consumers of a reduction's result are applied by its last pass as it
//! stores each line's value. Every case is compared bit for bit with the unfused result (the reduction recorded first, the
//! operations then running as a kernel of their own).

use crate::common::values;
use luma::{Array, Context, Decision, DecisionSite};

/// The shapes that reach every kind of last pass: short contiguous lines, long contiguous ones (chunks, then a second pass),
/// strided lines walked by one invocation each, and strided lines blocked into several passes.
fn shapes(ctx: &Context) -> Vec<([usize; 2], usize)> {
    let big = if ctx.is_cpu() { 1usize << 16 } else { 1 << 18 };
    vec![([32, big / 8], 0), ([big / 4, 64], 0), ([64, big], 1), ([big / 8, 40], 1)]
}

fn result_dims(dims: [usize; 2], axis: usize) -> [usize; 2] {
    let mut out = dims;
    out[axis] = 1;
    out
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn epilogue_decisions(decisions: &[Decision]) -> Vec<String> {
    decisions.iter().filter(|d| d.site == DecisionSite::Epilogue).map(|d| d.chosen.to_string()).collect()
}

/// What `consume` builds from a reduction `reduce` of `x` along an axis (and from `scale` and `other`, arrays of the result's shape),
/// with its outputs read: once on the reduction as it waits, which absorbs the operations, and once on the reduction recorded first,
/// which cannot. The two must hold the same bits; `expect` judges the epilogue decisions of the first.
fn check(
    name: &str,
    reduce: impl Fn(&Array<f32>, usize) -> Array<f32>,
    consume: impl Fn(&Array<f32>, &Array<f32>, &Array<f32>) -> Vec<Array<f32>>,
    expect: impl Fn(&[String]) -> bool,
) {
    let ctx = Context::new().unwrap();
    for (dims, axis) in shapes(&ctx) {
        let len = dims[0] * dims[1];
        let x = Array::from_slice_on(&ctx, &values(len, 11, -1.0, 1.0), dims).unwrap();
        let rd = result_dims(dims, axis);
        let scale = Array::from_slice_on(&ctx, &values(rd[0] * rd[1], 12, 0.5, 1.5), rd).unwrap();
        let other = Array::from_slice_on(&ctx, &values(rd[0] * rd[1], 13, -2.0, 2.0), rd).unwrap();
        for array in [&x, &scale, &other] {
            array.eval().unwrap();
        }
        ctx.flush().unwrap();
        ctx.record_decisions(true);
        let outs = consume(&reduce(&x, axis), &scale, &other);
        let fused: Vec<Vec<u32>> = outs.iter().map(|out| bits(&out.to_vec().unwrap())).collect();
        let log = epilogue_decisions(&ctx.take_decisions());
        ctx.record_decisions(false);
        assert!(expect(&log), "{name} {dims:?} axis {axis}: {log:?}");
        let first = reduce(&x, axis);
        first.eval().unwrap();
        ctx.flush().unwrap();
        ctx.record_decisions(true);
        let unfused: Vec<Vec<u32>> = consume(&first, &scale, &other).iter().map(|out| bits(&out.to_vec().unwrap())).collect();
        let log = epilogue_decisions(&ctx.take_decisions());
        ctx.record_decisions(false);
        assert!(log.iter().all(|chosen| !chosen.starts_with("absorbed")), "{name}: a recorded reduction absorbed {log:?}");
        assert_eq!(fused, unfused, "{name} {dims:?} axis {axis}");
    }
}

/// An expression of `x` of enough operations per cell that the CPU backend leaves its reduction pending too.
fn heavy(x: &Array<f32>) -> Array<f32> {
    let mut v = x * 1.0;
    for _ in 0..8 {
        v = &v * 1.0001 + 0.0001;
    }
    v
}

fn squares_sum(x: &Array<f32>, axis: usize) -> Array<f32> {
    (&heavy(x) * x).sum_axis(axis)
}

fn plain_sum(x: &Array<f32>, axis: usize) -> Array<f32> {
    heavy(x).sum_axis(axis)
}

fn absorbed_ops(log: &[String]) -> Option<usize> {
    log.iter().filter_map(|chosen| chosen.strip_prefix("absorbed ")?.strip_suffix(" ops")?.parse().ok()).max()
}

#[test]
fn the_root_of_the_mean_square_is_applied_by_the_last_pass() {
    check("rms", squares_sum, |s, _, _| vec![(s / 64.0).sqrt()], |log| absorbed_ops(log) == Some(2));
}

#[test]
fn a_scale_per_line_is_applied_by_the_last_pass() {
    check("scale", plain_sum, |s, scale, _| vec![s * scale], |log| absorbed_ops(log) == Some(1));
}

#[test]
fn the_reduction_may_be_the_right_operand() {
    check("rsub", plain_sum, |s, scale, _| vec![scale - s], |log| absorbed_ops(log) == Some(1));
}

#[test]
fn a_chain_with_inputs_and_scalars_is_one_program() {
    check(
        "chain",
        squares_sum,
        |s, scale, other| vec![(((s * scale + other) * 0.25).tanh() + 1.0).exp()],
        |log| absorbed_ops(log) == Some(6),
    );
}

#[test]
fn the_reduction_and_what_is_made_of_it_are_both_read() {
    check("both", plain_sum, |s, scale, _| vec![s.clone(), s * scale, s * 0.5], |log| absorbed_ops(log) == Some(1));
}

#[test]
fn two_reductions_of_the_same_lines_each_apply_their_own_epilogue() {
    let ctx = Context::new().unwrap();
    let dims = [64, if ctx.is_cpu() { 1usize << 16 } else { 1 << 18 }];
    let x = Array::from_slice_on(&ctx, &values(dims[0] * dims[1], 21, -1.0, 1.0), dims).unwrap();
    x.eval().unwrap();
    ctx.flush().unwrap();
    let run = |first: bool| {
        let sum = plain_sum(&x, 1);
        let squares = squares_sum(&x, 1);
        if first {
            sum.eval().unwrap();
            squares.eval().unwrap();
            ctx.flush().unwrap();
        }
        let mean = &sum / dims[1] as f32;
        let rms = (&squares / dims[1] as f32).sqrt();
        (bits(&mean.to_vec().unwrap()), bits(&rms.to_vec().unwrap()))
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn more_operations_than_the_epilogue_holds_run_on_their_own() {
    check(
        "limit",
        plain_sum,
        |s, _, _| {
            let mut v = s.clone();
            for _ in 0..20 {
                v = v.tanh();
            }
            vec![v]
        },
        |log| absorbed_ops(log) == Some(16) && log.iter().any(|chosen| chosen == "separate: limit"),
    );
}

#[test]
fn an_operand_of_another_shape_is_a_node_of_its_own() {
    let ctx = Context::new().unwrap();
    let dims = [64, if ctx.is_cpu() { 1usize << 16 } else { 1 << 18 }];
    let x = Array::from_slice_on(&ctx, &values(dims[0] * dims[1], 31, -1.0, 1.0), dims).unwrap();
    x.eval().unwrap();
    ctx.flush().unwrap();
    // The reduction's result is [64, 1]; the other operand broadcasts it to [64, 3].
    let wide = Array::from_slice_on(&ctx, &values(64 * 3, 32, -1.0, 1.0), [64, 3]).unwrap();
    wide.eval().unwrap();
    ctx.record_decisions(true);
    let out = (&plain_sum(&x, 1) * &wide).to_vec().unwrap();
    let log = epilogue_decisions(&ctx.take_decisions());
    ctx.record_decisions(false);
    assert_eq!(out.len(), 64 * 3);
    assert!(absorbed_ops(&log).is_none(), "{log:?}");
}

#[test]
fn a_write_to_a_scale_after_creation_does_not_change_the_value() {
    let ctx = Context::new().unwrap();
    let dims = [64, if ctx.is_cpu() { 1usize << 16 } else { 1 << 18 }];
    let x = Array::from_slice_on(&ctx, &values(dims[0] * dims[1], 41, -1.0, 1.0), dims).unwrap();
    let old = values(64, 42, 0.5, 1.5);
    let mut scale = Array::from_slice_on(&ctx, &old, [64, 1]).unwrap();
    x.eval().unwrap();
    scale.eval().unwrap();
    ctx.flush().unwrap();
    let scaled = &plain_sum(&x, 1) * &scale;
    scale.write(&values(64, 43, 5.0, 6.0)).unwrap();
    let reference = Array::from_slice_on(&ctx, &old, [64, 1]).unwrap();
    let sum = plain_sum(&x, 1);
    sum.eval().unwrap();
    assert_eq!(bits(&scaled.to_vec().unwrap()), bits(&(&sum * &reference).to_vec().unwrap()));
}
