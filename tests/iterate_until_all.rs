//! `iterate_until_all` replays a traced step and evaluates its stop rule on the GPU between steps;
//! the results match the same loop written on the host.

mod common;

use common::{assert_close, values};
use luma::*;

type One = (Array<f32>,);
type Two = (Array<f32>, Array<f32>);

/// Recordings are per context: every test has a context of its own, so that tests running in
/// parallel do not make each other fall back to the host loop.
fn ones(ctx: &Context, n: usize) -> Array<f32> {
    Array::from_slice_on(ctx, &vec![1.0f32; n], n).unwrap()
}

fn halving(max_steps: usize, tolerance: f32) -> (Run<One>, usize) {
    let ctx = Context::new().unwrap();
    let calls = std::cell::Cell::new(0);
    let run = iterate_until_all(
        (ones(&ctx, 100),),
        Until::new(max_steps)
            .record("max", |s: &One| s.0.max())
            .stop_when(move |h| h["max"].last().le(tolerance)),
        |s, _| {
            calls.set(calls.get() + 1);
            (&s.0 * 0.5,)
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 1, "the step is traced once and replayed");
    let steps = run.steps;
    (run, steps)
}

#[test]
fn stops_at_the_first_step_whose_condition_holds() {
    let (run, steps) = halving(100, 1e-3);
    assert_eq!(steps, 10);
    assert_eq!(run.reason, StopReason::Predicate(0));
    let history = run.history("max").unwrap();
    let want: Vec<f32> = (1..=10).map(|k| 0.5f32.powi(k)).collect();
    assert_close(history, &want, 1e-6, 0.0);
    assert_close(&run.state.0.to_vec().unwrap(), &[0.5f32.powi(10); 100], 1e-6, 0.0);
    assert!(run.history("nothing").is_none());
}

#[test]
fn max_steps_bounds_the_loop() {
    let (run, steps) = halving(4, 1e-3);
    assert_eq!(steps, 4);
    assert_eq!(run.reason, StopReason::MaxSteps);
    assert_close(run.history("max").unwrap(), &[0.5, 0.25, 0.125, 0.0625], 1e-6, 0.0);
    assert_close(&run.state.0.to_vec().unwrap(), &[0.0625; 100], 1e-6, 0.0);
}

#[test]
fn the_condition_may_hold_in_a_later_block() {
    let ctx = Context::new().unwrap();
    let run = iterate_until_all(
        (ones(&ctx, 100),),
        Until::new(100)
            .record("max", |s: &One| s.0.max())
            .stop_when(|h| h["max"].last().le(1e-6))
            .check_every(8),
        |s, _| (&s.0 * 0.5,),
    )
    .unwrap();
    assert_eq!(run.steps, 20);
    assert_eq!(run.reason, StopReason::Predicate(0));
    assert_eq!(run.history("max").unwrap().len(), 20);
    assert_close(&run.state.0.to_vec().unwrap(), &[0.5f32.powi(20); 100], 1e-6, 0.0);
}

#[test]
fn the_block_schedule_does_not_change_the_result() {
    // Counting 1, 2, 3, ... is exact in f32; the stop at 300 crosses the blocks of 16, 32, 64 and 128.
    let run_with = |every: Option<usize>| {
        let ctx = Context::new().unwrap();
        let mut until = Until::new(1000)
            .record("v", |s: &One| s.0.max())
            .stop_when(|h| h["v"].last().ge(300.0));
        if let Some(n) = every {
            until = until.check_every(n);
        }
        iterate_until_all((Array::from_slice_on(&ctx, &[0.0f32; 10], 10).unwrap(),), until, |s, _| (&s.0 + 1.0,)).unwrap()
    };
    let mut host = 0.0f32;
    let mut want = Vec::new();
    while host < 300.0 {
        host += 1.0;
        want.push(host);
    }
    for every in [None, Some(7), Some(1000)] {
        let run = run_with(every);
        assert_eq!(run.steps, 300, "{every:?}");
        assert_eq!(run.reason, StopReason::Predicate(0), "{every:?}");
        assert_eq!(run.history("v").unwrap(), &want[..], "{every:?}");
        assert_eq!(run.state.0.to_vec().unwrap(), vec![300.0f32; 10], "{every:?}");
    }
}

#[test]
fn the_step_gets_its_index() {
    let ctx = Context::new().unwrap();
    let zero = Array::<f32>::zeros_on(&ctx, 8).unwrap();
    let run = iterate_until_all((zero,), Until::new(5), |s, k| (&s.0 + k,)).unwrap();
    assert_eq!((run.steps, run.reason), (5, StopReason::MaxSteps));
    // 0 + 1 + 2 + 3 + 4
    assert_eq!(run.state.0.to_vec().unwrap(), vec![10.0; 8]);
}

#[test]
fn zero_steps_return_the_initial_state() {
    let ctx = Context::new().unwrap();
    let run = iterate_until_all((ones(&ctx, 4),), Until::new(0).record("max", |s: &One| s.0.max()), |s, _| (&s.0 * 0.5,)).unwrap();
    assert_eq!(run.steps, 0);
    assert_eq!(run.state.0.to_vec().unwrap(), vec![1.0; 4]);
    assert_eq!(run.history("max"), Some(&[][..]));
}

fn two_step_host(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = a.len();
    let na = (0..n).map(|i| a[i] * 0.9 + b[(i + n - 1) % n] * 0.1).collect();
    let nb = (0..n).map(|i| b[i] * 0.95 + a[i] * 0.02).collect();
    (na, nb)
}

fn two_step(s: &Two, _: &Array<f32>) -> Two {
    (&s.0 * 0.9 + s.1.shift(1) * 0.1, &s.1 * 0.95 + &s.0 * 0.02)
}

#[test]
fn two_arrays_with_a_shift_and_composite_conditions_match_a_host_loop() {
    let n = 64;
    let (a0, b0) = (values(n, 31, 0.5, 1.5), values(n, 32, 0.5, 1.5));
    let ctx = Context::new().unwrap();
    let init = (Array::from_slice_on(&ctx, &a0, n).unwrap(), Array::from_slice_on(&ctx, &b0, n).unwrap());
    let until = Until::new(300)
        .record("sum", |s: &Two| s.0.sum())
        .record("norm", |s: &Two| s.1.norm())
        .stop_when(|h| (h["sum"].peak() / h["sum"].last()).ge(10.0))
        .stop_when(|h| {
            (h["norm"].window_max(3) - h["norm"].window_min(3)).abs().lt(1e-9).and(h.step().gt(5.0))
        });
    let calls = std::cell::Cell::new(0);
    let run = iterate_until_all(init, until, |s, k| {
        calls.set(calls.get() + 1);
        two_step(s, k)
    })
    .unwrap();
    assert_eq!(calls.get(), 1, "the step is traced once and replayed");

    // The same loop and the same rule on the host.
    let (mut a, mut b) = (a0.clone(), b0.clone());
    let (mut sums, mut norms) = (Vec::new(), Vec::new());
    let mut reason = StopReason::MaxSteps;
    let mut steps = 0;
    for t in 0..300 {
        (a, b) = two_step_host(&a, &b);
        sums.push(a.iter().sum::<f32>());
        norms.push(b.iter().map(|v| v * v).sum::<f32>().sqrt());
        steps = t + 1;
        let peak = sums.iter().copied().fold(f32::MIN, f32::max);
        if peak / sums[t] >= 10.0 {
            reason = StopReason::Predicate(0);
            break;
        }
        let window = &norms[norms.len().saturating_sub(3)..];
        let spread = window.iter().copied().fold(f32::MIN, f32::max) - window.iter().copied().fold(f32::MAX, f32::min);
        if spread.abs() < 1e-9 && steps as f32 > 5.0 {
            reason = StopReason::Predicate(1);
            break;
        }
    }
    assert_eq!((run.steps, run.reason), (steps, reason));
    assert!(steps > 10 && steps < 300, "the rule should fire part-way ({steps} steps)");
    assert_close(run.history("sum").unwrap(), &sums, 1e-3, 1e-6);
    assert_close(run.history("norm").unwrap(), &norms, 1e-3, 1e-6);
    assert_close(&run.state.0.to_vec().unwrap(), &a, 1e-3, 1e-6);
    assert_close(&run.state.1.to_vec().unwrap(), &b, 1e-3, 1e-6);
}

#[test]
fn the_host_fallback_agrees_with_the_gated_loop() {
    let n = 64;
    let (a0, b0) = (values(n, 33, 0.5, 1.5), values(n, 34, 0.5, 1.5));
    let ctx = Context::new().unwrap();
    let make = || (Array::from_slice_on(&ctx, &a0, n).unwrap(), Array::from_slice_on(&ctx, &b0, n).unwrap());
    let until = || {
        Until::new(300)
            .record("sum", |s: &Two| s.0.sum())
            .stop_when(|h| (h["sum"].peak() / h["sum"].last()).ge(20.0))
    };
    let gated = iterate_until_all(make(), until(), two_step).unwrap();
    // Reading a value back inside the step cannot be recorded: the loop runs on the host.
    let read_back = iterate_until_all(make(), until(), |s: &Two, k| {
        let _ = s.0.max().item().unwrap();
        two_step(s, k)
    })
    .unwrap();
    assert_eq!((gated.steps, gated.reason), (read_back.steps, read_back.reason));
    assert_close(gated.history("sum").unwrap(), read_back.history("sum").unwrap(), 1e-4, 0.0);
    assert_close(&gated.state.1.to_vec().unwrap(), &read_back.state.1.to_vec().unwrap(), 1e-4, 1e-6);
}

#[test]
fn states_split_over_several_buffers_run_on_the_host() {
    let ctx = Context::with_options(ContextOptions::default().chunk_bytes(1024)).unwrap();
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; 1000], 1000).unwrap();
    let run = iterate_until_all(
        (x,),
        Until::new(50).record("max", |s: &One| s.0.max()).stop_when(|h| h["max"].last().le(1e-2)),
        |s, _| (&s.0 * 0.5,),
    )
    .unwrap();
    assert_eq!((run.steps, run.reason), (7, StopReason::Predicate(0)));
    assert_close(&run.state.0.to_vec().unwrap(), &[0.5f32.powi(7); 1000], 1e-6, 0.0);
}

#[test]
fn a_record_of_several_elements_is_an_error() {
    let ctx = Context::new().unwrap();
    let result = iterate_until_all(
        (ones(&ctx, 10),),
        Until::new(5).record("bad", |s: &One| s.0.slice(0, ..2)),
        |s, _| (&s.0 * 0.5,),
    );
    let error = result.err().unwrap();
    assert!(matches!(error.root(), LumaError::InvalidShape(_)), "{error}");
}

#[test]
fn a_step_that_assigns_into_its_state_leaves_the_initial_state_alone() {
    let n = 16;
    let ctx = Context::new().unwrap();
    let x0 = values(n, 35, 0.5, 1.5);
    let x = Array::from_slice_on(&ctx, &x0, n).unwrap();
    let step = |s: &One, _: &Array<f32>| {
        let mut c = s.0.clone();
        c.slice_mut(0, ..1).assign(0.0).unwrap();
        (&c * 0.5 + 1.0,)
    };
    let run = iterate_until_all(
        (Array::from_slice_on(&ctx, &x0, n).unwrap(),),
        Until::new(6).record("max", |s: &One| s.0.max()),
        step,
    )
    .unwrap();
    assert_eq!(x.to_vec().unwrap(), x0);

    let mut want = &x * 1.0;
    let mut maxes = Vec::new();
    for k in 0..6 {
        want = step(&(want,), &x).0;
        maxes.push(want.max().item().unwrap());
        assert_eq!(run.history("max").unwrap().len(), 6, "{k}");
    }
    assert_close(&run.state.0.to_vec().unwrap(), &want.to_vec().unwrap(), 1e-5, 1e-6);
    assert_close(run.history("max").unwrap(), &maxes, 1e-5, 1e-6);
}

#[test]
fn constants_may_be_infinite() {
    let ctx = Context::new().unwrap();
    let run = iterate_until_all(
        (ones(&ctx, 8),),
        Until::new(20)
            .record("max", |s: &One| s.0.max())
            .stop_when(|h| h["max"].last().ge(f32::INFINITY))
            .stop_when(|h| h["max"].trough().le(0.01)),
        |s, _| (&s.0 * 0.5,),
    )
    .unwrap();
    assert_eq!((run.steps, run.reason), (7, StopReason::Predicate(1)));
}

#[test]
fn a_nan_state_never_satisfies_a_condition() {
    let ctx = Context::new().unwrap();
    let mut data = [1.0f32; 8];
    data[5] = f32::NAN;
    let init = (Array::from_slice_on(&ctx, &data, 8).unwrap(),);
    let run = iterate_until_all(
        init,
        Until::new(50)
            .record("sum", |s: &One| s.0.sum())
            .stop_when(|h| h["sum"].last().le(1e30))
            .stop_when(|h| h["sum"].peak().le(1e30))
            .stop_when(|h| h["sum"].trough().le(1e30))
            .stop_when(|h| h["sum"].window_max(4).ge(-1e30)),
        |s, _| (&s.0 * 0.5,),
    )
    .unwrap();
    assert_eq!((run.steps, run.reason), (50, StopReason::MaxSteps));
    assert!(run.history("sum").unwrap().iter().all(|v| v.is_nan()));
}

#[test]
fn zero_steps_return_a_copy_that_does_not_alias_the_input() {
    let ctx = Context::new().unwrap();
    let x = ones(&ctx, 4);
    let mut run = iterate_until_all((x.clone(),), Until::new(0), |s, _| (&s.0 * 0.5,)).unwrap();
    run.state.0 += 1.0;
    assert_eq!(run.state.0.to_vec().unwrap(), vec![2.0; 4]);
    assert_eq!(x.to_vec().unwrap(), vec![1.0; 4]);
}
