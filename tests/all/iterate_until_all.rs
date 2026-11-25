//! `iterate_until_all` replays a traced step and evaluates its stop rule on the GPU between steps;
//! the results match the same loop written on the host.

use crate::common;

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
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024)).unwrap();
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

/// Where a run of the counting loop starts.
#[derive(Clone, Copy)]
enum Start {
    Value(f32),
    /// The state the run before returned.
    Previous,
}

/// What a run of the counting loop gives: steps, reason, the history of `v` and the state.
type Outcome = (usize, StopReason, Vec<f32>, f32);

/// The runs below, each with the most steps it may take (`None`: all the loop was compiled for).
const COUNTING_RUNS: [(Start, Option<usize>); 7] = [
    // Stops at step 20, in the third block of 7 steps.
    (Start::Value(0.0), None),
    // Stops at step 14, on the last step of a block of 7.
    (Start::Value(100.0), None),
    // Ends at 23 steps, a block that is cut short.
    (Start::Value(-100.0), Some(23)),
    // Stops at step 25: the blocks of steps 1, 15 and then 9 of 32 on the default schedule.
    (Start::Value(-100.0), None),
    // Never stops: 100 steps.
    (Start::Value(-10000.0), None),
    // Continues from the state of the run before (an even number of steps).
    (Start::Previous, None),
    (Start::Previous, Some(3)),
];

/// The counting loop on the host: step `k` adds `k`, so every number is an integer well inside
/// what `f32` holds exactly, and the loop stops once the state reaches 190.
fn counting_on_the_host(runs: &[(Start, Option<usize>)], compiled_steps: usize) -> Vec<Outcome> {
    let mut previous = 0.0f32;
    runs.iter()
        .map(|&(start, limit)| {
            let mut x = match start {
                Start::Value(v) => v,
                Start::Previous => previous,
            };
            let mut history = Vec::new();
            let mut stopped = None;
            for k in 0..limit.unwrap_or(compiled_steps).min(compiled_steps) {
                x += k as f32;
                history.push(x);
                if x >= 190.0 {
                    stopped = Some(k + 1);
                    break;
                }
            }
            previous = x;
            match stopped {
                Some(steps) => (steps, StopReason::Predicate(0), history, x),
                None => (history.len(), StopReason::MaxSteps, history, x),
            }
        })
        .collect()
}

/// The runs on a loop compiled once for 100 steps, checking the state every `every` steps; with
/// `profiled` they run inside a profiling scope, which replays every launch one by one.
fn counting_compiled(ctx: &Context, every: Option<usize>, profiled: bool) -> Vec<Outcome> {
    let mut until = Until::new(100)
        .record("v", |s: &One| s.0.max())
        .stop_when(|h| h["v"].last().ge(190.0))
        .stop_when(|h| h["v"].peak().ge(1e9));
    if let Some(n) = every {
        until = until.check_every(n);
    }
    let state = |v: f32| (Array::from_slice_on(ctx, &[v; 8], 8).unwrap(),);
    let mut compiled = compile_until_all(&state(0.0), until, |s: &One, k| (&s.0 + k,)).unwrap();
    let mut previous = state(0.0);
    COUNTING_RUNS
        .iter()
        .map(|&(start, limit)| {
            let init = match start {
                Start::Value(v) => state(v),
                Start::Previous => previous.clone(),
            };
            let run = |compiled: &mut CompiledUntil<One>| match limit {
                Some(n) => compiled.run_steps(init, n),
                None => compiled.run(init),
            };
            let run = if profiled {
                ctx.profile(|| run(&mut compiled)).unwrap().0
            } else {
                run(&mut compiled)
            }
            .unwrap();
            let values = run.state.0.to_vec().unwrap();
            assert!(values.iter().all(|&v| v == values[0]), "the elements of the state differ: {values:?}");
            previous = run.state.clone();
            (run.steps, run.reason, run.history("v").unwrap().to_vec(), values[0])
        })
        .collect()
}

/// Blocks of different lengths, a stop in the middle of one and a run cut short, on every backend
/// that is here: where the backend replays a block as one graph the result is that of the host loop,
/// that of the launches replayed one by one under a profiling scope, and the same for runs of a
/// compiled loop that follow one another and pass their states on.
#[test]
fn blocks_replayed_as_graphs_match_the_host_loop_and_the_launches_replayed_one_by_one() {
    let want = counting_on_the_host(&COUNTING_RUNS, 100);
    assert_eq!(want[0].0, 20);
    assert_eq!(want[1].0, 14);
    assert_eq!((want[2].0, want[2].1), (23, StopReason::MaxSteps));
    assert_eq!(want[3].0, 25);
    assert_eq!((want[4].0, want[4].1), (100, StopReason::MaxSteps));
    let mut ran = 0;
    for kind in [BackendKind::Cuda, BackendKind::Metal, BackendKind::Cpu, BackendKind::Wgpu] {
        let ctx = match Context::with_options(ContextOptions::from_env().backend(kind).calibrate(false)) {
            Ok(ctx) => ctx,
            Err(LumaError::NoAdapter(_) | LumaError::NoDevice(_)) => {
                println!("[iterate_until_all] {kind:?}: not available on this host");
                continue;
            }
            Err(error) => panic!("opening {kind:?}: {error}"),
        };
        ran += 1;
        for every in [None, Some(7), Some(1)] {
            let graphs = counting_compiled(&ctx, every, false);
            assert_eq!(graphs, want, "{kind:?}, every {every:?}");
            if ctx.profiling_supported() {
                let one_by_one = counting_compiled(&ctx, every, true);
                assert_eq!(one_by_one, want, "{kind:?}, every {every:?}, profiled");
            }
        }
    }
    assert!(ran > 0, "no backend to run on");
}

/// A long one-shot loop replays the blocks after its first as repeated graphs of a few steps
/// where the backend has them; stopping inside such a graph, or in a block that does not fill
/// whole ones, gives the same history and state as the host loop.
#[test]
fn a_long_one_shot_loop_stops_at_the_same_step_whatever_the_graph_units() {
    for every in [None, Some(7), Some(50), Some(1000)] {
        for stop in [5.0f32, 37.0, 300.0, 301.0, 999.0, 5000.0] {
            let ctx = Context::new().unwrap();
            let mut until = Until::new(1000).record("v", |s: &One| s.0.max()).stop_when(move |h| h["v"].last().ge(stop));
            if let Some(n) = every {
                until = until.check_every(n);
            }
            let start = Array::from_slice_on(&ctx, &[0.0f32; 33], 33).unwrap();
            let run = iterate_until_all((start,), until, |s, _| (&s.0 + 1.0,)).unwrap();
            let steps = (stop as usize).min(1000);
            let want: Vec<f32> = (1..=steps).map(|k| k as f32).collect();
            assert_eq!(run.steps, steps, "every {every:?}, stop {stop}");
            assert_eq!(run.reason, if stop <= 1000.0 { StopReason::Predicate(0) } else { StopReason::MaxSteps }, "every {every:?}, stop {stop}");
            assert_eq!(run.history("v").unwrap(), &want[..], "every {every:?}, stop {stop}");
            assert_eq!(run.state.0.to_vec().unwrap(), vec![steps as f32; 33], "every {every:?}, stop {stop}");
        }
    }
}
