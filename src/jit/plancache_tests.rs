//! Tests of the plan cache (see `plancache`): evaluations that repeat a structure replay its plan,
//! and what they compute is bit-identical to what the uncached path computes.

use crate::array::Array;
use crate::context::{Context, ContextOptions};
use crate::jit::cost::CostModel;
use crate::ops::eval_all;
use crate::view::PadMode;

type A = Array<f32>;

fn ctx() -> Context {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    ctx.set_cost_model(CostModel::TEST);
    ctx
}

/// A context whose products are all lazy (values of their own that the walk meets).
fn lazy_ctx() -> Context {
    let ctx = Context::with_tuning(ContextOptions::from_env().calibrate(false), |tuning| tuning.lazy = crate::jit::structured::sites::LazyMode::Always).unwrap();
    ctx.set_cost_model(CostModel::TEST);
    ctx
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f32 {
        (self.next() % (1 << 20)) as f32 / (1 << 20) as f32
    }
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

const W: usize = 12;
const H: usize = 9;

/// One operation of a random step: operands are positions in the pool of values made so far.
#[derive(Clone, Copy, Debug)]
enum Step {
    Neg(usize),
    Abs(usize),
    Tanh(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Min(usize, usize),
    /// `a op param`.
    AddScalar(usize, usize),
    MulScalar(usize, usize),
    DivScalar(usize, usize),
    /// A shifted read of `a` (a view of its padded self).
    Shift(usize, usize, PadMode, bool),
    Flip(usize, usize),
    /// `where(a > param, b, c)`.
    Select(usize, usize, usize, usize),
    /// Sum over an axis, broadcast back (a reduction in the step).
    ReduceBroadcast(usize, usize),
}

const MODES: [PadMode; 4] = [PadMode::Edge, PadMode::Mirror, PadMode::Wrap, PadMode::Zero];

/// A random step over `states` state arrays and `params` scalars, with `outputs` results.
#[derive(Clone, Debug)]
struct Program {
    steps: Vec<Step>,
    outputs: Vec<usize>,
}

impl Program {
    fn random(seed: u64, states: usize, reductions: bool) -> Program {
        let mut rng = Rng(seed * 7919 + 13);
        let params = 3;
        // The pool starts with the states and one constant array.
        let mut steps = Vec::new();
        let count = 4 + rng.below(14);
        for size in (states + 1..).take(count) {
            let a = rng.below(size);
            let b = rng.below(size);
            let c = rng.below(size);
            let p = rng.below(params);
            steps.push(match rng.below(if reductions { 14 } else { 13 }) {
                0 => Step::Neg(a),
                1 => Step::Abs(a),
                2 => Step::Tanh(a),
                3 => Step::Add(a, b),
                4 => Step::Sub(a, b),
                5 => Step::Mul(a, b),
                6 => Step::Min(a, b),
                7 => Step::AddScalar(a, p),
                8 => Step::MulScalar(a, p),
                9 => Step::DivScalar(a, p),
                10 => Step::Shift(a, rng.below(2), MODES[rng.below(4)], rng.below(2) == 0),
                11 => Step::Flip(a, rng.below(2)),
                12 => Step::Select(a, b, c, p),
                _ => Step::ReduceBroadcast(a, rng.below(2)),
            });
        }
        let total = states + 1 + steps.len();
        let outputs = (0..states).map(|k| if k == 0 { total - 1 } else { states + 1 + rng.below(steps.len()) }).collect();
        Program { steps, outputs }
    }

    /// The results of the step over `states`, with the scalars `params`.
    fn build(&self, constant: &A, states: &[A], params: &[f32]) -> Vec<A> {
        let mut pool: Vec<A> = states.to_vec();
        pool.push(constant.clone());
        for &step in &self.steps {
            let value = match step {
                Step::Neg(a) => -&pool[a],
                Step::Abs(a) => pool[a].abs(),
                Step::Tanh(a) => pool[a].tanh(),
                Step::Add(a, b) => &pool[a] + &pool[b],
                Step::Sub(a, b) => &pool[a] - &pool[b],
                Step::Mul(a, b) => &pool[a] * &pool[b],
                Step::Min(a, b) => pool[a].minimum(&pool[b]),
                Step::AddScalar(a, p) => &pool[a] + params[p],
                Step::MulScalar(a, p) => &pool[a] * params[p],
                Step::DivScalar(a, p) => &pool[a] / (params[p].abs() + 1.0),
                Step::Shift(a, axis, mode, forward) => {
                    let padded = pool[a].pad([(1, 1), (1, 1)], mode);
                    let (w, h) = (W, H);
                    let (lo0, lo1) = if axis == 0 { (if forward { 2 } else { 0 }, 1) } else { (1, if forward { 2 } else { 0 }) };
                    padded.slice(0, lo0..lo0 + w).slice(1, lo1..lo1 + h)
                }
                Step::Flip(a, axis) => pool[a].flip(axis),
                Step::Select(a, b, c, p) => crate::ops::select(&pool[a].gt(params[p] * 0.5), &pool[b], &pool[c]),
                Step::ReduceBroadcast(a, axis) => {
                    let sums = pool[a].sum_axis(axis);
                    let mut shape = [W, H];
                    shape[axis] = 1;
                    sums.reshape(shape).broadcast_to([W, H])
                }
            };
            pool.push(value);
        }
        self.outputs.iter().map(|&at| pool[at].clamp(-4.0, 4.0)).collect()
    }
}

/// Runs `steps` of `program` eagerly (`eval` every step) with `params(t)` as the scalars of step
/// `t`, and returns the bits of every state after every step.
fn run(ctx: &Context, program: &Program, params: &dyn Fn(usize) -> Vec<f32>, steps: usize, seed: u64) -> Vec<Vec<Vec<u32>>> {
    let constant = A::from_slice_on(ctx, &data(W * H, seed + 1000), [W, H]).unwrap();
    let mut states: Vec<A> = (0..program.outputs.len()).map(|k| A::from_slice_on(ctx, &data(W * H, seed + k as u64), [W, H]).unwrap()).collect();
    let mut seen = Vec::new();
    for t in 0..steps {
        let next = program.build(&constant, &states, &params(t));
        let refs: Vec<&A> = next.iter().collect();
        eval_all(&refs).unwrap();
        states = next;
        seen.push(states.iter().map(|state| bits(&state.to_vec().unwrap())).collect());
    }
    seen
}

fn compare(ctx: &Context, program: &Program, params: &dyn Fn(usize) -> Vec<f32>, steps: usize, seed: u64) -> super::plancache::StatsSnapshot {
    ctx.jit().set_plan_cache(false);
    let before = ctx.jit().plan_stats();
    let want = run(ctx, program, params, steps, seed);
    assert_eq!(ctx.jit().plan_stats(), before, "the cache ran while it was off");
    ctx.jit().set_plan_cache(true);
    let got = run(ctx, program, params, steps, seed);
    for (t, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "step {t} of {program:?} differs from the uncached path");
    }
    let after = ctx.jit().plan_stats();
    super::plancache::StatsSnapshot { hits: after.hits - before.hits, recorded: after.recorded - before.recorded, uncacheable: after.uncacheable - before.uncacheable }
}

#[test]
fn random_programs_with_fixed_scalars_match_the_uncached_path() {
    let ctx = ctx();
    let mut total = super::plancache::StatsSnapshot::default();
    for seed in 0..60 {
        let states = 1 + (seed % 3) as usize;
        let program = Program::random(seed, states, seed % 4 == 0);
        let stats = compare(&ctx, &program, &|_| vec![0.7, -0.3, 1.5], 7, seed);
        total.hits += stats.hits;
        total.recorded += stats.recorded;
        total.uncacheable += stats.uncacheable;
    }
    assert!(total.hits > 20, "plans were replayed {} times, recorded {}, uncacheable {}", total.hits, total.recorded, total.uncacheable);
}

#[test]
fn random_programs_with_changing_scalars_match_the_uncached_path() {
    let ctx = ctx();
    let mut total = super::plancache::StatsSnapshot::default();
    for seed in 100..160 {
        let states = 1 + (seed % 3) as usize;
        let program = Program::random(seed, states, false);
        // The scalars change every step, and now and then two of them are equal, or equal to 0 or 1.
        let params = |t: usize| {
            let mut rng = Rng(seed * 31 + t as u64);
            let mut values: Vec<f32> = (0..3).map(|_| rng.unit() * 3.0 - 1.5).collect();
            match t % 4 {
                1 => values[1] = values[0],
                2 => values[2] = 0.0,
                3 => values[0] = 1.0,
                _ => {}
            }
            values
        };
        let stats = compare(&ctx, &program, &params, 9, seed);
        total.hits += stats.hits;
        total.recorded += stats.recorded;
    }
    assert!(total.hits > 10, "plans were replayed {} times", total.hits);
}

#[test]
fn random_programs_with_special_scalars_match_the_uncached_path() {
    let ctx = ctx();
    for seed in 200..230 {
        let program = Program::random(seed, 2, false);
        let specials = [0.0f32, 1.0, -1.0, f32::MIN_POSITIVE, 1e30, -0.0, 2.0, f32::NAN, f32::INFINITY];
        let params = |t: usize| vec![specials[t % specials.len()], specials[(t * 5 + 1) % specials.len()], 0.25];
        compare(&ctx, &program, &params, 10, seed);
    }
}

/// A stencil step like the heat equation's: views of a padded state.
fn heat(u: &A, alpha: f32) -> A {
    let p = u.pad([(1, 1), (1, 1)], PadMode::Edge);
    let neighbours = p.slice(0, 0..W).slice(1, 1..1 + H) + p.slice(0, 2..2 + W).slice(1, 1..1 + H) + p.slice(0, 1..1 + W).slice(1, 0..H) + p.slice(0, 1..1 + W).slice(1, 2..2 + H);
    u + alpha * (neighbours - 4.0 * u)
}

#[test]
fn an_eager_stencil_loop_replays_one_plan() {
    let ctx = ctx();
    let start = data(W * H, 5);
    ctx.jit().set_plan_cache(false);
    let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
    let mut want = Vec::new();
    for _ in 0..12 {
        u = heat(&u, 0.1);
        u.eval().unwrap();
        want.push(bits(&u.to_vec().unwrap()));
    }
    ctx.jit().set_plan_cache(true);
    let before = ctx.jit().stats();
    let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
    let mut got = Vec::new();
    for _ in 0..12 {
        u = heat(&u, 0.1);
        u.eval().unwrap();
        got.push(bits(&u.to_vec().unwrap()));
    }
    assert_eq!(got, want);
    let stats = ctx.jit().plan_stats();
    assert!(stats.recorded >= 1 && stats.hits >= 8, "{stats:?}");
    // A replay launches what the evaluation launched: one kernel per step.
    let launched = ctx.jit().stats().elementwise_launches - before.elementwise_launches;
    assert!(launched >= 12, "{launched}");
}

#[test]
fn scalars_are_rebound_and_equal_scalars_are_told_from_different_ones() {
    let ctx = ctx();
    let start = data(W * H, 9);
    let run = |on: bool| {
        ctx.jit().set_plan_cache(on);
        let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
        let mut seen = Vec::new();
        for t in 0..16 {
            // Distinct values, then equal ones (they share a uniform slot), then a value that is 0.
            let (a, b) = match t % 6 {
                0..=1 => (0.5 + t as f32, 0.25),
                2..=3 => (0.5, 0.5),
                4 => (0.0, 0.5),
                _ => (2.0, 2.0),
            };
            u = (&u * a + b).clamp(-3.0, 3.0);
            u.eval().unwrap();
            seen.push(bits(&u.to_vec().unwrap()));
        }
        seen
    };
    let want = run(false);
    let got = run(true);
    assert_eq!(got, want);
    assert!(ctx.jit().plan_stats().hits > 0, "{:?}", ctx.jit().plan_stats());
}

#[test]
fn reading_one_buffer_twice_is_another_structure_than_reading_two() {
    let ctx = ctx();
    let a = A::from_slice_on(&ctx, &data(64, 1), 64).unwrap();
    let b = A::from_slice_on(&ctx, &data(64, 2), 64).unwrap();
    let run = |on: bool| {
        ctx.jit().set_plan_cache(on);
        let mut seen = Vec::new();
        for t in 0..10 {
            let sum = if t % 2 == 0 { (&a * &a).tanh() + 1.0 } else { (&a * &b).tanh() + 1.0 };
            sum.eval().unwrap();
            seen.push(bits(&sum.to_vec().unwrap()));
        }
        seen
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn another_array_that_is_pending_keeps_the_plan_from_running() {
    let ctx = ctx();
    let start = data(W * H, 3);
    ctx.jit().set_plan_cache(true);
    let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
    // Alive and pending all the time: the evaluations are not of the whole graph.
    let held = &u * 3.0;
    for _ in 0..8 {
        u = heat(&u, 0.1);
        u.eval().unwrap();
    }
    let stats = ctx.jit().plan_stats();
    assert_eq!(stats.hits, 0, "{stats:?}");
    // And what it computes is right.
    ctx.jit().set_plan_cache(false);
    let mut v = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
    for _ in 0..8 {
        v = heat(&v, 0.1);
        v.eval().unwrap();
    }
    assert_eq!(bits(&u.to_vec().unwrap()), bits(&v.to_vec().unwrap()));
    drop(held);
}

#[test]
fn the_cache_is_bounded() {
    let ctx = ctx();
    ctx.jit().set_plan_cache(true);
    for n in 8..400usize {
        let x = A::from_slice_on(&ctx, &data(n, n as u64), n).unwrap();
        let y = (&x * 2.0 + 1.0).tanh();
        y.eval().unwrap();
        // A different length is a different structure.
        let z = A::from_slice_on(&ctx, &data(n, n as u64), n).unwrap();
        let w = (&z + &x).abs();
        w.eval().unwrap();
    }
    assert!(ctx.jit().plan_entries() <= 128, "{}", ctx.jit().plan_entries());
}

#[test]
fn the_cache_switches_off() {
    let ctx = ctx();
    ctx.jit().set_plan_cache(false);
    let mut u = A::from_slice_on(&ctx, &data(W * H, 4), [W, H]).unwrap();
    for _ in 0..8 {
        u = heat(&u, 0.1);
        u.eval().unwrap();
    }
    assert_eq!(ctx.jit().plan_stats(), Default::default());
    assert_eq!(ctx.jit().plan_entries(), 0);
}

#[test]
fn threads_that_share_a_context_get_what_the_uncached_path_gets() {
    let ctx = ctx();
    ctx.jit().set_plan_cache(false);
    let starts: Vec<Vec<f32>> = (0..4).map(|k| data(W * H, 40 + k)).collect();
    let want: Vec<Vec<u32>> = starts
        .iter()
        .map(|start| {
            let mut u = A::from_slice_on(&ctx, start, [W, H]).unwrap();
            for _ in 0..10 {
                u = heat(&u, 0.1);
                u.eval().unwrap();
            }
            bits(&u.to_vec().unwrap())
        })
        .collect();
    ctx.jit().set_plan_cache(true);
    let got: Vec<Vec<u32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = starts
            .iter()
            .map(|start| {
                let ctx = ctx.clone();
                scope.spawn(move || {
                    let mut u = A::from_slice_on(&ctx, start, [W, H]).unwrap();
                    for _ in 0..10 {
                        u = heat(&u, 0.1);
                        u.eval().unwrap();
                    }
                    bits(&u.to_vec().unwrap())
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    assert_eq!(got, want);
}

#[test]
fn integer_programs_match_the_uncached_path() {
    type U = Array<u32>;
    let ctx = ctx();
    let start: Vec<u32> = (0..W * H).map(|k| (k as u32).wrapping_mul(2654435761)).collect();
    let run = |on: bool| {
        ctx.jit().set_plan_cache(on);
        let mut x = U::from_slice_on(&ctx, &start, [W, H]).unwrap();
        let mut seen = Vec::new();
        for t in 0..12u32 {
            let key = if t % 5 == 4 { 7 } else { t * 3 + 1 };
            let mixed = (&x * 1664525u32 + key) ^ (&x >> 7u32);
            let wrapped = crate::ops::select(&mixed.gt(1u32 << 31), mixed.minimum(key + 100), mixed.maximum(3u32));
            x = wrapped;
            x.eval().unwrap();
            seen.push(x.to_vec().unwrap());
        }
        seen
    };
    assert_eq!(run(true), run(false));
    assert!(ctx.jit().plan_stats().hits > 0, "{:?}", ctx.jit().plan_stats());
}

/// A step whose results have three lengths, two of them read from the first: several launches,
/// one of which reads what another wrote.
fn lengths(u: &A, s: f32) -> [A; 3] {
    let r1 = heat(u, 0.1);
    let r2 = r1.slice(1, 0..1) * s + 1.0;
    let r3 = (r1.slice(0, 0..3) - 0.5).abs();
    [r1, r2, r3]
}

#[test]
fn a_plan_of_several_launches_hands_results_from_one_to_the_next() {
    let ctx = ctx();
    let start = data(W * H, 21);
    let run = |on: bool| {
        ctx.jit().set_plan_cache(on);
        let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
        let mut seen = Vec::new();
        for t in 0..10 {
            let [r1, r2, r3] = lengths(&u, 0.5 + (t % 3) as f32);
            eval_all(&[&r1, &r2, &r3]).unwrap();
            seen.push((bits(&r1.to_vec().unwrap()), bits(&r2.to_vec().unwrap()), bits(&r3.to_vec().unwrap())));
            u = r1;
        }
        seen
    };
    let want = run(false);
    let got = run(true);
    assert_eq!(got, want);
    assert!(ctx.jit().plan_stats().hits >= 5, "{:?}", ctx.jit().plan_stats());
    assert!(ctx.jit().longest_plan() >= 2, "{}", ctx.jit().longest_plan());
}

#[test]
fn programs_cut_by_small_limits_match_the_uncached_path() {
    let ctx = ctx();
    for seed in 300..340 {
        let program = Program::random(seed, 1 + (seed % 2) as usize, false);
        // Few bindings and live values make the planner cut expressions into several kernels.
        ctx.jit().set_max_inputs(2 + (seed % 3) as usize);
        ctx.jit().set_max_live(6 + (seed % 5) as usize);
        compare(&ctx, &program, &|t| vec![0.5 + t as f32 * 0.25, -0.75, 1.25], 8, seed);
    }
    ctx.jit().set_max_inputs(usize::MAX);
    ctx.jit().set_max_live(0);
}

#[test]
fn the_environment_variable_switches_the_cache_off() {
    use super::plancache::enabled_by;
    assert!(enabled_by(None));
    assert!(enabled_by(Some("on")));
    assert!(enabled_by(Some("trace")));
    for off in ["off", "OFF", " 0 ", "false", "No"] {
        assert!(!enabled_by(Some(off)), "{off}");
    }
}

#[test]
fn recording_decisions_without_profiling_logs_the_replay_and_launches_the_same_kernels() {
    use crate::profile::DecisionSite;
    let ctx = ctx();
    let start = data(W * H, 11);
    let run = |record: bool| {
        ctx.set_plan_cache(true);
        ctx.record_decisions(record);
        let before = ctx.jit().stats();
        let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
        for _ in 0..6 {
            u = heat(&u, 0.1);
            u.eval().unwrap();
        }
        ctx.sync().unwrap();
        let launches = ctx.jit().stats().elementwise_launches - before.elementwise_launches;
        let decisions = ctx.take_decisions();
        ctx.record_decisions(false);
        (launches, decisions, bits(&u.to_vec().unwrap()))
    };
    let (off, none, want) = run(false);
    assert!(none.is_empty());
    let (on, decisions, got) = run(true);
    assert_eq!(on, off, "recording must not change the kernels launched");
    assert_eq!(got, want);
    let replays = decisions.iter().filter(|d| d.site == DecisionSite::PlanReplay).count();
    assert!(replays >= 1, "the plan was replayed after the first evaluations: {decisions:?}");
    assert!(decisions.iter().all(|d| !d.chosen.is_empty()));
    assert!(ctx.jit().plan_stats().hits >= 1);
}

#[test]
fn the_plan_cache_replays_a_program_with_a_call() {
    let ctx = ctx();
    let f = ctx.elementwise_fn::<f32>("pc_f", "fn pc_f(x: f32, y: f32) -> f32 { return x * y + 0.5; }").unwrap();
    let start = data(W * H, 21);
    let run = |cached: bool| {
        ctx.jit().set_plan_cache(cached);
        let mut u = A::from_slice_on(&ctx, &start, [W, H]).unwrap();
        let mut seen = Vec::new();
        for t in 0..10 {
            // A scalar that changes every step, a call on it and on the state, a view of the state.
            u = (f.call2(&u, 0.5 + 0.05 * t as f32) * 0.5 + u.flip(0) * 0.25).clamp(-4.0, 4.0);
            u.eval().unwrap();
            seen.push(bits(&u.to_vec().unwrap()));
        }
        seen
    };
    let before = ctx.jit().plan_stats();
    let want = run(false);
    assert_eq!(ctx.jit().plan_stats(), before, "the cache ran while it was off");
    let got = run(true);
    assert_eq!(got, want);
    let after = ctx.jit().plan_stats();
    assert!(after.hits - before.hits >= 3, "{after:?}");
}

// -- structured values: reductions and products in a step (W3-U5) --

/// A step over a one-element state that reads a reduction of an expression (an elementwise program over a buffer and a changing
/// scalar): the reduction is a value of its own that the elementwise program reads.
fn normalise(c: &A, u: &A, t: usize) -> A {
    let total = (c * (1.0 + 0.01 * t as f32)).abs().sum();
    (u * 0.5 + &total * 0.01).tanh()
}

/// Runs `steps` of `step` (eval every time) with the cache `on` or off: the bits of every state, and the reduction launches made.
fn run_steps(ctx: &Context, on: bool, steps: usize, start: &[f32], shape: [usize; 2], step: &dyn Fn(&A, usize) -> A) -> (Vec<Vec<u32>>, u64) {
    ctx.jit().set_plan_cache(on);
    let before = ctx.jit().stats();
    let mut u = A::from_slice_on(ctx, start, shape).unwrap();
    let mut seen = Vec::new();
    for t in 0..steps {
        u = step(&u, t);
        u.eval().unwrap();
        seen.push(bits(&u.to_vec().unwrap()));
    }
    (seen, ctx.jit().stats().reduce_launches - before.reduce_launches)
}

#[test]
fn a_step_with_a_reduction_hits() {
    let ctx = ctx();
    let c = A::from_slice_on(&ctx, &data(W * H, 31), [W, H]).unwrap();
    let step = move |u: &A, t: usize| normalise(&c, u, t);
    let start = data(1, 33);
    let (want, want_launches) = run_steps(&ctx, false, 10, &start, [1, 1], &step);
    let before = ctx.jit().plan_stats();
    let (got, launches) = run_steps(&ctx, true, 10, &start, [1, 1], &step);
    assert_eq!(got, want);
    assert_eq!(launches, want_launches, "a replay launches what the evaluation launched");
    let after = ctx.jit().plan_stats();
    // Where the reduction runs on the host the cache is bypassed (`can_run_on_host`) or the strategy is not replayable.
    if !ctx.can_run_on_host() && !ctx.caps().contains(crate::caps::Caps::HOST_PATH) {
        assert!(after.hits - before.hits >= 6, "{before:?} -> {after:?}");
    }
}

#[test]
fn a_step_with_a_matmul_replays_one_plan() {
    for ctx in [ctx(), lazy_ctx()] {
        let w = A::from_slice_on(&ctx, &data(W * W, 41), [W, W]).unwrap();
        let step = move |u: &A, t: usize| (u.matmul(&w).cbrt() * (1.0 + 0.01 * t as f32) + u * 0.5).clamp(-4.0, 4.0);
        let start = data(W * W, 43);
        let (want, _) = run_steps(&ctx, false, 10, &start, [W, W], &step);
        let before = ctx.jit().plan_stats();
        let (got, _) = run_steps(&ctx, true, 10, &start, [W, W], &step);
        assert_eq!(got, want);
        let after = ctx.jit().plan_stats();
        if !ctx.can_run_on_host() && !ctx.caps().contains(crate::caps::Caps::HOST_PATH) {
            assert!(after.hits - before.hits >= 5, "{before:?} -> {after:?}");
        }
    }
}

#[test]
fn a_product_with_an_absorbed_epilogue_runs_as_ever() {
    // The consumer is absorbed into the product (W3-U4): the step is not cached, and what it computes is the uncached path's.
    let ctx = lazy_ctx();
    let w = A::from_slice_on(&ctx, &data(W * W, 45), [W, W]).unwrap();
    let step = move |u: &A, t: usize| (u.matmul(&w) * (1.0 + 0.01 * t as f32)).clamp(-4.0, 4.0);
    let start = data(W * W, 47);
    let (want, _) = run_steps(&ctx, false, 8, &start, [W, W], &step);
    let (got, _) = run_steps(&ctx, true, 8, &start, [W, W], &step);
    assert_eq!(got, want);
}

#[test]
fn a_step_with_a_reduction_and_a_matmul_and_changing_scalars_matches_the_uncached_path() {
    let ctx = lazy_ctx();
    let w = A::from_slice_on(&ctx, &data(W * W, 51), [W, W]).unwrap();
    let step = move |u: &A, t: usize| {
        let p = u.matmul(&w);
        let total = p.abs().sum();
        ((&p / &(&total + 1.0)) * (0.5 + 0.1 * t as f32) + u * 0.25).clamp(-4.0, 4.0)
    };
    let start = data(W * W, 53);
    let (want, want_launches) = run_steps(&ctx, false, 12, &start, [W, W], &step);
    let (got, launches) = run_steps(&ctx, true, 12, &start, [W, W], &step);
    assert_eq!(got, want);
    assert_eq!(launches, want_launches);
}

#[test]
fn a_changed_shape_class_is_another_plan() {
    let ctx = ctx();
    let before = ctx.jit().plan_stats();
    for (w, h) in [(W, H), (W + 5, H), (W, H + 3)] {
        let c = A::from_slice_on(&ctx, &data(w * h, 61), [w, h]).unwrap();
        let step = move |u: &A, t: usize| normalise(&c, u, t);
        let start = data(1, 63);
        let (got, _) = run_steps(&ctx, true, 6, &start, [1, 1], &step);
        let (want, _) = run_steps(&ctx, false, 6, &start, [1, 1], &step);
        assert_eq!(got, want, "{w}x{h}");
    }
    let after = ctx.jit().plan_stats();
    // Every shape has its own plan: none that was made for another one ran.
    if !ctx.can_run_on_host() && !ctx.caps().contains(crate::caps::Caps::HOST_PATH) {
        assert!(after.recorded - before.recorded >= 3, "{before:?} -> {after:?}");
    }
}

#[test]
fn reductions_recorded_together_are_not_replayed_alone() {
    // Two reductions of one operand are recorded in one kernel (the merged strategy, which a replay does not run): the step is not
    // cached, and runs as ever.
    let ctx = ctx();
    let step = |u: &A, t: usize| {
        let (lo, hi) = (u.min(), u.max());
        ((u - &lo) / &(&hi - &lo + 1.0) * (1.0 + 0.01 * t as f32)).clamp(-4.0, 4.0)
    };
    let start = data(W * H, 71);
    let (want, want_launches) = run_steps(&ctx, false, 8, &start, [W, H], &step);
    let (got, launches) = run_steps(&ctx, true, 8, &start, [W, H], &step);
    assert_eq!(got, want);
    assert_eq!(launches, want_launches);
}

#[test]
fn an_unused_reduction_does_not_disable_the_cache() {
    let ctx = ctx();
    let step = |u: &A, t: usize| {
        // Dropped without being used: its handle is gone before the evaluation.
        let _ = u.sum();
        (u * (1.0 + 0.01 * t as f32) + 0.01).clamp(-4.0, 4.0)
    };
    let start = data(W * H, 81);
    let (want, _) = run_steps(&ctx, false, 8, &start, [W, H], &step);
    let before = ctx.jit().plan_stats();
    let (got, _) = run_steps(&ctx, true, 8, &start, [W, H], &step);
    assert_eq!(got, want);
    let after = ctx.jit().plan_stats();
    if !ctx.can_run_on_host() {
        assert!(after.hits - before.hits >= 3, "{before:?} -> {after:?}");
    }
}
