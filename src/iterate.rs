//! Loops over arrays: `Array::iterate` (time-blocked stencils) and `iterate_all` (register loops).

use std::any::TypeId;
use std::panic::Location;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::array::{Array, Failure, Storage};
use crate::error::LumaError;
use crate::gpu_type::ScalarType;
use crate::jit::{self, Plan, Stepper, Trace};
use crate::ops::Outcome;
use luma_core::BackendKind;

/// Runs `steps` iterations of `step` on several arrays at once (of any [`ScalarType`], `f32`,
/// [`Df32`](crate::Df32), `f64`, ...): `step` gets the current values
/// and returns the next ones, as a loop calling it `steps` times would. When every result is an
/// elementwise expression of the states, of other arrays of the same length and of scalars, the
/// whole loop is one kernel that keeps the states in registers and writes them once, for any
/// `steps` (one pipeline per step expression); that loop exists for `f32` and `u32` states (all of
/// one type). It runs in launches of about 40 ms of GPU time by the device's [`CostModel`](crate::CostModel), so a
/// long loop is a handful of launches. Anything else, such as shifts, `cache()` or `assign` or
/// states of another type, is traced once and its recorded kernels are replayed, step by step.
/// A step of lazy expressions only (shifts, say) is also traced as the body of several steps
/// (up to 16) when the planner fuses those into fewer launches than the steps make one by one,
/// which the loop then replays, and the steps that do not fill a body run one by one.
/// `step` is called once to trace it, and up to 16 times more for the body of several steps; it
/// must compute its results from its arguments and captured arrays only.
///
/// A register loop called again from the same place in the source whose step builds the same
/// graph (the same operations and scalar values) over other arrays, states or captured, is not
/// flattened again, and once two calls in a row did that, the GPU starts the loop before `step` is
/// traced (on the guess that it builds the same graph again; a wrong guess only wastes GPU time).
/// `step` is still called once per call, and a captured array is read when the loop runs.
///
/// ```
/// # use luma::*;
/// let c = Array::from_slice(&[0.1f32, -0.5, 0.3, 0.25], 4)?;
/// let zero = Array::<f32>::zeros(4)?;
/// let [x, count] = iterate_all([&zero, &zero], 64, |[x, count]| {
///     let inside = x.le(2.0);
///     [select(&inside, &(x * x + &c), x), count + select(&inside, 1.0, 0.0)]
/// });
/// # x.to_vec()?; count.to_vec()?;
/// # Ok::<(), luma::LumaError>(())
/// ```
#[track_caller]
pub fn iterate_all<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    steps: usize,
    step: impl Fn(&[Array<T>; N]) -> [Array<T>; N],
) -> [Array<T>; N] {
    const { assert!(N >= 1, "iterate_all takes at least one array") };
    let location = Location::caller();
    let ctx = init[0].context();
    match run_iterate_all(init, steps, &step, location) {
        Ok(states) => states,
        Err(failure) => {
            let error = match failure {
                Failure::Inherited(error) => error,
                Failure::New(error) => {
                    Arc::new(LumaError::Op { op: "iterate_all", location, source: Arc::new(error) })
                }
            };
            init.map(|array| {
                Array::resolve(ctx, array.shape(), Err(Failure::Inherited(Arc::clone(&error))), "iterate_all", location)
            })
        }
    }
}

fn run_iterate_all<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    steps: usize,
    step: &impl Fn(&[Array<T>; N]) -> [Array<T>; N],
    location: &'static Location<'static>,
) -> Outcome<[Array<T>; N]> {
    let ctx = init[0].context();
    if init.iter().any(|array| !ctx.same_as(array.context())) {
        return Err(Failure::New(LumaError::ContextMismatch));
    }
    for array in &init {
        if let Some(error) = array.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
    }
    if steps == 0 {
        return Ok(init.map(Array::copied));
    }
    let mut single = true;
    for array in &init {
        single &= array.storage()?.segments() == 1;
    }
    if single {
        return replayed_all(init, steps, step, location);
    }
    iterate_all_aliased(init, steps, step)
}

/// `iterate_all` for states of one buffer each: `step` is traced once on placeholders while the
/// context records its launches, and the recorded launches are replayed `steps` times. A step that
/// turns out to be one elementwise program runs as a register loop, from the program of the last
/// call at `location` when the step builds the same graph (see `jit::Attempt`).
fn replayed_all<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    steps: usize,
    step: &impl Fn(&[Array<T>; N]) -> [Array<T>; N],
    location: &'static Location<'static>,
) -> Outcome<[Array<T>; N]> {
    let ctx = init[0].context();
    let (mut trace, probes) = Trace::prepare(&init)?;
    let site = jit::Site::new(location, TypeId::of::<T>(), N, trace.init[0].len());
    let mut attempt = jit::Attempt::begin(ctx, site, &trace.init, &trace.probe_storages, steps);
    trace.begin()?;
    let Ok(probes): std::result::Result<[Array<T>; N], _> = probes.try_into() else { unreachable!() };
    let started = Instant::now();
    let first = step(&probes);
    if let Some(attempt) = &mut attempt {
        attempt.stepped();
    }
    let shapes_match = first.iter().zip(&init).all(|(next, array)| next.shape() == array.shape());

    let lazy = trace.untouched();
    if lazy {
        // Only lazy expressions: they may be a loop that runs in one kernel.
        trace.pause();
        let traced: Option<Vec<Arc<jit::Node>>> = first.iter().map(|next| next.pending().cloned()).collect();
        if shapes_match
            && let (Ok(count), Some(outs)) = (u32::try_from(steps), traced)
            && let Some(result) = match attempt.take() {
                Some(attempt) => attempt.run(&trace.probe_storages, &trace.init, &outs),
                None => jit::run_loop(ctx, &trace.probe_storages, &trace.init, &outs, count),
            }
        {
            let mut outputs = result?.into_iter();
            return Ok(init.map(|array| match outputs.next() {
                Some(storage) => Array::from_storage(ctx, array.shape(), storage),
                None => unreachable!(),
            }));
        }
        trace.resume()?;
    }

    let stepper = trace.finish(&first.each_ref(), shapes_match)?;
    let Some(stepper) = stepper else {
        // Everything ran on the placeholders already: carry on from the first result.
        let mut states = first;
        for _ in 1..steps {
            states = step(&states);
        }
        for state in &states {
            state.check()?;
        }
        return Ok(states);
    };
    let unrolled = if lazy && shapes_match { Some(started.elapsed()) } else { None };
    let mut finals = run_stepper(init, steps, step, stepper, unrolled)?.into_iter();
    Ok(init.map(|array| match finals.next() {
        Some(storage) => Array::from_storage(ctx, array.shape(), storage),
        None => unreachable!(),
    }))
}

/// The storages of the state after `steps` steps of `stepper`, the traced single step of a loop
/// from `init`. With `traced`, the time tracing it took, given where `step` is lazy expressions
/// only, the loop runs as replays of an unrolled body where that pays (see [`trace_unrolled`]).
fn run_stepper<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    steps: usize,
    step: &impl Fn(&[Array<T>; N]) -> [Array<T>; N],
    mut stepper: Stepper,
    traced: Option<Duration>,
) -> Outcome<Vec<Arc<Storage>>> {
    // The CPU backend has no workgroup memory to stage the shifted values of fused steps in: it
    // recomputes them, which is slower than the steps one by one for a large state.
    if let Some(traced) = traced.filter(|_| init[0].context().backend() != BackendKind::Cpu) {
        let depth = unroll_depth(steps, stepper.plan(), traced);
        if let Some(unrolled) = trace_unrolled(init, depth, step, stepper.plan())? {
            return run_unrolled(&mut stepper, &unrolled, depth, steps);
        }
    }
    stepper.run_steps(steps)?;
    Ok(stepper.state_after(steps).to_vec())
}

/// Most steps of a pure expression traced as one body of a replayed loop (see [`trace_unrolled`]).
const UNROLL_STEPS: usize = 16;

/// The share of the time the cost model gives the loop that tracing its unrolled body may take.
const UNROLL_TRACE_SHARE: f64 = 0.5;

/// How many times the time a step moves its data takes the time of its launches, at least, for
/// fusing steps to pay: a step that is a launch of a few microseconds costs about as much fused
/// with others (a fused kernel is as long as the steps it holds, minus what they move).
const UNROLL_BANDWIDTH_BOUND: f64 = 2.0;

/// The steps in the unrolled body of a loop of `steps` steps (below 2: none) whose single step
/// is `single` and took `traced` to trace: none for a step that is bound by its launches, else as
/// many as the tracing (one trace per step) costs less than [`UNROLL_TRACE_SHARE`] of what the
/// loop would take step by step by the cost model, which is all that fusing the steps can save.
fn unroll_depth(steps: usize, single: &Plan, traced: Duration) -> usize {
    let (launching, moving) = single.modeled_ns();
    if moving < UNROLL_BANDWIDTH_BOUND * launching {
        return 0;
    }
    let affordable = (steps as f64 * (launching + moving) * UNROLL_TRACE_SHARE / (traced.as_nanos() as f64).max(1.0)).min(UNROLL_STEPS as f64) as usize;
    steps.min(affordable)
}

/// The stepper of `depth` steps of `step` traced as one body, where `depth` is at least 2: the
/// launches the planner makes of the chain of `depth` steps of lazy expressions (it fuses steps
/// of shifted expressions into kernels, as an eager loop that evaluates every step lazily
/// does), when they make fewer launches than `depth` steps of `single` (the plan of one step)
/// and the cost model gives them less time. `None` when unrolling does not help, or the
/// traced code aborted the recording.
fn trace_unrolled<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    depth: usize,
    step: &impl Fn(&[Array<T>; N]) -> [Array<T>; N],
    single: &Plan,
) -> Outcome<Option<Stepper>> {
    if depth < 2 {
        return Ok(None);
    }
    let (mut trace, probes) = Trace::start(&init)?;
    let Ok(probes): std::result::Result<[Array<T>; N], _> = probes.try_into() else { unreachable!() };
    if !trace.recording() {
        return Ok(None);
    }
    let mut states = step(&probes);
    for _ in 1..depth {
        states = step(&states);
    }
    // The planner evaluates the steps it does not fuse while the chain is built: those launches are
    // recorded with the rest.
    let shapes_match = states.iter().zip(&init).all(|(next, array)| next.shape() == array.shape());
    if !shapes_match || !trace.recording() {
        return Ok(None);
    }
    let unrolled = trace.finish_chain(&states.each_ref(), depth * single.launches())?;
    let total = |plan: &Plan| plan.modeled_ns().0 + plan.modeled_ns().1;
    Ok(unrolled.filter(|unrolled| total(unrolled.plan()) < UNROLL_GAIN * depth as f64 * total(single)))
}

/// The share of the time of `depth` steps one by one that the unrolled body must take, at most.
const UNROLL_GAIN: f64 = 0.9;

/// The storages of the state after `steps` steps: replays of `unrolled` (`depth` steps each),
/// then `stepper` for the rest, which takes the state where the replays left it.
fn run_unrolled(stepper: &mut Stepper, unrolled: &Stepper, depth: usize, steps: usize) -> Outcome<Vec<Arc<Storage>>> {
    let (full, rest) = (steps / depth, steps % depth);
    if rest > 0 && !stepper.rebind(unrolled.state_after(full)) {
        stepper.run_steps(steps)?;
        return Ok(stepper.state_after(steps).to_vec());
    }
    unrolled.run_steps(full)?;
    if rest == 0 {
        return Ok(unrolled.state_after(full).to_vec());
    }
    stepper.run_steps(rest)?;
    Ok(stepper.state_after(rest).to_vec())
}

/// The step-by-step loop of `iterate_all` on placeholders that alias the states' buffers, for
/// states that span several buffers.
fn iterate_all_aliased<T: ScalarType, const N: usize>(
    init: [&Array<T>; N],
    steps: usize,
    step: &impl Fn(&[Array<T>; N]) -> [Array<T>; N],
) -> Outcome<[Array<T>; N]> {
    let ctx = init[0].context();
    let mut probes = Vec::with_capacity(N);
    for array in &init {
        probes.push(array.probe()?);
    }
    let Ok(probes): std::result::Result<[Array<T>; N], _> = probes.try_into() else { unreachable!() };
    let first = step(&probes);

    let shapes_match = first.iter().zip(&init).all(|(next, array)| next.shape() == array.shape());
    let traced: Option<Vec<Arc<jit::Node>>> = first.iter().map(|next| next.pending().cloned()).collect();
    if shapes_match
        && let (Ok(count), Some(outs)) = (u32::try_from(steps), traced)
    {
        let storages = |arrays: &[&Array<T>]| -> Outcome<Vec<Arc<Storage>>> {
            arrays.iter().map(|array| array.storage().map(Arc::clone)).collect()
        };
        let probe_storages = storages(&probes.each_ref())?;
        let init_storages = storages(&init)?;
        if let Some(result) = jit::run_loop(ctx, &probe_storages, &init_storages, &outs, count) {
            let mut outputs = result?.into_iter();
            return Ok(init.map(|array| match outputs.next() {
                Some(storage) => Array::from_storage(ctx, array.shape(), storage),
                None => unreachable!(),
            }));
        }
    }

    for (array, probe) in init.iter().zip(&probes) {
        if let (Ok(owner), Ok(alias)) = (array.storage(), probe.storage()) {
            owner.readers.absorb(&alias.readers);
        }
    }
    let mut states = first;
    for _ in 1..steps {
        states = step(&states);
    }
    for state in &states {
        state.check()?;
    }
    // A step that returns its argument unchanged returns the placeholder.
    Ok(states.map(Array::detached))
}

impl Array<f32> {
    /// `steps` applications of `step`: `u = step(u)`, starting from this array, like a loop but
    /// run time-blocked where possible. When `step` is a stencil — one elementwise expression of
    /// its argument, shifts of it by a few elements (circular, as [`Array::shift`]), scalars and
    /// other arrays of the same shape that do not change — each launch loads a tile into
    /// workgroup memory and runs several steps on it, so the data passes through memory once per
    /// several steps instead of once per step. Any other `step` (shifts of a computed expression,
    /// `cache()`, `assign`, reductions) is traced once and its recorded kernels are replayed, step
    /// by step.
    ///
    /// `step` is called once, with a placeholder, to trace it; it must compute its result from its
    /// argument and captured arrays only, and not keep the argument.
    ///
    /// ```no_run
    /// # let u0 = luma::Array::<f32>::zeros([512, 512])?;
    /// let alpha = 0.2;
    /// let u = u0.iterate(100, |u| {
    ///     let lap = u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1]) - u * 4.0;
    ///     u + lap * alpha
    /// });
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn iterate(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_iterate(steps, step);
        Array::resolve(self.context(), self.shape(), result, "iterate", location)
    }

    fn run_iterate(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Outcome<Array<f32>> {
        if let Some(error) = self.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
        if steps == 0 {
            return Ok(self.copied());
        }
        if self.storage()?.segments() == 1 {
            return self.replayed(steps, step);
        }
        let probe = self.probe()?;
        let first = step(&probe);
        match jit::stencil_plan(&probe, &first) {
            Some(stencil) => stencil.run(self, steps),
            None => {
                if let (Ok(owner), Ok(alias)) = (self.storage(), probe.storage()) {
                    owner.readers.absorb(&alias.readers);
                }
                let mut u = first;
                for _ in 1..steps {
                    u = step(&u);
                }
                u.check()?;
                Ok(u.detached())
            }
        }
    }

    /// `iterate` for a state of one buffer: stencils run time-blocked, any other step is traced
    /// once and its recorded launches replayed.
    fn replayed(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Outcome<Array<f32>> {
        let (mut trace, mut probes) = Trace::start(&[self])?;
        let Some(probe) = probes.pop() else { unreachable!() };
        let started = Instant::now();
        let first = step(&probe);
        let lazy = trace.untouched();
        if lazy {
            trace.pause();
            if let Some(stencil) = jit::stencil_plan(&probe, &first) {
                return stencil.run(self, steps);
            }
            trace.resume()?;
        }
        let Some(stepper) = trace.finish(&[&first], first.shape() == self.shape())? else {
            let mut u = first;
            for _ in 1..steps {
                u = step(&u);
            }
            u.check()?;
            return Ok(u);
        };
        let shapes_match = first.shape() == self.shape();
        drop((probe, first));
        let traced = if lazy && shapes_match { Some(started.elapsed()) } else { None };
        let mut finals = run_stepper([self], steps, &|[u]: &[Array<f32>; 1]| [step(u)], stepper, traced)?.into_iter();
        match finals.next() {
            Some(storage) => Ok(Array::from_storage(self.context(), self.shape(), storage)),
            None => unreachable!(),
        }
    }

    /// Runs `step` until the largest `|next - prev|` of a step is at most `tol`, or `max_steps` steps,
    /// and returns the last state. Convergence is checked on the GPU; the host waits once per
    /// block of steps (the first block is 16 steps, every block that ends without convergence doubles
    /// the next one, up to 128), and steps after convergence run no work. The result does not depend
    /// on the block schedule.
    #[track_caller]
    pub fn iterate_until(&self, max_steps: usize, tol: f32, step: impl Fn(&Array<f32>) -> Array<f32>) -> Array<f32> {
        let location = Location::caller();
        let result = self.run_iterate_until(max_steps, tol, step);
        Array::resolve(self.context(), self.shape(), result, "iterate_until", location)
    }

    fn run_iterate_until(
        &self,
        max_steps: usize,
        tol: f32,
        step: impl Fn(&Array<f32>) -> Array<f32>,
    ) -> Outcome<Array<f32>> {
        if let Some(error) = self.error() {
            return Err(Failure::Inherited(Arc::new(error.clone())));
        }
        if max_steps == 0 {
            return Ok(self.copied());
        }
        let probe = self.probe()?;
        let first = step(&probe);
        let change = nan_to_inf((&first - &probe).abs());
        if first.shape() == self.shape()
            && let (Some(first), Some(change)) = (first.pending(), change.pending())
            && let (Ok(from), Ok(alias)) = (self.storage(), probe.storage())
            && let Some(result) = jit::run_until(self.context(), from, alias, first, change, max_steps, tol)
        {
            return Ok(Array::from_storage(self.context(), self.shape(), result?));
        }
        if let (Ok(owner), Ok(alias)) = (self.storage(), probe.storage()) {
            owner.readers.absorb(&alias.readers);
        }
        let (mut u, mut change) = (first, change);
        for done in 1..=max_steps {
            if done == max_steps || change.max().item().map_err(Failure::New)? <= tol {
                break;
            }
            let next = step(&u);
            change = nan_to_inf((&next - &u).abs());
            u = next;
        }
        u.check()?;
        Ok(u)
    }
}

/// `diff` with every NaN replaced by infinity, so that a NaN step never counts as converged.
fn nan_to_inf(diff: Array<f32>) -> Array<f32> {
    crate::ops::select(&diff.is_nan(), f32::INFINITY, &diff)
}
