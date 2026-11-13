//! Loops over arrays: `Array::iterate` (time-blocked stencils) and `iterate_all` (register loops).

use std::any::TypeId;
use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::error::LumaError;
use crate::gpu_type::ScalarType;
use crate::jit::{self, Trace};
use crate::ops::Outcome;

/// Runs `steps` iterations of `step` on several arrays at once (of any [`ScalarType`], `f32`,
/// [`Df32`](crate::Df32), `f64`, ...): `step` gets the current values
/// and returns the next ones, as a loop calling it `steps` times would. When every result is an
/// elementwise expression of the states, of other arrays of the same length and of scalars, the
/// whole loop is one kernel that keeps the states in registers and writes them once, for any
/// `steps` (one pipeline per step expression); that loop exists for `f32` and `u32` states (all of
/// one type). It runs in launches of about 40 ms of GPU time by the device's [`CostModel`](crate::CostModel), so a
/// long loop is a handful of launches. Anything else, such as shifts, `cache()` or `assign` or
/// states of another type, is traced once and its recorded kernels are replayed, step by step.
/// `step` is called once to trace it; it must compute its results from its arguments and captured
/// arrays only.
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
    let first = step(&probes);
    if let Some(attempt) = &mut attempt {
        attempt.stepped();
    }
    let shapes_match = first.iter().zip(&init).all(|(next, array)| next.shape() == array.shape());

    if trace.untouched() {
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
    for t in 1..=steps {
        stepper.plan().replay(&stepper.subst(t), None)?;
    }
    let mut finals = stepper.state_after(steps).iter();
    Ok(init.map(|array| match finals.next() {
        Some(storage) => Array::from_storage(ctx, array.shape(), Arc::clone(storage)),
        None => unreachable!(),
    }))
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
    Ok(states)
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
                Ok(u)
            }
        }
    }

    /// `iterate` for a state of one buffer: stencils run time-blocked, any other step is traced
    /// once and its recorded launches replayed.
    fn replayed(&self, steps: usize, step: impl Fn(&Array<f32>) -> Array<f32>) -> Outcome<Array<f32>> {
        let (mut trace, mut probes) = Trace::start(&[self])?;
        let Some(probe) = probes.pop() else { unreachable!() };
        let first = step(&probe);
        if trace.untouched() {
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
        for t in 1..=steps {
            stepper.plan().replay(&stepper.subst(t), None)?;
        }
        match stepper.state_after(steps).first() {
            Some(storage) => Ok(Array::from_storage(self.context(), self.shape(), Arc::clone(storage))),
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
