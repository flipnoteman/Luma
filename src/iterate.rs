//! Loops over arrays: `Array::iterate` (time-blocked stencils) and `iterate_all` (register loops).

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure, Storage};
use crate::error::LumaError;
use crate::jit;
use crate::ops::Outcome;

/// Runs `steps` iterations of `step` on several arrays at once: `step` gets the current values
/// and returns the next ones, as a loop calling it `steps` times would. When every result is an
/// elementwise expression of the states, of other arrays of the same length and of scalars, the
/// whole loop is one kernel that keeps the states in registers and writes them once, for any
/// `steps` (one pipeline per step expression). Anything else, such as shifts, runs step by step.
/// `step` is called once more than that to trace it.
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
pub fn iterate_all<const N: usize>(
    init: [&Array<f32>; N],
    steps: usize,
    step: impl Fn(&[Array<f32>; N]) -> [Array<f32>; N],
) -> [Array<f32>; N] {
    const { assert!(N >= 1, "iterate_all takes at least one array") };
    let location = Location::caller();
    let ctx = init[0].context();
    match run_iterate_all(init, steps, &step) {
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

fn run_iterate_all<const N: usize>(
    init: [&Array<f32>; N],
    steps: usize,
    step: &impl Fn(&[Array<f32>; N]) -> [Array<f32>; N],
) -> Outcome<[Array<f32>; N]> {
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
        return Ok(init.map(|array| array * 1.0));
    }
    let mut probes = Vec::with_capacity(N);
    for array in &init {
        probes.push(array.probe()?);
    }
    let Ok(probes): std::result::Result<[Array<f32>; N], _> = probes.try_into() else { unreachable!() };
    let first = step(&probes);

    let shapes_match = first.iter().zip(&init).all(|(next, array)| next.shape() == array.shape());
    let traced: Option<Vec<Arc<jit::Node>>> = first.iter().map(|next| next.pending().cloned()).collect();
    if shapes_match
        && let (Ok(count), Some(outs)) = (u32::try_from(steps), traced)
    {
        let storages = |arrays: &[&Array<f32>]| -> Outcome<Vec<Arc<Storage>>> {
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
    /// several steps instead of once per step. Any other `step` simply runs in a loop.
    ///
    /// `step` is called with a placeholder first to inspect the expression it builds; it must
    /// compute its result from its argument and not keep it.
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
            return Ok(self * 1.0);
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

    /// Runs `step` until the largest `|next - prev|` of a step is at most `tol`, or `max_steps` steps,
    /// and returns the last state. Convergence is checked on the GPU; the host waits once per
    /// block of steps, and steps after convergence run no work.
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
            return Ok(self * 1.0);
        }
        let probe = self.probe()?;
        let first = step(&probe);
        let change = (&first - &probe).abs();
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
            if done == max_steps || change.max().map_err(Failure::New)? <= tol {
                break;
            }
            let next = step(&u);
            change = (&next - &u).abs();
            u = next;
        }
        u.check()?;
        Ok(u)
    }
}
