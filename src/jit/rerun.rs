//! Register loops that run again. `iterate_all` traces its step on every call: the step builds its
//! graph, the graph is flattened into a program, the program is looked up (interned, hashed) and
//! only then launched. A program that runs the same step over new arrays (a hash of a new batch, a
//! time loop called per frame) repeats all of it, and for a step of thousands of operations that
//! host work is as long as the GPU's.
//!
//! The step still runs once per call, so the graph is still built; what is skipped is everything
//! after it, and the wait for it:
//!
//! - While the step runs, every node the thread creates is hashed into a *fingerprint* of the
//!   graph: its operations, lengths and operands (nodes by creation order, buffers by order of
//!   first use, scalars by value). Two traces with the same fingerprint build the same program over
//!   the same scalars, so the flattened program of the earlier one is the program of this one: a
//!   [`Rerun`], found by the call site of `iterate_all`, replaces the flattening, and the buffers
//!   it reads (the captured arrays, which may be others than the last time) are read off the
//!   trace.
//! - A rerun that matched on the call before is *speculated on*: the loop launches over the new
//!   states before the step runs, on the guess that the step is the same, and the GPU computes
//!   while the host builds the graph that tells whether it was. The speculation only writes buffers
//!   of its own, so a wrong guess (the fingerprint differs, a captured array is another one) costs
//!   GPU time and nothing else; the call then runs the loop as if there had been no rerun.

use std::any::TypeId;
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::panic::Location;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::array::{Failure, Storage};
use crate::context::Context;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;

use super::program::FusedKey;
use super::{launch, Args, Node, Op, Src};

/// Hashes the nodes of a trace into 128 bits: two multiply-xorshift streams, each a bijection of
/// its state for any word, so traces that differ in one word never meet again.
#[derive(Clone, Copy)]
struct Mix {
    a: u64,
    b: u64,
}

impl Mix {
    const A: u64 = 0x517c_c1b7_2722_0a95;
    const B: u64 = 0x9e37_79b9_7f4a_7c15;

    fn new(seed: u64) -> Self {
        let mut mix = Mix { a: 0x243f_6a88_85a3_08d3, b: 0x1319_8a2e_0370_7344 };
        mix.word(seed);
        mix
    }

    #[inline]
    fn word(&mut self, w: u64) {
        self.a = (self.a.rotate_left(5) ^ w).wrapping_mul(Self::A);
        self.b = (self.b ^ w.rotate_left(17)).wrapping_mul(Self::B);
        self.b ^= self.b >> 29;
    }
}

impl Hasher for Mix {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            self.word(chunk.iter().rev().fold(0u64, |word, &byte| (word << 8) | u64::from(byte)));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.word(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.word(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.word(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.word(i as u64);
    }

    fn finish(&self) -> u64 {
        self.a ^ self.b
    }
}

/// Numbers the traces of a thread (never 0: a node with trace tag 0 was created outside any).
static GENERATION: AtomicU32 = AtomicU32::new(1);

/// The trace being hashed.
struct State {
    generation: u32,
    /// Nodes created so far.
    next: u32,
    mix: Mix,
    /// A node read something the hash does not describe (a node of no trace, of an earlier one).
    poisoned: bool,
    /// The distinct buffers the nodes read, in order of first use; the states' probes come first.
    buffers: Vec<Weak<Storage>>,
    index: FxMap<u64, u32>,
}

impl State {
    fn operand(&mut self, arg: &Src) {
        match arg {
            Src::Node(node) => {
                if node.trace >> 32 == u64::from(self.generation) {
                    self.mix.word(1);
                    self.mix.word(node.trace & 0xffff_ffff);
                } else {
                    self.poisoned = true;
                }
            }
            Src::Buffer(storage) => {
                let next = self.buffers.len() as u32;
                let at = *self.index.entry(storage.id()).or_insert(next);
                if at == next {
                    self.buffers.push(Arc::downgrade(storage));
                }
                self.mix.word(2);
                self.mix.word(u64::from(at));
            }
            Src::Scalar(scalar) => {
                self.mix.word(3);
                scalar.ty.hash(&mut self.mix);
                self.mix.word(scalar.bits);
            }
        }
    }
}

thread_local! {
    /// The traces under way on this thread, innermost last.
    static STACK: RefCell<Vec<State>> = const { RefCell::new(Vec::new()) };
    /// Whether `STACK` is not empty, so that the thread that traces nothing reads one `Cell`.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Records that the thread created a node: the tag the node carries (0 outside a trace).
pub(super) fn created(op: Op, len: usize, args: &Args, arity: usize) -> u64 {
    if !ACTIVE.with(Cell::get) {
        return 0;
    }
    STACK.with_borrow_mut(|stack| {
        let Some(state) = stack.last_mut() else { return 0 };
        state.mix.word(4);
        op.hash(&mut state.mix);
        state.mix.word(len as u64);
        for arg in args.iter().take(arity) {
            state.operand(arg);
        }
        let tag = u64::from(state.generation) << 32 | u64::from(state.next);
        state.next += 1;
        tag
    })
}

/// Records that creating a node returned `found`, an existing one.
pub(super) fn reused(found: &Node) {
    if !ACTIVE.with(Cell::get) {
        return;
    }
    STACK.with_borrow_mut(|stack| {
        let Some(state) = stack.last_mut() else { return };
        if found.trace >> 32 == u64::from(state.generation) {
            state.mix.word(5);
            state.mix.word(found.trace & 0xffff_ffff);
        } else {
            state.poisoned = true;
        }
    });
}

/// What the graph a trace built is, once the step returned: see [`Print::seal`].
struct Print {
    generation: u32,
    mix: Mix,
    buffers: Vec<Weak<Storage>>,
}

impl Print {
    /// The fingerprint of the trace with `outs` as the results of the step (nodes of the trace, or
    /// `None`).
    fn seal(&self, outs: &[Arc<Node>]) -> Option<[u64; 2]> {
        let mut mix = self.mix;
        mix.word(outs.len() as u64);
        for out in outs {
            if out.trace >> 32 != u64::from(self.generation) {
                return None;
            }
            mix.word(out.trace & 0xffff_ffff);
        }
        Some([mix.a, mix.b])
    }
}

/// A trace under way: from its creation until [`Tracing::end`] the nodes this thread creates are
/// hashed. Dropping it ends it.
struct Tracing {
    depth: usize,
}

impl Tracing {
    fn begin(seed: u64, probes: &[Arc<Storage>]) -> Tracing {
        let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        let mut state = State {
            generation,
            next: 0,
            mix: Mix::new(seed),
            poisoned: false,
            buffers: Vec::with_capacity(probes.len() + 8),
            index: FxMap::default(),
        };
        for (j, probe) in probes.iter().enumerate() {
            state.index.insert(probe.id(), j as u32);
            state.buffers.push(Arc::downgrade(probe));
        }
        let depth = STACK.with_borrow_mut(|stack| {
            stack.push(state);
            stack.len()
        });
        ACTIVE.with(|active| active.set(true));
        Tracing { depth }
    }

    fn end(&mut self) -> Option<Print> {
        let state = STACK.with_borrow_mut(|stack| {
            let state = (stack.len() >= self.depth && self.depth > 0).then(|| stack.drain(self.depth - 1..).next()).flatten();
            ACTIVE.with(|active| active.set(!stack.is_empty()));
            state
        });
        self.depth = 0;
        let state = state?;
        (!state.poisoned).then_some(Print { generation: state.generation, mix: state.mix, buffers: state.buffers })
    }
}

impl Drop for Tracing {
    fn drop(&mut self) {
        drop(self.end());
    }
}

/// Where `iterate_all` was called, with what it was called on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Site {
    location: usize,
    ty: TypeId,
    states: usize,
    len: usize,
}

impl Site {
    pub(crate) fn new(location: &'static Location<'static>, ty: TypeId, states: usize, len: usize) -> Self {
        Site { location: std::ptr::from_ref(location) as usize, ty, states, len }
    }
}

/// A register loop whose program is known: what the flattening of a trace with `hash` gave.
struct Rerun {
    hash: [u64; 2],
    key: FusedKey,
    scalars: Vec<u32>,
    /// Per input of the program, the state it reads.
    states: Vec<Option<usize>>,
    /// Per input that stays fixed (in input order), the position of its buffer in the trace's
    /// buffers (see [`State::buffers`]).
    fixed: Vec<u32>,
    /// The weight of the program per element and step (see [`launch::loop_ops`]).
    ops: f64,
    /// Calls in a row that ran this rerun after the one that made it; a guess that turned out
    /// wrong (see [`Attempt`]'s drop) sets it below zero, by `penalty` calls.
    streak: AtomicI32,
    /// Calls the next wrong guess makes the rerun sit out: doubles with every wrong guess, back to
    /// 1 with a right one, so that a call site that alternates is not speculated on every call.
    penalty: AtomicU32,
    /// The buffers the last call that ran it read through `fixed`.
    last_fixed: Mutex<Vec<Weak<Storage>>>,
}

/// The reruns of the most recent call sites, recently used last.
#[derive(Default)]
pub(super) struct Cache {
    sites: Vec<(Site, Vec<Arc<Rerun>>)>,
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cache({} call sites)", self.sites.len())
    }
}

const MAX_SITES: usize = 32;
const MAX_RERUNS: usize = 4;

impl Cache {
    fn find(&mut self, site: &Site, hash: [u64; 2]) -> Option<Arc<Rerun>> {
        let at = self.sites.iter().position(|(known, _)| known == site)?;
        let entry = self.sites.remove(at);
        self.sites.push(entry);
        let (_, reruns) = self.sites.last_mut()?;
        let found = reruns.iter().position(|rerun| rerun.hash == hash)?;
        let rerun = reruns.remove(found);
        reruns.push(Arc::clone(&rerun));
        for other in &reruns[..reruns.len() - 1] {
            other.streak.fetch_min(0, Ordering::Relaxed);
        }
        Some(rerun)
    }

    /// The rerun of `site` that ran last, if it ran twice in a row.
    fn confident(&self, site: &Site) -> Option<Arc<Rerun>> {
        let (_, reruns) = self.sites.iter().find(|(known, _)| known == site)?;
        reruns.last().filter(|rerun| rerun.streak.load(Ordering::Relaxed) >= 1).cloned()
    }

    fn insert(&mut self, site: Site, rerun: Arc<Rerun>) {
        let at = match self.sites.iter().position(|(known, _)| *known == site) {
            Some(at) => at,
            None => {
                if self.sites.len() >= MAX_SITES {
                    self.sites.remove(0);
                }
                self.sites.push((site, Vec::new()));
                self.sites.len() - 1
            }
        };
        let (_, reruns) = &mut self.sites[at];
        if reruns.len() >= MAX_RERUNS {
            reruns.remove(0);
        }
        for other in reruns.iter() {
            other.streak.fetch_min(0, Ordering::Relaxed);
        }
        reruns.push(rerun);
        let entry = self.sites.remove(at);
        self.sites.push(entry);
    }
}

/// The loop started on a guess, writing buffers of its own (see the module documentation).
struct Speculation {
    rerun: Arc<Rerun>,
    fixed: Vec<Arc<Storage>>,
    outputs: Vec<Arc<Storage>>,
}

/// What of the environment changes the program a trace flattens to.
fn stamp(ctx: &Context) -> u64 {
    let jit = ctx.jit();
    let mut mix = Mix::new(jit.max_nodes() as u64);
    mix.word(jit.max_inputs() as u64);
    mix.word(u64::from(jit.simplify_enabled()));
    mix.word(u64::from(ctx.fast_math()));
    mix.finish()
}

/// One call of a register loop over single-buffer states (`iterate_all`): begun before the step
/// runs, it speculates on the rerun of the call site and hashes what the step builds; [`Attempt::run`]
/// then runs the loop.
pub(crate) struct Attempt {
    ctx: Context,
    site: Site,
    steps: u32,
    tracing: Tracing,
    print: Option<Print>,
    speculation: Option<Speculation>,
}

impl Attempt {
    /// Starts the call: `inits` are the states, `probes` the placeholders the step will run on.
    /// Must run while this thread is not recording launches (before `Trace::begin`): the
    /// speculation launches for real.
    pub(crate) fn begin(ctx: &Context, site: Site, inits: &[Arc<Storage>], probes: &[Arc<Storage>], steps: usize) -> Option<Attempt> {
        let steps = u32::try_from(steps).ok()?;
        let speculation = speculate(ctx, &site, inits, steps);
        let tracing = Tracing::begin(stamp(ctx), probes);
        Some(Attempt { ctx: ctx.clone(), site, steps, tracing, print: None, speculation })
    }

    /// The step returned: stops hashing.
    pub(crate) fn stepped(&mut self) {
        self.print = self.tracing.end();
    }

    /// Runs the loop whose next states are `outs` (computed from `probes`, placeholders of
    /// `inits`): from the rerun the fingerprint found, else from the flattened trace, which
    /// becomes a rerun. `None` if the step is not a register loop (the caller runs it another way).
    pub(crate) fn run(
        mut self,
        probes: &[Arc<Storage>],
        inits: &[Arc<Storage>],
        outs: &[Arc<Node>],
    ) -> Option<std::result::Result<Vec<Arc<Storage>>, Failure>> {
        let ctx = self.ctx.clone();
        let jit = ctx.jit();
        let print = self.print.take();
        let sealed = print.as_ref().and_then(|print| print.seal(outs));
        if let (Some(print), Some(hash)) = (&print, sealed) {
            let rerun = lock(&jit.reruns).find(&self.site, hash);
            if let Some(rerun) = &rerun
                && let Some(result) = self.replay(rerun, print, probes, inits)
            {
                return Some(result);
            }
        }

        let (fused, states) = launch::prepare_loop(probes, inits, outs)?;
        let ops = launch::loop_ops(&ctx, &fused.key, &states)?;
        if fused.inputs.iter().any(|input| input.segments() != 1) {
            return None;
        }
        let fixed: Vec<Arc<Storage>> =
            fused.inputs.iter().zip(&states).filter(|(_, state)| state.is_none()).map(|(input, _)| Arc::clone(input)).collect();
        if let (Some(print), Some(hash)) = (&print, sealed) {
            let positions: Option<Vec<u32>> = fixed
                .iter()
                .map(|input| print.buffers.iter().position(|known| std::ptr::eq(known.as_ptr(), Arc::as_ptr(input))).map(|at| at as u32))
                .collect();
            if let Some(fixed_positions) = positions {
                let rerun = Rerun {
                    hash,
                    key: fused.key.clone(),
                    scalars: fused.scalars.clone(),
                    states: states.clone(),
                    fixed: fixed_positions,
                    ops,
                    streak: AtomicI32::new(0),
                    penalty: AtomicU32::new(1),
                    last_fixed: Mutex::new(fixed.iter().map(Arc::downgrade).collect()),
                };
                lock(&jit.reruns).insert(self.site, Arc::new(rerun));
            }
        }
        Some(launch::run_loop_program(&ctx, &fused.key, &fused.scalars, &states, &fixed, inits, self.steps, ops))
    }

    /// Runs `rerun` on this trace: `None` if the trace cannot run it (a buffer is gone, a
    /// state spans several buffers).
    fn replay(
        &mut self,
        rerun: &Arc<Rerun>,
        print: &Print,
        probes: &[Arc<Storage>],
        inits: &[Arc<Storage>],
    ) -> Option<std::result::Result<Vec<Arc<Storage>>, Failure>> {
        let len = inits.first()?.len();
        let fixed: Vec<Arc<Storage>> =
            rerun.fixed.iter().map(|&at| print.buffers.get(at as usize)?.upgrade()).collect::<Option<_>>()?;
        if probes.iter().any(|probe| probe.was_materialized())
            || fixed.iter().any(|input| input.segments() != 1)
            || inits.iter().any(|init| init.elem() != 4 || init.len() != len || init.segments() != 1)
        {
            return None;
        }
        let ctx = &self.ctx;
        let jit = ctx.jit();
        rerun.streak.fetch_add(1, Ordering::Relaxed);
        *lock(&rerun.last_fixed) = fixed.iter().map(Arc::downgrade).collect();
        jit.stats.replays.fetch_add(1, Ordering::Relaxed);
        if self.speculation.as_ref().is_some_and(|speculation| {
            Arc::ptr_eq(&speculation.rerun, rerun)
                && speculation.fixed.len() == fixed.len()
                && speculation.fixed.iter().zip(&fixed).all(|(a, b)| Arc::ptr_eq(a, b))
        }) && let Some(speculation) = self.speculation.take()
        {
            jit.stats.speculated.fetch_add(1, Ordering::Relaxed);
            rerun.penalty.store(1, Ordering::Relaxed);
            return Some(Ok(speculation.outputs));
        }
        Some(launch::run_loop_program(ctx, &rerun.key, &rerun.scalars, &rerun.states, &fixed, inits, self.steps, rerun.ops))
    }
}

impl Drop for Attempt {
    fn drop(&mut self) {
        // A guess that was not used was wrong: the next calls do not start on it, the more of
        // them the more often it was wrong.
        if let Some(speculation) = self.speculation.take() {
            let rerun = &speculation.rerun;
            let penalty = rerun.penalty.load(Ordering::Relaxed);
            rerun.streak.store(-i32::try_from(penalty).unwrap_or(1), Ordering::Relaxed);
            rerun.penalty.store((penalty * 2).min(64), Ordering::Relaxed);
        }
    }
}

/// Starts the loop of the rerun `site` ran twice in a row over `inits`, if it has one and its
/// buffers are alive.
fn speculate(ctx: &Context, site: &Site, inits: &[Arc<Storage>], steps: u32) -> Option<Speculation> {
    if ctx.recording_here() || steps == 0 {
        return None;
    }
    let rerun = lock(&ctx.jit().reruns).confident(site)?;
    let fixed: Vec<Arc<Storage>> = lock(&rerun.last_fixed).iter().map(Weak::upgrade).collect::<Option<_>>()?;
    let len = inits.first()?.len();
    if rerun.states.iter().flatten().any(|&j| j >= inits.len())
        || fixed.iter().any(|input| input.segments() != 1)
        || inits.iter().any(|init| init.elem() != 4 || init.len() != len || init.segments() != 1)
    {
        return None;
    }
    let outputs = launch::run_loop_program(ctx, &rerun.key, &rerun.scalars, &rerun.states, &fixed, inits, steps, rerun.ops).ok()?;
    // The GPU starts now, while the host builds the graph that tells whether this was the loop.
    let _ = ctx.submit();
    Some(Speculation { rerun, fixed, outputs })
}
