//! A cache of the plans of eager evaluations.
//!
//! A program that advances its arrays with `x = step(&x); x.eval()` builds the same graph, over
//! new buffers, again and again, and every `eval` flattens it again: it walks the graph, looks
//! its program up, cuts it where the cost model says, and launches. Planning is a function of the
//! *structure* of the pending graph (operations, lengths, wiring, which operands are the same
//! buffer, which scalars are equal, the flags the planner reads) and of the configuration; what a
//! later evaluation of the same structure differs in is the buffers it reads and the values of
//! its scalars. [`evaluate`] therefore walks the graph the call is about to evaluate once,
//! hashes what planning depends on into a 128-bit key (see [`Walk`]) and looks the key up:
//!
//! - **hit**: the *plan* found under the key, the list of the store launches the evaluation made
//!   and which nodes each one computed, is run again over the leaves and scalars of this walk
//!   ([`replay`]): the launches go through `launch::run_store` as they did, only flattening and
//!   everything the planner did around it are skipped.
//! - **miss**: the evaluation runs as it always does with a recorder watching ([`Recorder`]): every
//!   `run_store` and every node it finishes is noted. A plan is made of it only if nothing that
//!   the plan could not reproduce happened (see [`taint`]): a decision that looked at state
//!   outside the graph (arrays that are alive elsewhere, the live list), a launch that is not
//!   an elementwise store, a failure, and an end state of the nodes that the recorded steps do not
//!   explain. Such structures are remembered as uncacheable.
//!
//! Results are bit-identical to the uncached path: the same kernels run over the same buffers
//! with the same words in their uniforms. `LUMA_PLAN_CACHE=off` switches the cache off.

use std::cell::{Cell, RefCell};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use crate::array::Storage;
use crate::context::Context;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::profile::{Decision, DecisionSite};

use super::follow;
use super::launch;
use super::program::{Fused, FusedKey};
use super::rerun::Mix;
use super::{current_thread, ElemTy, JitState, Node, Src};

/// Environment variable that switches the cache off (`off`, `0`, `false`, `no`) or makes it
/// print what it does (`trace`).
pub(super) const ENV: &str = "LUMA_PLAN_CACHE";

/// Whether the plan cache is on when the environment variable is `value`: on unless `off`, `0`,
/// `false` or `no`.
pub(super) fn enabled_by(value: Option<&str>) -> bool {
    !value.is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"))
}

/// Plans kept per context (the least recently used goes first).
const MAX_ENTRIES: usize = 128;

/// Nodes of a graph the cache looks at; bigger evaluations are not cached.
const MAX_NODES: usize = 1 << 14;

/// Misses in a row after which the cache sits out evaluations (see `JitState::plan_skip`).
const MISSES_BEFORE_SKIPPING: u32 = 6;

/// Most evaluations skipped after a run of misses.
const MAX_SKIP: u32 = 64;

/// Entries of the walk's tables that stay allocated between walks.
const SCRATCH_KEPT: usize = 1 << 14;

const NONE: u32 = u32::MAX;

/// What the cache did, read by tests.
#[derive(Debug, Default)]
pub(super) struct Stats {
    pub(super) hits: AtomicU64,
    pub(super) recorded: AtomicU64,
    pub(super) uncacheable: AtomicU64,
}

/// A snapshot of [`Stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StatsSnapshot {
    pub(crate) hits: u64,
    pub(crate) recorded: u64,
    pub(crate) uncacheable: u64,
}

impl Stats {
    pub(super) fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            recorded: self.recorded.load(Ordering::Relaxed),
            uncacheable: self.uncacheable.load(Ordering::Relaxed),
        }
    }
}

/// The plans of a context by key.
#[derive(Default)]
pub(super) struct Table {
    map: FxMap<[u64; 2], Entry>,
    clock: u64,
    /// Evaluations in a row that found no plan (a new structure every time cannot be cached: the
    /// walk and the recording are then pure overhead).
    misses: u32,
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Table({} plans)", self.map.len())
    }
}

struct Entry {
    /// `None`: this structure cannot be replayed (why: see [`Recorder::tainted`]).
    plan: Option<Arc<Plan>>,
    used: u64,
}

impl Table {
    fn find(&mut self, key: [u64; 2]) -> Option<Option<Arc<Plan>>> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.map.get_mut(&key)?;
        entry.used = clock;
        Some(entry.plan.clone())
    }

    fn insert(&mut self, key: [u64; 2], plan: Option<Arc<Plan>>) {
        if self.map.len() >= MAX_ENTRIES
            && let Some(oldest) = self.map.iter().min_by_key(|(_, entry)| entry.used).map(|(&key, _)| key)
        {
            self.map.remove(&oldest);
        }
        self.clock += 1;
        self.map.insert(key, Entry { plan, used: self.clock });
    }

    /// Drops every plan.
    pub(super) fn clear(&mut self) {
        self.map.clear();
        self.misses = 0;
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.map.len()
    }

    /// The most launches one plan makes (tests).
    #[cfg(test)]
    pub(super) fn longest(&self) -> usize {
        self.map.values().filter_map(|entry| entry.plan.as_ref()).map(|plan| plan.steps.len()).max().unwrap_or(0)
    }
}

/// Where an input of a recorded launch comes from.
#[derive(Clone, Copy, Debug)]
pub(super) enum Input {
    /// A buffer the walk found, by its position in [`Walk::leaves`].
    Leaf(u32),
    /// Output `out` of an earlier launch of the plan.
    Out { step: u32, out: u32 },
}

/// One recorded `run_store`.
#[derive(Debug)]
pub(super) struct Step {
    key: FusedKey,
    len: usize,
    pub(super) inputs: Vec<Input>,
    /// The words `run_store` got, with those that come from scalars of the graph at 0.
    scalars: Vec<u32>,
    /// (slot, class): the slots to fill with the value of scalar class `class` of the walk.
    patches: Vec<(u32, u32)>,
    /// The nodes the program computed inside itself (see [`Fused::fused_nodes`]).
    fused: Vec<u32>,
    /// (output, node): the nodes the launch's outputs became the results of, in the order the
    /// evaluation finished them.
    pub(super) done: Vec<(u32, u32)>,
}

/// What an evaluation of one structure did.
#[derive(Debug)]
pub(super) struct Plan {
    nodes: usize,
    leaves: usize,
    classes: usize,
    pub(super) steps: Vec<Step>,
}

/// A scalar the walk met: its type and bits.
type Class = (ElemTy, u64);

/// The evaluation of a graph, as the key that tells its structure and what a replay rebinds.
#[derive(Default)]
struct Walk {
    /// The pending nodes of the graph: the roots first, then in the order they were met.
    nodes: Vec<Arc<Node>>,
    /// The buffers the graph reads, in the order they were met.
    leaves: Vec<Arc<Storage>>,
    leaf_at: FxMap<u64, u32>,
    /// The distinct scalar values (by type and bits) in the order they were met.
    classes: Vec<Class>,
    stack: Vec<u32>,
    base: u64,
    dense: Vec<u32>,
    spill: FxMap<u64, u32>,
    mix: Mix,
}

impl Walk {
    fn clear(&mut self) {
        self.nodes.clear();
        self.leaves.clear();
        self.leaf_at.clear();
        self.classes.clear();
        self.stack.clear();
        self.dense.clear();
        self.spill.clear();
        // A walk of a huge graph does not leave its tables behind.
        if self.dense.capacity() > SCRATCH_KEPT {
            self.dense.shrink_to(SCRATCH_KEPT);
        }
        if self.nodes.capacity() > SCRATCH_KEPT {
            self.nodes.shrink_to(SCRATCH_KEPT);
        }
    }

    fn lookup(&self, id: u64) -> u32 {
        match self.dense.get(id.wrapping_sub(self.base) as usize) {
            Some(&at) if id >= self.base => at,
            _ if self.spill.is_empty() => NONE,
            _ => self.spill.get(&id).copied().unwrap_or(NONE),
        }
    }

    fn record(&mut self, id: u64, at: u32) {
        match self.dense.get_mut(id.wrapping_sub(self.base) as usize) {
            Some(slot) if id >= self.base => *slot = at,
            _ => {
                self.spill.insert(id, at);
            }
        }
    }

    /// The position of `node` among `nodes`, adding it (to be visited) if it is new.
    fn add(&mut self, node: &Arc<Node>) -> u32 {
        let known = self.lookup(node.id);
        if known != NONE {
            return known;
        }
        let at = self.nodes.len() as u32;
        self.record(node.id, at);
        self.nodes.push(Arc::clone(node));
        self.stack.push(at);
        at
    }

    /// Notes the buffer `storage` as a leaf of the graph; `None` for one a replay cannot rebind.
    fn leaf(&mut self, storage: &Arc<Storage>) -> Option<u32> {
        if let Some(&at) = self.leaf_at.get(&storage.id()) {
            return Some(at);
        }
        if storage.segments() != 1 || storage.is_alias() || storage.is_window() {
            return None;
        }
        let at = self.leaves.len() as u32;
        self.leaf_at.insert(storage.id(), at);
        self.leaves.push(Arc::clone(storage));
        self.mix.word(5);
        self.mix.word(storage.len() as u64);
        storage.layout().hash(&mut self.mix);
        Some(at)
    }

    /// Walks the graph below `roots` and hashes what planning depends on; `None` if the graph is
    /// not one a replay can run (see [`Walk::visit`]).
    fn run(&mut self, jit: &JitState, ctx: &Context, roots: &[Arc<Node>]) -> Option<[u64; 2]> {
        // Loops' and reductions' nodes hold no operands to walk (the commonest reason to give up).
        if roots.iter().any(|root| root.looped.get().is_some() || root.deferred.get().is_some()) {
            return None;
        }
        self.clear();
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for root in roots {
            let (first, last) = root.below_span().unwrap_or((root.id, root.id));
            lo = lo.min(first).min(root.id);
            hi = hi.max(last).max(root.id);
        }
        let width = hi.checked_sub(lo).and_then(|span| usize::try_from(span).ok()).map_or(0, |span| span.saturating_add(1));
        let expected: usize = roots.iter().map(|root| root.size()).sum();
        if width != 0 && width <= expected.saturating_mul(64).saturating_add(1024) {
            self.base = lo;
            self.dense.resize(width, NONE);
        } else {
            self.base = 0;
        }
        self.mix = Mix::new(stamp(jit, ctx));
        self.mix.word(roots.len() as u64);
        for root in roots {
            self.add(root);
        }
        let seeds = self.nodes.len();
        // The roots are visited first, in order.
        self.stack.reverse();
        debug_assert_eq!(self.stack.len(), seeds);
        while let Some(at) = self.stack.pop() {
            self.visit(at)?;
            if self.nodes.len() > MAX_NODES {
                return None;
            }
        }
        self.seal(jit)
    }

    /// Hashes node `at` and finds its operands. `None` if the node is not plain: another thread's,
    /// being evaluated, a loop's or a reduction's, or one of its operands is a failure or a
    /// buffer a replay cannot rebind (split over segments, a window, a view of a state).
    fn visit(&mut self, at: u32) -> Option<()> {
        // SAFETY: `self.nodes` holds a reference to the node for the whole walk and only grows, so
        // the node outlives this function (a `Vec` that reallocates moves the `Arc`s, not the
        // nodes they point to).
        let node: &Node = unsafe { &*Arc::as_ptr(&self.nodes[at as usize]) };
        if node.thread != current_thread() || node.looped.get().is_some() || node.deferred.get().is_some() {
            return None;
        }
        let guard = lock(&node.args);
        let args = guard.as_ref()?;
        let arity = node.op.kind.arity();
        let flags = u64::from(node.requested.load(Ordering::Relaxed))
            | u64::from(node.fused.load(Ordering::Relaxed)) << 1
            | u64::from(node.handles.load(Ordering::Relaxed) > 0) << 2
            | u64::from(node.shifted_reads.load(Ordering::Relaxed) == 0) << 3;
        let mix = &mut self.mix;
        mix.word(node.op.fingerprint);
        mix.word(node.len as u64);
        mix.word(node.weight.to_bits());
        mix.word(node.below_len as u64 | u64::from(node.depth) << 32 | u64::from(node.shift_depth) << 48);
        mix.word(flags);
        for arg in args.iter().take(arity) {
            let word = match arg {
                Src::Node(child) => match child.result.get() {
                    Some(Ok(storage)) => 2 << 60 | u64::from(self.leaf(storage)?),
                    Some(Err(_)) => return None,
                    None => 1 << 60 | u64::from(self.add(child)),
                },
                Src::Buffer(storage) => 2 << 60 | u64::from(self.leaf(storage)?),
                Src::Scalar(scalar) => {
                    // The words of a wide scalar are placed next to one another in the uniform,
                    // where they cannot be told from scalars of their own.
                    if scalar.ty.slots() != 1 {
                        return None;
                    }
                    let class = (scalar.ty, scalar.bits);
                    let at = match self.classes.iter().position(|known| *known == class) {
                        Some(at) => at,
                        None => {
                            self.classes.push(class);
                            self.classes.len() - 1
                        }
                    };
                    3 << 60 | (scalar.ty as u64) << 32 | at as u64
                }
            };
            self.mix.word(word);
        }
        Some(())
    }

    /// Hashes what is known only once every node was visited: which scalars equal a value the
    /// flattening makes up (so that they share a uniform slot with it); and finds out that
    /// nothing outside the graph is pending.
    fn seal(&mut self, jit: &JitState) -> Option<[u64; 2]> {
        let mut lens: Vec<u64> = self.nodes.iter().map(|node| node.len as u64).collect();
        lens.sort_unstable();
        lens.dedup();
        let mix = &mut self.mix;
        for &(ty, bits) in &self.classes {
            // The flattening makes up a `u32` zero and the length of what a circular shift reads.
            let special = ty == ElemTy::U32 && (bits == 0 || lens.binary_search(&bits).is_ok());
            mix.word(6 | (ty as u64) << 8 | u64::from(special) << 16);
            if special {
                mix.word(bits);
            }
        }
        // Planning looks at the other arrays that are alive (what shares work with an expression
        // that is cut is evaluated alongside it, and who reads a node decides whether it is
        // computed once): a replay is right only if there are none, that is if the graph is all
        // the pending nodes there are.
        if jit.pending_nodes.load(Ordering::Relaxed) != self.nodes.len() as i64 {
            return None;
        }
        self.mix.word(self.nodes.len() as u64);
        Some([self.mix.a, self.mix.b])
    }
}

/// What of the environment decides how a graph is planned (see [`JitState::plan_epoch`] for the
/// settings that only tests change).
pub(super) fn stamp(jit: &JitState, ctx: &Context) -> u64 {
    let mut mix = Mix::new(jit.plan_epoch.load(Ordering::Relaxed));
    mix.word(jit.max_nodes() as u64);
    mix.word(jit.max_inputs() as u64);
    mix.word(jit.max_live(ctx.tuning()) as u64);
    mix.word(jit.max_inlined(ctx.tuning()) as u64);
    mix.word(u64::from(ctx.fast_math()));
    // The cost model decides where expressions are cut and what is stored (it is measured when
    // the context opens, or set by tests).
    let cost = ctx.cost_model();
    for number in [cost.launch_us, cost.bandwidth_gbs, cost.f32_ops_per_ns, cost.df32_factor, cost.f64_factor.unwrap_or(-1.0)] {
        mix.word(number.to_bits());
    }
    mix.a ^ mix.b
}

/// Whether evaluations of `ctx` may use the cache now (the cheapest checks first: this runs for
/// every `eval`).
pub(super) fn usable(jit: &JitState, ctx: &Context) -> bool {
    jit.plan_cache.load(Ordering::Relaxed)
        && !recording_has(WALKING)
        && !ctx.check_finite()
        && !ctx.recording_here()
        && !ctx.is_profiling()
        && !ctx.can_run_on_host()
}

thread_local! {
    /// Which recordings this thread has under way: [`WALKING`] and [`TAPING`].
    static RECORDING: Cell<u8> = const { Cell::new(0) };
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    /// The recording of a whole step that trace following is making (see [`super::follow`]): the
    /// launches of its creates and of its evaluation, by position in the tape.
    static TAPE: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    static SCRATCH: RefCell<Walk> = RefCell::new(Walk::default());
}

/// `RECORDING`: the plan cache is recording an evaluation (see [`Watching`]).
const WALKING: u8 = 1;
/// `RECORDING`: trace following is recording a step.
const TAPING: u8 = 2;

fn recording_has(flag: u8) -> bool {
    RECORDING.with(Cell::get) & flag != 0
}

fn recording_set(flag: u8, on: bool) {
    RECORDING.with(|recording| recording.set(if on { recording.get() | flag } else { recording.get() & !flag }));
}

/// Whether this thread's launches are being recorded, by the plan cache or by a tape.
pub(super) fn recording() -> bool {
    RECORDING.with(Cell::get) != 0
}

/// Marks the recordings (if any) as ones that cannot become a plan, because something happens that
/// a replay would not reproduce.
pub(super) fn taint(why: &'static str) {
    if recording_has(WALKING) {
        RECORDER.with_borrow_mut(|recorder| {
            if let Some(recorder) = recorder
                && recorder.tainted.is_none()
            {
                recorder.tainted = Some(why);
            }
        });
    }
    taint_tape(why);
}

/// [`taint`] for the tape's recording alone.
pub(super) fn taint_tape(why: &'static str) {
    if recording_has(TAPING) {
        TAPE.with_borrow_mut(|recorder| {
            if let Some(recorder) = recorder
                && recorder.tainted.is_none()
            {
                recorder.tainted = Some(why);
            }
        });
    }
}

/// What `run_store` hands back to [`end_store`]: the step each recorder made of the launch.
#[derive(Clone, Copy)]
pub(super) struct StoreToken {
    walk: Option<usize>,
    tape: Option<usize>,
}

/// Starts the recording of a step for trace following.
pub(super) fn tape_begin() {
    TAPE.with_borrow_mut(|slot| *slot = Some(Recorder::open()));
    recording_set(TAPING, true);
}

/// Ends the recording of a step: the recorder, if there was one.
pub(super) fn tape_end() -> Option<Recorder> {
    recording_set(TAPING, false);
    TAPE.with_borrow_mut(Option::take)
}

/// Ends the recording of a step without making anything of it.
pub(super) fn tape_abort() {
    let _ = RECORDING.try_with(|recording| recording.set(recording.get() & !TAPING));
    let _ = TAPE.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            slot.take();
        }
    });
}

/// Runs `f` on the tape's recorder, if one is recording.
pub(super) fn with_tape<R>(f: impl FnOnce(&mut Recorder) -> R) -> Option<R> {
    if !recording_has(TAPING) {
        return None;
    }
    TAPE.with_borrow_mut(|recorder| recorder.as_mut().map(f))
}

/// Watches an evaluation that no plan covers, to make one.
pub(super) struct Recorder {
    nodes: Vec<Weak<Node>>,
    node_at: FxMap<u64, u32>,
    leaf_at: FxMap<u64, u32>,
    classes: Vec<Class>,
    lens: Vec<u64>,
    fused_before: Vec<bool>,
    steps: Vec<Step>,
    /// The step and output each new buffer is.
    produced: FxMap<u64, (u32, u32)>,
    done: Vec<bool>,
    fused_marks: Vec<bool>,
    /// Nodes a replay of the plan cache marked fused before the launch that is about to be made.
    pending_fused: Vec<u32>,
    leaves: usize,
    tainted: Option<&'static str>,
}

impl Recorder {
    /// A recorder for a whole step of trace following: nodes, buffers and scalars are told to it as
    /// the step makes them, in the order of the tape.
    fn open() -> Recorder {
        Recorder {
            nodes: Vec::new(),
            node_at: FxMap::default(),
            leaf_at: FxMap::default(),
            classes: Vec::new(),
            lens: Vec::new(),
            fused_before: Vec::new(),
            steps: Vec::new(),
            produced: FxMap::default(),
            done: Vec::new(),
            fused_marks: Vec::new(),
            pending_fused: Vec::new(),
            leaves: 0,
            tainted: None,
        }
    }

    /// The step (the launches so far) a node, buffer or scalar is told in.
    pub(super) fn steps_len(&self) -> usize {
        self.steps.len()
    }

    /// Notes the node the entry that comes next made.
    pub(super) fn note_node(&mut self, node: &Arc<Node>) -> u32 {
        let at = self.nodes.len() as u32;
        self.nodes.push(Arc::downgrade(node));
        self.node_at.insert(node.id, at);
        self.fused_before.push(node.fused.load(Ordering::Relaxed));
        self.done.push(false);
        self.fused_marks.push(false);
        let len = node.len as u64;
        if let Err(place) = self.lens.binary_search(&len) {
            self.lens.insert(place, len);
        }
        at
    }

    /// Notes an entry that made no node (hash-consing handed one out).
    pub(super) fn note_no_node(&mut self) {
        self.nodes.push(Weak::new());
        self.fused_before.push(false);
        self.done.push(false);
        self.fused_marks.push(false);
    }

    /// The entry that made the node with this id.
    pub(super) fn node_index(&self, id: u64) -> Option<u32> {
        self.node_at.get(&id).copied()
    }

    /// The leaf with the buffer with this id, and the output of a step it is, if it was made by one.
    pub(super) fn leaf_index(&self, id: u64) -> Option<u32> {
        self.leaf_at.get(&id).copied()
    }

    pub(super) fn produced_by(&self, id: u64) -> Option<(u32, u32)> {
        self.produced.get(&id).copied()
    }

    pub(super) fn note_leaf(&mut self, id: u64) -> u32 {
        let at = self.leaves as u32;
        self.leaf_at.insert(id, at);
        self.leaves += 1;
        at
    }

    pub(super) fn note_class(&mut self, class: Class) -> u32 {
        self.classes.push(class);
        self.classes.len() as u32 - 1
    }

    /// How many references each node noted so far holds (0 for a call that made none).
    pub(super) fn strong_counts(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.strong_count() as u32).collect()
    }

    /// The plan of a recorded step, if it can be one (see [`Recorder::finish`]).
    pub(super) fn into_plan(self) -> Result<Plan, &'static str> {
        self.finish()
    }

    fn new(walk: &Walk) -> Recorder {
        let mut lens: Vec<u64> = walk.nodes.iter().map(|node| node.len as u64).collect();
        lens.sort_unstable();
        lens.dedup();
        Recorder {
            nodes: walk.nodes.iter().map(Arc::downgrade).collect(),
            node_at: walk.nodes.iter().enumerate().map(|(at, node)| (node.id, at as u32)).collect(),
            leaf_at: walk.leaf_at.clone(),
            classes: walk.classes.clone(),
            lens,
            fused_before: walk.nodes.iter().map(|node| node.fused.load(Ordering::Relaxed)).collect(),
            steps: Vec::new(),
            produced: FxMap::default(),
            done: vec![false; walk.nodes.len()],
            fused_marks: vec![false; walk.nodes.len()],
            pending_fused: Vec::new(),
            leaves: walk.leaves.len(),
            tainted: None,
        }
    }

    /// The step for a `run_store` of `len` elements of `fused`, or `Err` why it cannot be one.
    fn step(&mut self, len: usize, fused: &Fused) -> Result<Step, &'static str> {
        let key = &fused.key;
        let inputs = fused
            .inputs
            .iter()
            .map(|input| match (self.leaf_at.get(&input.id()), self.produced.get(&input.id())) {
                (Some(&at), _) => Ok(Input::Leaf(at)),
                (None, Some(&(step, out))) => Ok(Input::Out { step, out }),
                (None, None) => Err("an input that is neither a buffer of the graph nor a result of the evaluation"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut scalars = fused.scalars.clone();
        let mut patches = Vec::new();
        for (slot, (&word, &ty)) in fused.scalars.iter().zip(&key.scalar_tys).enumerate() {
            match self.classes.iter().position(|&(class_ty, bits)| class_ty == ty && bits as u32 == word && ty.slots() == 1) {
                Some(class) => {
                    scalars[slot] = 0;
                    patches.push((slot as u32, class as u32));
                }
                // The guard of the double-float library and what the flattening makes up.
                None if ty == ElemTy::U32 && (word == 0 || self.lens.binary_search(&u64::from(word)).is_ok()) => {}
                None => return Err("a uniform word that is not a scalar of the graph"),
            }
        }
        let fused_nodes = fused
            .fused_nodes
            .iter()
            .map(|node| self.node_at.get(&node.id).copied().ok_or("a fused node outside the graph"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Step { key: key.clone(), len, inputs, scalars, patches, fused: fused_nodes, done: Vec::new() })
    }

    /// Makes the plan of what was recorded, if it can be one: every node of the graph is in the
    /// state the steps leave it in.
    fn finish(self) -> Result<Plan, &'static str> {
        if let Some(why) = self.tainted {
            return Err(why);
        }
        if self.steps.is_empty() {
            return Err("nothing was launched");
        }
        for (at, node) in self.nodes.iter().enumerate() {
            // A node that is gone cannot be looked at; the steps still say what happened to it.
            let Some(node) = node.upgrade() else { continue };
            if node.result.get().is_some() != self.done[at] {
                return Err("a node was finished or left pending by something the steps do not explain");
            }
            if node.fused.load(Ordering::Relaxed) != (self.fused_before[at] || self.fused_marks[at]) {
                return Err("a node was marked fused by something the steps do not explain");
            }
        }
        if self.steps.iter().enumerate().any(|(s, step)| {
            let outputs = step.key.outputs.len();
            (0..outputs).any(|out| !step.done.iter().any(|&(o, _)| o as usize == out)) || self.produced.values().filter(|&&(from, _)| from as usize == s).count() != outputs
        }) {
            return Err("an output that no node took");
        }
        Ok(Plan { nodes: self.nodes.len(), leaves: self.leaves, classes: self.classes.len(), steps: self.steps })
    }
}

/// Called by `run_store` before it runs `fused` (while recording): the step it will be.
pub(super) fn begin_store(len: usize, fused: &Fused) -> StoreToken {
    fn note(slot: &'static std::thread::LocalKey<RefCell<Option<Recorder>>>, len: usize, fused: &Fused) -> Option<usize> {
        slot.with_borrow_mut(|recorder| {
            let recorder = recorder.as_mut()?;
            if recorder.tainted.is_some() {
                return None;
            }
            match recorder.step(len, fused) {
                Ok(mut step) => {
                    step.fused.append(&mut recorder.pending_fused);
                    for &at in &step.fused {
                        recorder.fused_marks[at as usize] = true;
                    }
                    recorder.steps.push(step);
                    Some(recorder.steps.len() - 1)
                }
                Err(why) => {
                    recorder.tainted = Some(why);
                    None
                }
            }
        })
    }
    StoreToken {
        walk: if recording_has(WALKING) { note(&RECORDER, len, fused) } else { None },
        tape: if recording_has(TAPING) { note(&TAPE, len, fused) } else { None },
    }
}

/// Called by `run_store` when it returns while recording: what the step wrote.
pub(super) fn end_store<E>(token: StoreToken, outputs: &Result<Vec<Storage>, E>) {
    fn note<E>(slot: &'static std::thread::LocalKey<RefCell<Option<Recorder>>>, step: Option<usize>, outputs: &Result<Vec<Storage>, E>) {
        let Some(step) = step else { return };
        slot.with_borrow_mut(|recorder| {
            let Some(recorder) = recorder.as_mut() else { return };
            match outputs {
                Ok(outputs) => {
                    for (out, storage) in outputs.iter().enumerate() {
                        recorder.produced.insert(storage.id(), (step as u32, out as u32));
                    }
                    // The number of outputs a plan expects is that of the program.
                    if outputs.len() != recorder.steps[step].key.outputs.len() && recorder.tainted.is_none() {
                        recorder.tainted = Some("a launch with other outputs than its program");
                    }
                }
                Err(_) => {
                    if recorder.tainted.is_none() {
                        recorder.tainted = Some("a launch failed");
                    }
                }
            }
        });
    }
    note(&RECORDER, token.walk, outputs);
    note(&TAPE, token.tape, outputs);
}

/// Called when `node` got `storage` as its result while recording.
#[inline(always)]
pub(super) fn finished(node: &Node, storage: &Storage) {
    if RECORDING.with(Cell::get) != 0 {
        finished_recording(node, storage);
    }
}

#[cold]
#[inline(never)]
fn finished_recording(node: &Node, storage: &Storage) {
    fn note(slot: &'static std::thread::LocalKey<RefCell<Option<Recorder>>>, node: &Node, storage: &Storage) {
        slot.with_borrow_mut(|recorder| {
            let Some(recorder) = recorder.as_mut() else { return };
            if recorder.tainted.is_some() {
                return;
            }
            let (Some(&(step, out)), Some(&at)) = (recorder.produced.get(&storage.id()), recorder.node_at.get(&node.id)) else {
                recorder.tainted = Some("a node finished with a buffer that no recorded launch wrote, or a node outside the graph");
                return;
            };
            if recorder.done[at as usize] {
                recorder.tainted = Some("a node finished twice");
                return;
            }
            recorder.done[at as usize] = true;
            recorder.steps[step as usize].done.push((out, at));
        });
    }
    let recording = RECORDING.with(Cell::get);
    if recording & WALKING != 0 {
        note(&RECORDER, node, storage);
    }
    if recording & TAPING != 0 {
        note(&TAPE, node, storage);
    }
}

/// Evaluates `pending` (the nodes `eval` asked for that are still pending, all of this thread's)
/// with `run`, which plans and launches, or from the plan an earlier evaluation of the same
/// structure left.
pub(super) fn evaluate(jit: &JitState, pending: Vec<Arc<Node>>, run: impl FnOnce(Vec<Arc<Node>>)) {
    // A step that follows its tape replays without making a plan; one that was recorded is
    // complete when its evaluation is.
    let mut recording = None;
    if jit.follow_on() {
        match follow::eval_begin(jit, &pending) {
            follow::Eval::Replayed => return,
            follow::Eval::Failed => return run(pending.into_iter().filter(|node| node.is_pending()).collect()),
            follow::Eval::Normal(made) => recording = made,
        }
    }
    evaluate_planned(jit, pending, run);
    if recording.is_some() {
        follow::eval_end(jit, recording);
    }
}

/// [`evaluate`] with the plan cache.
#[inline(always)]
fn evaluate_planned(jit: &JitState, pending: Vec<Arc<Node>>, run: impl FnOnce(Vec<Arc<Node>>)) {
    let Some(first) = pending.first() else { return run(pending) };
    // A tape is made of an evaluation the plan cache could have cached, and of no other.
    let skip = jit.plan_skip.load(Ordering::Relaxed);
    if skip > 0 {
        jit.plan_skip.store(skip - 1, Ordering::Relaxed);
        taint_tape("the plan cache sits out evaluations");
        return run(pending);
    }
    if !usable(jit, first.context()) {
        taint_tape("the plan cache cannot be used");
        return run(pending);
    }
    let ctx = first.context().clone();
    // The graph can be all of the pending nodes only if they are not more than the roots stand
    // above (an upper bound: `below_len` counts what was evaluated since too).
    let most: i64 = pending.iter().map(|root| 1 + root.below_len() as i64).sum();
    if jit.pending_nodes.load(Ordering::Relaxed) > most {
        note_miss(jit);
        taint_tape("other arrays are pending");
        return run(pending);
    }
    let mut walk = SCRATCH.with(|scratch| std::mem::take(&mut *scratch.borrow_mut()));
    let found = walk.run(jit, &ctx, &pending).map(|key| (key, lock(&jit.plans).find(key)));
    match found {
        // Not a graph the cache can handle: nothing is remembered.
        None => {
            note_miss(jit);
            taint_tape("the graph is not one the plan cache can walk");
            walk.clear();
            run(pending);
        }
        Some((_, Some(Some(plan)))) if plan.nodes == walk.nodes.len() && plan.leaves == walk.leaves.len() && plan.classes == walk.classes.len() => {
            lock(&jit.plans).misses = 0;
            // A replay makes no planning decisions (the launches re-derive what the plan holds): it
            // records one `PlanReplay` instead.
            let recording = jit.recording_decisions();
            if recording {
                jit.pause_decisions(true);
            }
            let replayed = replay(&ctx, &plan, &walk);
            if recording {
                jit.pause_decisions(false);
                let first = &pending[0];
                if replayed.is_ok() {
                    jit.note_decision(Decision::new(first.name, first.location, DecisionSite::PlanReplay, "replay"));
                }
            }
            if replayed.is_ok() {
                jit.plan_stats.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                taint_tape("the plan cache's replay failed");
                // Whatever the plan finished is finished; the rest is evaluated as ever.
                walk.clear();
                run(pending.into_iter().filter(|node| node.is_pending()).collect());
            }
        }
        // A structure that cannot be replayed (or a key that matched a plan of another shape).
        Some((_, Some(_))) => {
            note_miss(jit);
            taint_tape("the structure is one the plan cache cannot replay");
            walk.clear();
            run(pending);
        }
        Some((key, None)) => {
            note_miss(jit);
            let recorder = Recorder::new(&walk);
            walk.clear();
            let watching = Watching::begin(recorder);
            run(pending);
            if let Some(recorder) = watching.end() {
                match recorder.finish() {
                    Ok(plan) => {
                        if trace() {
                            eprintln!("luma plan cache: recorded a plan of {} launches over {} nodes", plan.steps.len(), plan.nodes);
                        }
                        jit.plan_stats.recorded.fetch_add(1, Ordering::Relaxed);
                        lock(&jit.plans).insert(key, Some(Arc::new(plan)));
                    }
                    Err(why) => {
                        if trace() {
                            eprintln!("luma plan cache: not cacheable: {why}");
                        }
                        jit.plan_stats.uncacheable.fetch_add(1, Ordering::Relaxed);
                        lock(&jit.plans).insert(key, None);
                    }
                }
            }
        }
    }
    walk.clear();
    SCRATCH.with(|scratch| *scratch.borrow_mut() = walk);
}

/// The recording of this thread, which ends when this is dropped, whatever happens in between.
struct Watching;

impl Watching {
    fn begin(recorder: Recorder) -> Watching {
        RECORDER.with_borrow_mut(|slot| *slot = Some(recorder));
        recording_set(WALKING, true);
        Watching
    }

    fn end(self) -> Option<Recorder> {
        recording_set(WALKING, false);
        RECORDER.with_borrow_mut(Option::take)
    }
}

impl Drop for Watching {
    fn drop(&mut self) {
        recording_set(WALKING, false);
        RECORDER.with_borrow_mut(|slot| slot.take());
    }
}

/// Whether `LUMA_PLAN_CACHE=trace`.
fn trace() -> bool {
    static TRACE: LazyLock<bool> = LazyLock::new(|| std::env::var(ENV).is_ok_and(|value| value.trim().eq_ignore_ascii_case("trace")));
    *TRACE
}

/// Notes an evaluation that found no plan, and makes the cache sit out the next ones after a run
/// of misses.
fn note_miss(jit: &JitState) {
    let mut table = lock(&jit.plans);
    table.misses = table.misses.saturating_add(1);
    if table.misses >= MISSES_BEFORE_SKIPPING {
        jit.plan_skip.store((1u32 << (table.misses - MISSES_BEFORE_SKIPPING).min(6)).min(MAX_SKIP), Ordering::Relaxed);
    }
}

/// Notes an evaluation that a followed tape replayed: to the cache it is a hit (the evaluation
/// would have found its plan), so it ends a run of misses as a hit does.
pub(super) fn note_replayed(jit: &JitState) {
    lock(&jit.plans).misses = 0;
}

/// Runs `plan` over the leaves and scalars of `walk`.
fn replay(ctx: &Context, plan: &Plan, walk: &Walk) -> Result<(), ()> {
    let class_words: Vec<u32> = walk.classes.iter().map(|&(_, bits)| bits as u32).collect();
    let mut outs: Vec<Vec<Arc<Storage>>> = Vec::with_capacity(plan.steps.len());
    let taping = recording_has(TAPING);
    for step in &plan.steps {
        let inputs: Vec<Arc<Storage>> = step
            .inputs
            .iter()
            .map(|input| match *input {
                Input::Leaf(at) => Arc::clone(&walk.leaves[at as usize]),
                Input::Out { step, out } => Arc::clone(&outs[step as usize][out as usize]),
            })
            .collect();
        let mut scalars = step.scalars.clone();
        for &(slot, class) in &step.patches {
            scalars[slot as usize] = class_words[class as usize];
        }
        for &at in &step.fused {
            let node = &walk.nodes[at as usize];
            node.fused.store(true, Ordering::Relaxed);
            if taping {
                // A tape that is being recorded learns which nodes the launch computed inside itself.
                with_tape(|recorder| recorder.pending_fused.extend(recorder.node_at.get(&node.id).copied()));
            }
        }
        let fused = Fused { key: step.key.clone(), inputs, scalars, fused_nodes: Vec::new(), instr_nodes: Vec::new(), sub_roots: Vec::new() };
        let outputs = launch::run_store(ctx, step.len, fused).map_err(|_| ())?;
        let outputs: Vec<Arc<Storage>> = outputs.into_iter().map(Arc::new).collect();
        for &(out, at) in &step.done {
            walk.nodes[at as usize].finish_shared(Arc::clone(&outputs[out as usize]));
        }
        outs.push(outputs);
    }
    Ok(())
}

/// What replaying a step of a tape rebinds: the buffers, scalar words and nodes of the step being
/// followed, by position in the tape (see [`super::follow`]).
pub(super) trait Rebind {
    fn leaf(&self, at: u32) -> Option<Arc<Storage>>;
    fn word(&self, class: u32) -> u32;
    /// The node of an entry that is still alive.
    fn node(&self, at: u32) -> Option<Arc<Node>>;
}

impl Plan {
    /// The leaves a launch could take over the buffer of if nothing else held it (see
    /// `launch::run_store`): inputs of launches with one output that the program can overwrite.
    pub(super) fn takeover_leaves(&self) -> Vec<u32> {
        let mut leaves = Vec::new();
        for step in &self.steps {
            if step.key.outputs.len() != 1 {
                continue;
            }
            for (k, input) in step.inputs.iter().enumerate() {
                if let Input::Leaf(at) = *input
                    && launch::in_place_eligible(&step.key, k as u32)
                    && !leaves.contains(&at)
                {
                    leaves.push(at);
                }
            }
        }
        leaves
    }
}

/// Replays `step` over what `with` rebinds. `outs` holds the outputs of the steps before it (and
/// gets its own).
pub(super) fn run_step(ctx: &Context, step: &Step, outs: &mut Vec<Vec<Arc<Storage>>>, with: &dyn Rebind) -> Result<(), ()> {
    let mut inputs: Vec<Arc<Storage>> = Vec::with_capacity(step.inputs.len());
    for input in &step.inputs {
        inputs.push(match *input {
            Input::Leaf(at) => with.leaf(at).ok_or(())?,
            Input::Out { step, out } => Arc::clone(outs.get(step as usize).and_then(|outputs| outputs.get(out as usize)).ok_or(())?),
        });
    }
    let mut scalars = step.scalars.clone();
    for &(slot, class) in &step.patches {
        scalars[slot as usize] = with.word(class);
    }
    for &at in &step.fused {
        if let Some(node) = with.node(at) {
            node.fused.store(true, Ordering::Relaxed);
        }
    }
    let fused = Fused { key: step.key.clone(), inputs, scalars, fused_nodes: Vec::new(), instr_nodes: Vec::new(), sub_roots: Vec::new() };
    let outputs = launch::run_store(ctx, step.len, fused).map_err(|_| ())?;
    let outputs: Vec<Arc<Storage>> = outputs.into_iter().map(Arc::new).collect();
    for &(out, at) in &step.done {
        if let Some(node) = with.node(at) {
            node.finish_shared(Arc::clone(&outputs[out as usize]));
        }
    }
    outs.push(outputs);
    Ok(())
}

/// The key a walk of `pending` makes (what `follow` checks its tapes against, see `LUMA_TRACE_FOLLOW=verify`).
pub(super) fn walk_key(jit: &JitState, ctx: &Context, pending: &[Arc<Node>]) -> Option<[u64; 2]> {
    let mut walk = SCRATCH.with(|scratch| std::mem::take(&mut *scratch.borrow_mut()));
    let key = walk.run(jit, ctx, pending);
    walk.clear();
    SCRATCH.with(|scratch| *scratch.borrow_mut() = walk);
    key
}
