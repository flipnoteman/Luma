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

use super::launch;
use super::program::{Fused, FusedKey};
use super::rerun::Mix;
use super::{current_thread, ElemTy, JitState, Node, Src};

/// Environment variable that switches the cache off (`off`, `0`, `false`, `no`) or makes it
/// print what it does (`trace`).
pub(super) const ENV: &str = "LUMA_PLAN_CACHE";

/// Plans kept per context (the least recently used goes first).
const MAX_ENTRIES: usize = 128;

/// Nodes of a graph the cache looks at; bigger evaluations are not cached.
const MAX_NODES: usize = 1 << 14;

/// Misses in a row after which the cache sits out evaluations (see [`Table::skip`]).
const MISSES_BEFORE_SKIPPING: u32 = 6;

/// Most evaluations skipped after a run of misses.
const MAX_SKIP: u32 = 64;

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
    /// Evaluations the cache sits out because of `misses`.
    skip: u32,
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
        self.skip = 0;
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.map.len()
    }
}

/// Where an input of a recorded launch comes from.
#[derive(Clone, Copy, Debug)]
enum Input {
    /// A buffer the walk found, by its position in [`Walk::leaves`].
    Leaf(u32),
    /// Output `out` of an earlier launch of the plan.
    Out { step: u32, out: u32 },
}

/// One recorded `run_store`.
#[derive(Debug)]
struct Step {
    key: FusedKey,
    len: usize,
    inputs: Vec<Input>,
    /// The words `run_store` got, with those that come from scalars of the graph at 0.
    scalars: Vec<u32>,
    /// (slot, class): the slots to fill with the value of scalar class `class` of the walk.
    patches: Vec<(u32, u32)>,
    /// The nodes the program computed inside itself (see [`Fused::fused_nodes`]).
    fused: Vec<u32>,
    /// (output, node): the nodes the launch's outputs became the results of, in the order the
    /// evaluation finished them.
    done: Vec<(u32, u32)>,
}

/// What an evaluation of one structure did.
#[derive(Debug)]
pub(super) struct Plan {
    nodes: usize,
    leaves: usize,
    classes: usize,
    steps: Vec<Step>,
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
    /// How many nodes of the graph read each node.
    readers: Vec<u32>,
    stack: Vec<u32>,
    base: u64,
    dense: Vec<u32>,
    spill: FxMap<u64, u32>,
    mix: Option<Mix>,
}

impl Walk {
    fn clear(&mut self) {
        self.nodes.clear();
        self.leaves.clear();
        self.leaf_at.clear();
        self.classes.clear();
        self.readers.clear();
        self.stack.clear();
        self.dense.clear();
        self.spill.clear();
    }

    fn mix(&mut self) -> &mut Mix {
        self.mix.get_or_insert_with(|| Mix::new(0))
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
        self.readers.push(0);
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
        let mix = self.mix();
        mix.word(5);
        mix.word(storage.len() as u64);
        storage.layout().hash(mix);
        Some(at)
    }

    /// Walks the graph below `roots` and hashes what planning depends on; `None` if the graph is
    /// not one a replay can run (see [`Walk::visit`]).
    fn run(&mut self, jit: &JitState, ctx: &Context, roots: &[Arc<Node>]) -> Option<[u64; 2]> {
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
        self.mix = Some(Mix::new(stamp(jit, ctx)));
        self.mix().word(roots.len() as u64);
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
        let node = Arc::clone(&self.nodes[at as usize]);
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
        let mix = self.mix();
        mix.word(node.op.fingerprint);
        mix.word(node.len as u64);
        mix.word(node.weight.to_bits());
        mix.word(node.below_len as u64 | u64::from(node.depth) << 32 | u64::from(node.shift_depth) << 48 | flags << 52);
        for (slot, arg) in args.iter().take(arity).enumerate() {
            let repeated = args[..slot].iter().any(|other| match (arg, other) {
                (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
                (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
                _ => false,
            });
            let word = match arg {
                Src::Node(child) => match child.result.get() {
                    Some(Ok(storage)) => 2 << 60 | u64::from(self.leaf(storage)?),
                    Some(Err(_)) => return None,
                    None => {
                        let child_at = self.add(child);
                        if !repeated {
                            self.readers[child_at as usize] += 1;
                        }
                        1 << 60 | u64::from(child_at)
                    }
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
            self.mix().word(word);
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
        let mut classes = std::mem::take(&mut self.classes);
        let mix = self.mix();
        for &(ty, bits) in &classes {
            // The flattening makes up a `u32` zero and the length of what a circular shift reads.
            let special = ty == ElemTy::U32 && (bits == 0 || lens.binary_search(&bits).is_ok());
            mix.word(6 | (ty as u64) << 8 | u64::from(special) << 16);
            if special {
                mix.word(bits);
            }
        }
        std::mem::swap(&mut classes, &mut self.classes);
        // Planning looks at the other arrays that are alive (what shares work with an expression
        // that is cut is evaluated alongside it, and who reads a node decides whether it is
        // computed once): a replay is right only if there are none, that is if the graph is all
        // the pending nodes there are.
        if jit.pending_nodes.load(Ordering::Relaxed) != self.nodes.len() as i64 {
            return None;
        }
        let mix = self.mix.as_mut()?;
        mix.word(self.nodes.len() as u64);
        Some([mix.a, mix.b])
    }
}

/// What of the environment decides how a graph is planned (see [`JitState::plan_epoch`] for the
/// settings that only tests change).
fn stamp(jit: &JitState, ctx: &Context) -> u64 {
    let mut mix = Mix::new(jit.plan_epoch.load(Ordering::Relaxed));
    mix.word(jit.max_nodes() as u64);
    mix.word(jit.max_inputs() as u64);
    mix.word(jit.max_live(ctx.tuning()) as u64);
    mix.word(jit.max_inlined(ctx.tuning()) as u64);
    mix.word(u64::from(ctx.fast_math()));
    mix.a ^ mix.b
}

/// Whether evaluations of `ctx` may use the cache now.
fn usable(jit: &JitState, ctx: &Context) -> bool {
    jit.plan_cache.load(Ordering::Relaxed)
        && !ACTIVE.with(Cell::get)
        && !ctx.recording_here()
        && !ctx.is_profiling()
        && !ctx.check_finite()
        && !ctx.can_run_on_host()
        && lock(&jit.splits).is_none()
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    static SCRATCH: RefCell<Walk> = RefCell::new(Walk::default());
}

/// Whether this thread's evaluation is being recorded.
pub(super) fn recording() -> bool {
    ACTIVE.with(Cell::get)
}

/// Marks the recording (if any) as one that cannot become a plan, because something happens that
/// a replay would not reproduce.
pub(super) fn taint(why: &'static str) {
    if recording() {
        RECORDER.with_borrow_mut(|recorder| {
            if let Some(recorder) = recorder
                && recorder.tainted.is_none()
            {
                recorder.tainted = Some(why);
            }
        });
    }
}

/// Watches an evaluation that no plan covers, to make one.
struct Recorder {
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
    leaves: usize,
    tainted: Option<&'static str>,
}

impl Recorder {
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
pub(super) fn begin_store(len: usize, fused: &Fused) -> Option<usize> {
    RECORDER.with_borrow_mut(|recorder| {
        let recorder = recorder.as_mut()?;
        if recorder.tainted.is_some() {
            return None;
        }
        match recorder.step(len, fused) {
            Ok(step) => {
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

/// Called by `run_store` when it returns while recording: what the step wrote.
pub(super) fn end_store<E>(step: Option<usize>, outputs: &Result<Vec<Storage>, E>) {
    let Some(step) = step else { return };
    RECORDER.with_borrow_mut(|recorder| {
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

/// Called when `node` got `storage` as its result while recording.
pub(super) fn finished(node: &Node, storage: &Storage) {
    if !recording() {
        return;
    }
    RECORDER.with_borrow_mut(|recorder| {
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

/// Evaluates `pending` (the nodes `eval` asked for that are still pending, all of this thread's)
/// with `run`, which plans and launches, or from the plan an earlier evaluation of the same
/// structure left.
pub(super) fn evaluate(jit: &JitState, pending: Vec<Arc<Node>>, run: impl FnOnce(Vec<Arc<Node>>)) {
    let Some(first) = pending.first() else { return run(pending) };
    let ctx = first.context().clone();
    if !usable(jit, &ctx) {
        return run(pending);
    }
    {
        let mut table = lock(&jit.plans);
        if table.skip > 0 {
            table.skip -= 1;
            drop(table);
            return run(pending);
        }
    }
    let mut walk = SCRATCH.with(|scratch| std::mem::take(&mut *scratch.borrow_mut()));
    let found = walk.run(jit, &ctx, &pending).map(|key| (key, lock(&jit.plans).find(key)));
    match found {
        // Not a graph the cache can handle: nothing is remembered.
        None => {
            note_miss(jit);
            walk.clear();
            run(pending);
        }
        Some((_, Some(Some(plan)))) if plan.nodes == walk.nodes.len() && plan.leaves == walk.leaves.len() && plan.classes == walk.classes.len() => {
            lock(&jit.plans).misses = 0;
            if replay(&ctx, &plan, &walk).is_ok() {
                jit.plan_stats.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                // Whatever the plan finished is finished; the rest is evaluated as ever.
                walk.clear();
                run(pending.into_iter().filter(|node| node.is_pending()).collect());
            }
        }
        // A structure that cannot be replayed (or a key that matched a plan of another shape).
        Some((_, Some(_))) => {
            note_miss(jit);
            walk.clear();
            run(pending);
        }
        Some((key, None)) => {
            note_miss(jit);
            let recorder = Recorder::new(&walk);
            walk.clear();
            RECORDER.with_borrow_mut(|slot| *slot = Some(recorder));
            ACTIVE.with(|active| active.set(true));
            run(pending);
            ACTIVE.with(|active| active.set(false));
            if let Some(recorder) = RECORDER.with_borrow_mut(Option::take) {
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
        table.skip = (1u32 << (table.misses - MISSES_BEFORE_SKIPPING).min(6)).min(MAX_SKIP);
    }
}

/// Runs `plan` over the leaves and scalars of `walk`.
fn replay(ctx: &Context, plan: &Plan, walk: &Walk) -> Result<(), ()> {
    let class_words: Vec<u32> = walk.classes.iter().map(|&(_, bits)| bits as u32).collect();
    let mut outs: Vec<Vec<Arc<Storage>>> = Vec::with_capacity(plan.steps.len());
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
            walk.nodes[at as usize].fused.store(true, Ordering::Relaxed);
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
