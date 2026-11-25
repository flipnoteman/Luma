//! Trace following (W1-U1b): a loop that builds the same graph again and again (`x = step(&x);
//! x.eval()`) stops paying for it.
//!
//! A *step* is what a thread does between two evaluations. The first step of a kind is
//! *recorded*: every `create` call leaves an [`Entry`] (the operation, the operands as
//! descriptors, what the slow path decided: the rewrites, the cut, hash-consing, the numbers
//! `measure` made) and every launch the step makes (the cuts inside `create`, and the evaluation
//! that ends it) is recorded too, as a [`Plan`] over the tape's own numbering. The next step with
//! the same calls on the same kinds of operands *follows* the tape:
//!
//! - a call that matches the entry at the cursor makes a **phantom**: a [`Node`] (an `Array`
//!   holds an `Arc<Node>`, so the handle is real and observable) that was made without `measure`,
//!   `settle`, the hash-consing table, the readers of its operands, the live list and its ids
//!   (see [`super::SPECIAL_LAZY`], [`super::SPECIAL_UNREGISTERED`]). The pending count is kept: a
//!   phantom is pending like any node;
//! - a cut that the entry recorded is replayed from the plan before the node is made;
//! - the evaluation that ends the step replays the rest of the plan over the step's new buffers
//!   and scalars, without a walk of the graph, and finishes the nodes the plan finishes;
//! - anything else *diverges*: the phantoms are registered (in the order they were made, so the
//!   lists are what they would have been) and the rest of the step runs as ever. Nothing a
//!   phantom does not yet need to be told is lost: whatever reads the ids makes them first
//!   ([`make_ids`]), a write, an evaluation that is not the recorded one, a read of an
//!   intermediate, a reduction, an assignment all register the phantoms first.
//!
//! The tape holds descriptors and the plan, never nodes or buffers (see the keccak entry of the
//! changelog); the step being followed holds weak references.
//!
//! One thread follows per context: a write by another thread would have to flush the phantoms,
//! which are on no list ([`Shared::shared`]).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::array::Storage;
use crate::context::Context;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu_type::ElemLayout;
use crate::profile::{Decision, DecisionSite, SplitReason};

use super::plancache::{self, Plan, Rebind};
use super::{cons_hash, convert_evaluated, new_node, rerun, simplify, Args, ElemTy, Ids, JitState, Node, Op, OpKind, Src, UnOp, FUSION_LIMIT, registered_of, Registered, SPECIAL_GUARDED, SPECIAL_LAZY, SPECIAL_UNREGISTERED};

/// Whether trace following is on when `LUMA_TRACE_FOLLOW` is `value` (`on`/`off`, with `+verify`
/// to check every replay against a walk of the graph).
const DEFAULT_ON: bool = true;

const ENV: &str = "LUMA_TRACE_FOLLOW";

fn env_words() -> Vec<String> {
    std::env::var(ENV).unwrap_or_default().to_ascii_lowercase().split('+').map(|word| word.trim().to_string()).collect()
}

/// Whether the environment switches trace following on or off.
fn default_on() -> bool {
    let mut on = DEFAULT_ON;
    for word in env_words() {
        match word.as_str() {
            "on" | "1" | "true" | "yes" => on = true,
            "off" | "0" | "false" | "no" => on = false,
            _ => {}
        }
    }
    on
}

/// Whether `LUMA_TRACE_FOLLOW` has `+verify`: every replay of a tape is checked against the walk
/// of the plan cache (a test of the tape, not a cheap thing to leave on).
fn verifying() -> bool {
    static VERIFY: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| env_words().iter().any(|word| word == "verify"));
    *VERIFY
}

/// Whether `LUMA_TRACE_FOLLOW` has `+trace`: every end of a step that left no tape, and every
/// divergence, is printed with its reason.
fn tracing() -> bool {
    static TRACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| env_words().iter().any(|word| word == "trace"));
    *TRACE
}

fn note(what: &str) {
    if tracing() {
        eprintln!("luma trace following: {what}");
    }
}

/// What trace following did (tests).
#[derive(Debug, Default)]
pub(super) struct Stats {
    recorded: AtomicU64,
    replayed: AtomicU64,
    diverged: AtomicU64,
    abandoned: AtomicU64,
    cuts: AtomicU64,
}

/// A snapshot of [`Stats`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StatsSnapshot {
    /// Tapes made.
    pub(crate) recorded: u64,
    /// Steps whose evaluation replayed a tape.
    pub(crate) replayed: u64,
    /// Steps that followed a tape up to a call that did not match, and went on as ever.
    pub(crate) diverged: u64,
    /// Steps that were recorded and left no tape.
    pub(crate) abandoned: u64,
    /// Cuts replayed from a tape.
    pub(crate) cuts: u64,
}

/// What a context knows of trace following.
#[derive(Debug)]
pub(super) struct Shared {
    on: AtomicBool,
    /// The thread that records and follows (its ordinal, 0 for none yet).
    owner: AtomicU64,
    /// Set once another thread followed or wrote: phantoms are on no list, so a write of another
    /// thread would not flush them. Following stops for the context.
    shared: AtomicBool,
    stats: Arc<Stats>,
}

impl Shared {
    /// Prints the counters (`LUMA_TRACE_FOLLOW=+stats`, one line per call).
    pub(super) fn report(&self) {
        if env_words().iter().any(|word| word == "stats") {
            let s = &self.stats;
            eprintln!(
                "luma trace following: stats recorded {} replayed {} diverged {} abandoned {} cuts {}",
                s.recorded.load(Ordering::Relaxed),
                s.replayed.load(Ordering::Relaxed),
                s.diverged.load(Ordering::Relaxed),
                s.abandoned.load(Ordering::Relaxed),
                s.cuts.load(Ordering::Relaxed)
            );
        }
    }
}

impl Shared {
    pub(super) fn new() -> Shared {
        Shared { on: AtomicBool::new(default_on()), owner: AtomicU64::new(0), shared: AtomicBool::new(false), stats: Arc::new(Stats::default()) }
    }

    #[inline(always)]
    pub(super) fn on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn set(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
        self.owner.store(0, Ordering::Relaxed);
        self.shared.store(false, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            recorded: self.stats.recorded.load(Ordering::Relaxed),
            replayed: self.stats.replayed.load(Ordering::Relaxed),
            diverged: self.stats.diverged.load(Ordering::Relaxed),
            abandoned: self.stats.abandoned.load(Ordering::Relaxed),
            cuts: self.stats.cuts.load(Ordering::Relaxed),
        }
    }

    /// A use of the context by a thread other than the one that follows ends following.
    fn foreign_use(&self) {
        let owner = self.owner.load(Ordering::Relaxed);
        if owner != 0 && owner != thread_ordinal() {
            self.shared.store(true, Ordering::Relaxed);
        }
    }

    /// Lets the calling thread follow on this context, unless another one does.
    fn claim(&self) -> bool {
        let me = thread_ordinal();
        match self.owner.compare_exchange(0, me, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => true,
            Err(owner) if owner == me => !self.shared.load(Ordering::Relaxed),
            Err(_) => {
                self.shared.store(true, Ordering::Relaxed);
                false
            }
        }
    }
}

fn thread_ordinal() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static ORDINAL: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ORDINAL.with(|ordinal| *ordinal)
}

/// Called when `jit` is written to: a write by a thread that does not follow ends following.
pub(super) fn note_write(jit: &JitState) {
    let owner = jit.follow.owner.load(Ordering::Relaxed);
    if owner != 0 && owner != thread_ordinal() {
        jit.follow.shared.store(true, Ordering::Relaxed);
    }
}

// -- Serializing the writes of the ids of nodes that were made without them ----------------------

/// Serializes the writes of [`make_ids`] (the cells are read without a lock, see [`super::IdsCell`]).
static IDS_LOCK: Mutex<()> = Mutex::new(());

/// Makes the ids of `root` and of every node below it that was made without them (see
/// [`super::SPECIAL_LAZY`]), operands first and without recursion.
#[cold]
#[inline(never)]
pub(super) fn make_ids(root: &Node) {
    let lazy = |node: &Node| node.special.load(Ordering::Acquire) & SPECIAL_LAZY != 0;
    fn operands(node: &Node) -> Vec<Arc<Node>> {
        let arity = node.op.kind.arity();
        match lock(&node.args).as_ref() {
            Some(args) => args.iter().take(arity).filter_map(|arg| if let Src::Node(operand) = arg { Some(Arc::clone(operand)) } else { None }).collect(),
            None => Vec::new(),
        }
    }
    // Post-order over the operands that still need theirs (the expression is a DAG).
    let mut order: Vec<Arc<Node>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack: Vec<(Arc<Node>, bool)> = operands(root).into_iter().filter(|operand| lazy(operand)).map(|operand| (operand, false)).collect();
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            order.push(node);
            continue;
        }
        if !lazy(&node) || !seen.insert(Arc::as_ptr(&node) as usize) {
            continue;
        }
        let below = operands(&node);
        stack.push((node, true));
        stack.extend(below.into_iter().filter(|operand| lazy(operand)).map(|operand| (operand, false)));
    }
    for node in &order {
        fill(node);
    }
    fill(root);
}

/// Makes the ids of `node` from those of its operands (which are made).
fn fill(node: &Node) {
    if node.special.load(Ordering::Acquire) & SPECIAL_LAZY == 0 {
        return;
    }
    let args = lock(&node.args).clone();
    let ids = match args {
        Some(args) => Ids::from(super::measure(args.get(..node.op.kind.arity()).unwrap_or(&[]))),
        // Evaluated meanwhile: nothing reads what is below a node that is not pending.
        None => no_ids(),
    };
    let _guard = IDS_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if node.special.load(Ordering::Acquire) & SPECIAL_LAZY != 0 {
        // SAFETY: the flag is set and the lock held: nobody reads the cell, nobody else writes it.
        unsafe { *node.ids.0.get() = ids };
        node.special.fetch_and(!SPECIAL_LAZY, Ordering::Release);
    }
}

fn no_ids() -> Ids {
    Ids { ids: Box::new([]), base: 0, words: 0, sparse: true }
}

// -- Registering phantoms ---------------------------------------------------------------------------

/// Registers a phantom as `create` would have: with the readers of its operands (a buffer's
/// readers, a pending node's readers and consumer count; the buffer of an operand that was
/// evaluated meanwhile), the live list and the hash-consing table. Does nothing for a node that is
/// registered, evaluated, or has no operands left.
pub(super) fn register(node: &Arc<Node>) {
    if node.special.fetch_and(!SPECIAL_UNREGISTERED, Ordering::AcqRel) & SPECIAL_UNREGISTERED == 0 {
        return;
    }
    if node.result.get().is_some() {
        return;
    }
    let Some(args) = node.operands() else { return };
    let jit = node.ctx.jit();
    let arity = node.op.kind.arity();
    if node.thread != super::current_thread() {
        jit.follow.shared.store(true, Ordering::Relaxed);
    }
    let weak = Arc::downgrade(node);
    let guarded = node.special.load(Ordering::Acquire) & SPECIAL_GUARDED != 0;
    for (i, arg) in args.iter().enumerate().take(arity) {
        let repeated = |other: &Src| match (arg, other) {
            (Src::Buffer(a), Src::Buffer(b)) => Arc::ptr_eq(a, b),
            (Src::Node(a), Src::Node(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if args.iter().take(i).any(repeated) {
            continue;
        }
        match arg {
            Src::Buffer(storage) => {
                if !guarded {
                    storage.readers.add_weak(weak.clone());
                }
            }
            Src::Node(operand) => match operand.result.get() {
                // Its readers went to its buffer when it was evaluated.
                Some(Ok(storage)) => storage.readers.add_weak(weak.clone()),
                Some(Err(_)) => {}
                None => {
                    operand.consumers.fetch_add(1, Ordering::Relaxed);
                    operand.readers.add_weak(weak.clone());
                }
            },
            Src::Scalar(_) => {}
        }
    }
    jit.with_live(|live| live.add_weak(weak.clone()));
    if !matches!(node.op.kind, OpKind::Unary(UnOp::Id)) {
        jit.cons_insert(cons_hash(&node.op, node.len, &args), weak);
    }
}

// -- The tape ---------------------------------------------------------------------------------------

const NONE: u32 = u32::MAX;

/// Longest step a tape covers.
const MAX_ENTRIES: usize = 1 << 14;

/// What an operand of a recorded call is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Desc {
    None,
    /// The node made by entry `j` of this step.
    Node(u32),
    /// Leaf buffer `k` (in order of first use).
    Leaf(u32),
    /// Scalar class `c` (in order of first use), of this type.
    Scalar(u32, ElemTy),
    /// Output `out` of launch `step` of this step (what a cut made of a node's value).
    Out(u32, u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// A node was made from the operands.
    Create,
    /// The call returned the node of entry `j` (hash-consing).
    Hit(u32),
}

/// The final form of a call that rewrites or a cut changed.
struct Final {
    op: Op,
    arity: u8,
    desc: [Desc; 3],
}

/// What a cut inside `create` did: the launches it made (positions in the plan), why it cut, and
/// how many references each node of the step held after it (see [`strong_counts`]).
struct Cut {
    steps: std::ops::Range<u32>,
    reason: Option<SplitReason>,
    strong: Vec<u32>,
}

struct Entry {
    op: Op,
    len: usize,
    arity: u8,
    raw: [Desc; 3],
    kind: Kind,
    fin: Option<Box<Final>>,
    /// Rewrites that fired.
    rounds: u8,
    /// Whether the rewrites must be tried again (some fired, or a scalar operand is one that rules
    /// look for, see [`simplify::sensitive`]).
    rewrite: bool,
    /// An earlier entry with the same final operation and operands: the node that hash-consing
    /// would hand out if it were unheld, pending and not fused.
    cand: u32,
    below: usize,
    depth: u16,
    shift_depth: u8,
    weight: f64,
    cut: Option<Box<Cut>>,
}

struct LeafSig {
    len: usize,
    layout: ElemLayout,
}

struct ClassSig {
    ty: ElemTy,
    /// Whether the planner treats the value as one it makes up (see `Walk::seal`).
    special: bool,
    /// See [`simplify::sensitive`].
    sensitive: bool,
}

/// How `step_boundary` ends when it finds no loop.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Exit {
    /// `tracker.remember`: the array of the step before is gone and nothing reads its buffer.
    Remember,
    /// `tracker.reset`: no loop could be traced.
    Reset,
}

/// For each array of the step before: whether it is still alive, and the leaf of the step that
/// holds its value (the state it reads), if any.
type Pattern = Vec<(bool, Option<u32>)>;

struct Tape {
    jit: u64,
    stamp: u64,
    pending_start: i64,
    pending_eval: i64,
    entries: Vec<Entry>,
    leaves: Vec<LeafSig>,
    classes: Vec<ClassSig>,
    /// The lengths of the nodes, sorted.
    lens: Vec<u64>,
    /// The roots the evaluation computed, as entries, in the order `eval` asked for them.
    roots: Vec<u32>,
    plan: Plan,
    /// The launches that cuts made are replayed with the calls; those from here on with the evaluation.
    eval_from: usize,
    /// How many references each node of the step held when the evaluation began.
    strong: Vec<u32>,
    /// The key of the walk of the plan cache at that point (for `+verify`).
    key: Option<[u64; 2]>,
    /// The leaves a launch of the plan could take over the buffer of, and the entries whose nodes
    /// hold the leaf as an operand (with how many times): see [`takes_over`].
    watch: Vec<(u32, Vec<(u32, u32)>)>,
    boundary: RefCell<Option<(Exit, Pattern)>>,
}

// -- State ------------------------------------------------------------------------------------------

/// A step being recorded.
struct Rec {
    jit: u64,
    /// The kind of step this is (see [`Kind`]).
    kind: u64,
    stamp: u64,
    pending_start: i64,
    write_epoch: u64,
    poisoned: Option<&'static str>,
    entries: Vec<Entry>,
    leaves: Vec<Weak<Storage>>,
    leaf_sigs: Vec<LeafSig>,
    classes: Vec<(ElemTy, u64)>,
    class_sigs: Vec<ClassSig>,
    class_at: FxMap<(ElemTy, u64), u32>,
    cons: FxMap<(u64, usize, [Desc; 3]), u32>,
    boundary: Option<(Exit, Pattern)>,
    /// The call under way, if it is recorded.
    call: Option<Call>,
}

/// A call being recorded: what it asked for, and where the plan stood when it began.
struct Call {
    op: Op,
    len: usize,
    raw: [Desc; 3],
    steps_at_start: usize,
    reason: Option<SplitReason>,
}

impl Drop for Rec {
    fn drop(&mut self) {
        plancache::tape_abort();
    }
}

/// A step being followed.
struct Fol {
    /// The kind of step this is (see [`Kind`]).
    kind: u64,
    tape: Rc<Tape>,
    pos: usize,
    leaves: Vec<Weak<Storage>>,
    classes: Vec<(ElemTy, u64)>,
    class_set: Option<FxMap<(ElemTy, u64), ()>>,
    /// The node of each entry (dangling for a call that made none).
    nodes: Vec<Weak<Node>>,
    /// The outputs of the launches replayed so far.
    outs: Vec<Vec<Arc<Storage>>>,
    write_epoch: u64,
    fast: bool,
    /// Entries below this are registered.
    registered: usize,
}

impl Drop for Fol {
    fn drop(&mut self) {
        self.register_all();
    }
}

enum Phase {
    /// The next call starts a step.
    Idle,
    /// This step is neither recorded nor followed.
    Skip,
    Rec(Box<Rec>),
    Fol(Box<Fol>),
}

/// What the follower knows of one kind of step: the steps that begin with the same call. A loop
/// of several steps (an `eval` inside what the program calls a step, a `cache`, a read) is
/// several kinds, each with a tape of its own and a backoff of its own, so a kind that cannot be
/// followed costs the others nothing.
struct StepKind {
    key: u64,
    /// The tape of the last step of this kind that was recorded to its end.
    tape: Option<Rc<Tape>>,
    /// Steps of this kind that were recorded or followed in vain, in a row.
    fails: u32,
    /// Steps of this kind still to leave alone.
    cooldown: u32,
    used: u64,
}

/// Kinds of steps a thread keeps (the one longest unused goes first). A kind without a tape is
/// a few words: what it remembers is that it did not work.
const MAX_KINDS: usize = 64;

/// Tapes a thread keeps (the one longest unused goes first).
const MAX_TAPES: usize = 8;

struct State {
    jit: u64,
    /// The counters of that context.
    stats: Option<Arc<Stats>>,
    kinds: Vec<StepKind>,
    clock: u64,
    phase: Phase,
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State { jit: 0, stats: None, kinds: Vec::new(), clock: 0, phase: Phase::Idle }) };
    /// Whether the calls of this thread need to be looked at: false while the step is neither
    /// recorded nor followed, which is all a call costs then.
    static GATE: Cell<bool> = const { Cell::new(true) };
}

/// Whether `create` has to ask [`begin`] about the call.
#[inline(always)]
pub(super) fn gate() -> bool {
    GATE.with(Cell::get)
}

/// What kind of step begins with this call: the operation, the length and what the operands are.
fn step_key(op: &Op, len: usize, args: &Args, arity: usize) -> u64 {
    let mut key = op.fingerprint ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for arg in args.iter().take(arity) {
        key = key.rotate_left(5)
            ^ match arg {
                Src::Node(_) => 1,
                Src::Buffer(_) => 2,
                Src::Scalar(_) => 3,
            };
    }
    key
}

impl State {
    fn set(&mut self, phase: Phase) {
        GATE.with(|gate| gate.set(!matches!(phase, Phase::Skip)));
        self.phase = phase;
    }

    /// The record of kind `key` (made if there is none).
    fn kind(&mut self, key: u64) -> &mut StepKind {
        self.clock += 1;
        let clock = self.clock;
        let at = match self.kinds.iter().position(|kind| kind.key == key) {
            Some(at) => at,
            None => {
                if self.kinds.len() >= MAX_KINDS {
                    let oldest = (0..self.kinds.len()).min_by_key(|&at| self.kinds[at].used).unwrap_or(0);
                    self.kinds.swap_remove(oldest);
                }
                self.kinds.push(StepKind { key, tape: None, fails: 0, cooldown: 0, used: 0 });
                self.kinds.len() - 1
            }
        };
        let kind = &mut self.kinds[at];
        kind.used = clock;
        kind
    }

    /// Makes `tape` the tape of kind `key`, dropping the one longest unused if there are too many.
    fn keep(&mut self, key: u64, tape: Rc<Tape>) {
        if self.kinds.iter().filter(|kind| kind.tape.is_some() && kind.key != key).count() >= MAX_TAPES
            && let Some(oldest) = self.kinds.iter_mut().filter(|kind| kind.tape.is_some() && kind.key != key).min_by_key(|kind| kind.used)
        {
            oldest.tape = None;
        }
        self.kind(key).tape = Some(tape);
    }

    /// A step of kind `key` was recorded or followed in vain: the next ones of its kind are left
    /// alone for a while if this keeps happening (a step that cannot be followed then costs a
    /// recording one time in 65). A tape that did not hold goes unless `keep_tape`.
    fn failed(&mut self, key: u64, keep_tape: bool) {
        let kind = self.kind(key);
        if !keep_tape {
            kind.tape = None;
        }
        kind.fails = (kind.fails + 1).min(7);
        kind.cooldown = if kind.fails >= 2 { (1u32 << (kind.fails - 1)).min(64) } else { 0 };
    }

    /// A step of kind `key` was replayed: only that earns the kind its trust back (a tape that is
    /// made and never followed does not).
    fn replayed(&mut self, key: u64) {
        let kind = self.kind(key);
        kind.fails = 0;
        kind.cooldown = 0;
    }
}

// -- Calls ------------------------------------------------------------------------------------------

/// What `begin` hands back to `create`.
pub(super) enum Begun {
    /// The call is answered with this node.
    Done(Arc<Node>),
    /// The call goes on as ever, with the operands as they are now (see [`begin`]).
    Go(Go),
}

/// What `create` tells [`finish_created`] and [`finish_hit`].
#[derive(Clone, Copy)]
pub(super) enum Go {
    Off,
    /// Neither recorded nor followed (any more).
    Skip,
    /// Being recorded.
    Record,
}

/// Whether the usable node `node` is one a hash-consing lookup would hand out.
fn consable(node: &Node) -> bool {
    node.is_pending() && node.handles.load(Ordering::Relaxed) == 0 && !node.fused.load(Ordering::Relaxed) && node.looped.get().is_none()
}

/// Called by `create` for every call while the thread is not skipping (see [`gate`]).
///
/// `args` are the operands of the call; they are handed back (as they are after the rewrites a
/// followed call tried, and the cut it replayed) unless the call is answered with a node.
#[inline(never)]
pub(super) fn begin(ctx: &Context, len: usize, op: &Op, args: &mut Args, arity: usize, name: &'static str, location: &'static std::panic::Location<'static>) -> Begun {
    let entered = STATE.with_borrow_mut(|state| state.enter(ctx, len, op, args, arity, name, location));
    match entered {
        Entered::Done(begun) => begun,
        Entered::Cut(job) => finish_cut(ctx, *job, args, name, location),
    }
}

/// The operands left behind by a call that was answered with a node.
fn taken(args: &mut Args) -> Args {
    std::mem::replace(args, [Src::unused(), Src::unused(), Src::unused()])
}

/// A call whose entry holds a cut: the launches are made outside the state.
struct CutJob {
    fol: Box<Fol>,
    op: Op,
    len: usize,
    arity: usize,
    rewritten: Option<(Op, usize)>,
    /// The operands as the call came, if rewrites changed them.
    saved: Option<Args>,
}

enum Entered {
    Done(Begun),
    Cut(Box<CutJob>),
}

impl State {
    #[allow(clippy::too_many_arguments)]
    fn enter(&mut self, ctx: &Context, len: usize, op: &Op, args: &mut Args, arity: usize, name: &'static str, location: &'static std::panic::Location<'static>) -> Entered {
        let jit = ctx.jit();
        if self.jit != jit.id {
            self.set(Phase::Idle);
            self.kinds.clear();
            self.jit = jit.id;
            self.stats = Some(Arc::clone(&jit.follow.stats));
        }
        if jit.follow.shared.load(Ordering::Relaxed) || rerun::active() || FUSION_LIMIT.with(Cell::get).is_some() {
            self.stop("following is not possible now");
            return Entered::Done(Begun::Go(Go::Skip));
        }
        match &mut self.phase {
            Phase::Skip => return Entered::Done(Begun::Go(Go::Skip)),
            Phase::Rec(rec) => {
                let begun = rec.begin(len, op, args, arity);
                if let Some(why) = rec.poisoned {
                    self.stop(why);
                }
                return Entered::Done(begun);
            }
            Phase::Fol(_) => return self.follow(ctx, len, op, args, arity, name, location),
            Phase::Idle => {}
        }
        // A step begins.
        if !jit.follow.claim() {
            self.set(Phase::Skip);
            return Entered::Done(Begun::Go(Go::Skip));
        }
        let key = step_key(op, len, args, arity);
        let kind = self.kind(key);
        if kind.cooldown > 0 {
            kind.cooldown -= 1;
            self.set(Phase::Skip);
            return Entered::Done(Begun::Go(Go::Skip));
        }
        let stamp = plancache::stamp(jit, ctx);
        let pending = jit.pending_nodes.load(Ordering::Relaxed);
        if let Some(tape) = kind.tape.as_ref().filter(|tape| tape.jit == jit.id && tape.stamp == stamp && tape.pending_start == pending) {
            let tape = Rc::clone(tape);
            let fol = Fol {
                kind: key,
                pos: 0,
                leaves: Vec::with_capacity(tape.leaves.len()),
                classes: Vec::with_capacity(tape.classes.len()),
                class_set: None,
                nodes: Vec::with_capacity(tape.entries.len()),
                outs: Vec::with_capacity(tape.plan.steps.len()),
                write_epoch: jit.write_epoch.load(Ordering::Relaxed),
                fast: ctx.fast_math(),
                registered: 0,
                tape,
            };
            self.set(Phase::Fol(Box::new(fol)));
            return self.follow(ctx, len, op, args, arity, name, location);
        }
        // The tape of the step before stays until a step makes another.
        plancache::tape_begin();
        let rec = Rec {
            jit: jit.id,
            kind: key,
            stamp,
            pending_start: pending,
            write_epoch: jit.write_epoch.load(Ordering::Relaxed),
            poisoned: None,
            entries: Vec::new(),
            leaves: Vec::new(),
            leaf_sigs: Vec::new(),
            classes: Vec::new(),
            class_sigs: Vec::new(),
            class_at: FxMap::default(),
            cons: FxMap::default(),
            boundary: None,
            call: None,
        };
        let mut rec = Box::new(rec);
        let begun = rec.begin(len, op, args, arity);
        let poisoned = rec.poisoned;
        self.set(Phase::Rec(rec));
        if let Some(why) = poisoned {
            self.stop(why);
        }
        Entered::Done(begun)
    }

    /// Stops recording or following (the step goes on as ever) and leaves the next steps alone
    /// for a while.
    fn stop(&mut self, why: &'static str) {
        if !matches!(self.phase, Phase::Skip | Phase::Idle) {
            note(why);
        }
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Fol(mut fol) => {
                fol.register_all();
                if let Some(stats) = &self.stats {
                    stats.diverged.fetch_add(1, Ordering::Relaxed);
                }
                self.failed(fol.kind, false);
            }
            Phase::Rec(rec) => {
                if let Some(stats) = &self.stats {
                    stats.abandoned.fetch_add(1, Ordering::Relaxed);
                }
                self.failed(rec.kind, true);
            }
            _ => {}
        }
        self.set(Phase::Skip);
    }

    /// Whether `d` describes the operand `src`.
    #[allow(clippy::too_many_arguments)]
    fn follow(&mut self, ctx: &Context, len: usize, op: &Op, args: &mut Args, arity: usize, name: &'static str, location: &'static std::panic::Location<'static>) -> Entered {
        let jit = ctx.jit();
        let Phase::Fol(fol) = &mut self.phase else { unreachable!("follow is called while following") };
        let tape = Rc::clone(&fol.tape);
        let verdict = (|| {
            let Some(e) = tape.entries.get(fol.pos) else { return Err("the step is longer than the tape") };
            if e.len != len || e.arity as usize != arity || e.op != *op {
                if tracing() {
                    note(&format!("entry {} of {}: len {} vs {len}, arity {} vs {arity}, op {:?} vs {:?}", fol.pos, tape.entries.len(), e.len, e.arity, e.op.kind, op.kind));
                }
                return Err("a call that is not the next entry");
            }
            for i in 0..arity {
                if !fol.operand(e.raw[i], &args[i]) {
                    return Err("an operand that is not the entry's");
                }
            }
            if fol.write_epoch != jit.write_epoch.load(Ordering::Relaxed) {
                return Err("a write");
            }
            Ok(e)
        })();
        let e = match verdict {
            Ok(e) => e,
            Err(why) => {
                self.stop(why);
                return Entered::Done(Begun::Go(Go::Skip));
            }
        };
        // Rewrites depend on the values of some scalars: try them again where they could differ.
        let mut rewritten: Option<(Op, usize)> = None;
        let mut saved: Option<Args> = None;
        if jit.simplify_enabled() && (e.rewrite || fol.fast) {
            saved = Some(args.clone());
            let mut rounds = 0u8;
            let mut now: Option<Op> = None;
            while rounds < 4
                && let Some((next_op, operands)) = simplify::rewrite(now.as_ref().unwrap_or(op), args, fol.fast)
            {
                now = Some(next_op);
                *args = operands;
                rounds += 1;
            }
            if rounds != e.rounds {
                if tracing() {
                    note(&format!("entry {}: rewrites {rounds} vs {} recorded (fast {}, op {:?})", fol.pos, e.rounds, fol.fast, op.kind));
                }
                *args = saved.take().expect("saved above");
                self.stop("a rewrite that is not the recorded one");
                return Entered::Done(Begun::Go(Go::Skip));
            }
            if let Some(now) = now {
                let farity = now.kind.arity();
                rewritten = Some((now, farity));
                if convert_evaluated(args, farity).is_err() {
                    *args = saved.take().expect("saved above");
                    self.stop("an operand failed");
                    return Entered::Done(Begun::Go(Go::Skip));
                }
            } else {
                saved = None;
            }
        } else if e.rounds != 0 {
            self.stop("a rewrite that is not the recorded one");
            return Entered::Done(Begun::Go(Go::Skip));
        }
        let (fop, farity) = rewritten.unwrap_or((*op, arity));
        if e.cut.is_some() {
            // The launches of the cut are made without the state borrowed (launching can create nodes).
            let Phase::Fol(fol) = std::mem::replace(&mut self.phase, Phase::Skip) else { unreachable!() };
            GATE.with(|gate| gate.set(false));
            return Entered::Cut(Box::new(CutJob { fol, op: *op, len, arity, rewritten: Some((fop, farity)), saved }));
        }
        self.make(ctx, len, fop, args, saved, farity, name, location)
    }

    /// Makes the phantom of the entry at the cursor (its operands are the final ones: `args`).
    #[allow(clippy::too_many_arguments)]
    fn make(&mut self, ctx: &Context, len: usize, fop: Op, args: &mut Args, saved: Option<Args>, farity: usize, name: &'static str, location: &'static std::panic::Location<'static>) -> Entered {
        let jit = ctx.jit();
        let Phase::Fol(fol) = &mut self.phase else { unreachable!("make is called while following") };
        let tape = Rc::clone(&fol.tape);
        let e = &tape.entries[fol.pos];
        let wanted = e.fin.as_deref().map_or((&e.op, e.arity as usize, &e.raw), |fin| (&fin.op, fin.arity as usize, &fin.desc));
        let mut matches = fop == *wanted.0 && farity == wanted.1;
        if matches && e.fin.is_some() {
            for i in 0..farity {
                matches &= fol.operand(wanted.2[i], &args[i]);
            }
        }
        if !matches {
            if let Some(saved) = saved {
                *args = saved;
            }
            self.stop("a call whose final form is not the recorded one");
            return Entered::Done(Begun::Go(Go::Skip));
        }
        match e.kind {
            Kind::Hit(j) => {
                let found = fol.nodes.get(j as usize).and_then(Weak::upgrade).filter(|node| consable(node));
                match found {
                    Some(node) => {
                        fol.nodes.push(Weak::new());
                        fol.pos += 1;
                        Entered::Done(Begun::Done(node))
                    }
                    None => {
                        if let Some(saved) = saved {
                            *args = saved;
                        }
                        self.stop("a hash-consing hit that is not one now");
                        Entered::Done(Begun::Go(Go::Skip))
                    }
                }
            }
            Kind::Create => {
                if e.cand != NONE && fol.nodes.get(e.cand as usize).and_then(Weak::upgrade).is_some_and(|node| consable(&node)) {
                    if let Some(saved) = saved {
                        *args = saved;
                    }
                    self.stop("an equal node that hash-consing would hand out now");
                    return Entered::Done(Begun::Go(Go::Skip));
                }
                let (registered, known) = registered_of(args, farity);
                let node = new_node(ctx, len, &fop, taken(args), name, location, 0, e.below, no_ids(), SPECIAL_LAZY | SPECIAL_UNREGISTERED, e.shift_depth, e.depth, e.weight);
                jit.pending_nodes.fetch_add(1, Ordering::Relaxed);
                let weak = Arc::downgrade(&node);
                // A node that reads a buffer is on its readers from the start: whoever overwrites the
                // buffer (another thread, say) flushes it, which evaluates it as any node would be.
                for entry in &registered[..known] {
                    match *entry {
                        Registered::Buffer(storage) => {
                            // SAFETY: the node's `args` hold the buffer.
                            unsafe { (*storage).readers.add_weak(weak.clone()) };
                            node.special.fetch_or(SPECIAL_GUARDED, Ordering::Release);
                        }
                        // An operand an array holds can be reached by another thread, which may overwrite
                        // its value once it has one: the readers it hands to its buffer must be there.
                        Registered::Node(operand) => {
                            // SAFETY: the node's `args` hold the operand.
                            unsafe {
                                if (*operand).handles.load(Ordering::Relaxed) > 0 {
                                    (*operand).readers.add_weak(weak.clone());
                                }
                            }
                        }
                        Registered::None => {}
                    }
                }
                fol.nodes.push(weak);
                fol.pos += 1;
                if let Some(cut) = &e.cut {
                    // The references the nodes of the step hold are the recorded ones, or the
                    // launches the plan made were for other nodes than these.
                    if fol.nodes[..fol.pos - 1].iter().map(Weak::strong_count).ne(cut.strong.iter().map(|&count| count as usize)) {
                        self.stop("a cut at a time that other nodes are alive");
                    }
                }
                Entered::Done(Begun::Done(node))
            }
        }
    }
}

/// The launches of a cut, then the rest of the call.
fn finish_cut(ctx: &Context, job: CutJob, args: &mut Args, name: &'static str, location: &'static std::panic::Location<'static>) -> Begun {
    let jit = ctx.jit();
    let CutJob { mut fol, op, len, arity, rewritten, saved } = job;
    let tape = Rc::clone(&fol.tape);
    let e = &tape.entries[fol.pos];
    let cut = e.cut.as_deref().expect("a job is made for a cut");
    let replayed = if cut.steps.start as usize == fol.next_step() && !takes_over(&fol) {
        run_steps(ctx, &mut fol, cut.steps.end as usize)
    } else {
        Err(())
    };
    let ok = replayed.is_ok() && {
        let farity = rewritten.map_or(arity, |(_, farity)| farity);
        convert_evaluated(args, farity).is_ok()
    };
    if ok {
        jit.follow.stats.cuts.fetch_add(1, Ordering::Relaxed);
        if let Some(reason) = cut.reason {
            jit.note_cut(name, location, reason);
        }
        // `create` does this after a cut: without it a long chain built without `eval` keeps the GPU idle.
        let _ = ctx.submit();
    }
    let outcome = STATE.with_borrow_mut(|state| {
        if state.jit != jit.id || !matches!(state.phase, Phase::Skip | Phase::Idle) {
            // Something else took the state over meanwhile: the step goes on as ever.
            fol.register_all();
            return Entered::Done(Begun::Go(Go::Skip));
        }
        state.set(Phase::Fol(fol));
        if !ok {
            if let Some(saved) = saved {
                *args = saved;
            }
            state.stop("the launches of a cut failed");
            return Entered::Done(Begun::Go(Go::Skip));
        }
        let (fop, farity) = rewritten.unwrap_or((op, arity));
        state.make(ctx, len, fop, args, saved, farity, name, location)
    });
    match outcome {
        Entered::Done(begun) => begun,
        Entered::Cut(_) => unreachable!("an entry with a cut makes no second cut"),
    }
}

impl Fol {
    /// Whether `src` is operand `d` of the entry being followed: the node its entry made, the
    /// next leaf or the one it was, a scalar of the class it was, the output of a launch of
    /// the plan.
    fn operand(&mut self, d: Desc, src: &Src) -> bool {
        match (d, src) {
            (Desc::Node(j), Src::Node(node)) => self.nodes.get(j as usize).is_some_and(|weak| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(node))),
            (Desc::Leaf(k), Src::Buffer(storage)) => self.leaf(k, storage),
            (Desc::Out(step, out), Src::Buffer(storage)) => self.outs.get(step as usize).and_then(|outs| outs.get(out as usize)).is_some_and(|made| Arc::ptr_eq(made, storage)),
            (Desc::Scalar(c, ty), Src::Scalar(scalar)) => self.class(c, ty, scalar),
            _ => false,
        }
    }

    fn leaf(&mut self, k: u32, storage: &Arc<Storage>) -> bool {
        let k = k as usize;
        if let Some(have) = self.leaves.get(k) {
            return std::ptr::eq(have.as_ptr(), Arc::as_ptr(storage));
        }
        if k != self.leaves.len() || self.leaves.iter().any(|have| std::ptr::eq(have.as_ptr(), Arc::as_ptr(storage))) {
            return false;
        }
        let Some(sig) = self.tape.leaves.get(k) else { return false };
        if storage.len() != sig.len || storage.layout() != sig.layout || storage.segments() != 1 || storage.is_alias() || storage.is_window() {
            return false;
        }
        self.leaves.push(Arc::downgrade(storage));
        true
    }

    fn class(&mut self, c: u32, ty: ElemTy, scalar: &super::Scalar) -> bool {
        if scalar.ty != ty {
            return false;
        }
        let c = c as usize;
        if let Some(&(have_ty, bits)) = self.classes.get(c) {
            return have_ty == ty && bits == scalar.bits;
        }
        if c != self.classes.len() || ty.slots() != 1 {
            return false;
        }
        let key = (ty, scalar.bits);
        if let Some(set) = &mut self.class_set {
            if set.insert(key, ()).is_some() {
                return false;
            }
        } else if self.classes.contains(&key) {
            return false;
        } else if self.classes.len() >= 16 {
            let mut set: FxMap<(ElemTy, u64), ()> = self.classes.iter().map(|&k| (k, ())).collect();
            set.insert(key, ());
            self.class_set = Some(set);
        }
        let sig = &self.tape.classes[c];
        if ty == ElemTy::U32 && (scalar.bits == 0 || self.tape.lens.binary_search(&scalar.bits).is_ok()) != sig.special {
            return false;
        }
        // A value that rewrites look for where the step before had none, or the other way round, is
        // another step.
        if !sig.sensitive && simplify::sensitive(ty, scalar.bits) {
            return false;
        }
        self.classes.push(key);
        true
    }

    fn next_step(&self) -> usize {
        self.outs.len()
    }

    /// Registers the phantoms that are not yet, in the order they were made.
    fn register_all(&mut self) {
        for weak in &self.nodes[self.registered..] {
            if let Some(node) = weak.upgrade() {
                register(&node);
            }
        }
        self.registered = self.nodes.len();
    }
}

/// Whether a launch of the plan would take over a buffer that nothing outside the step holds, which
/// `run_store` does when it runs a program that can (a replay holds the buffers it is given, so
/// it would not): the step is then left to `create`'s own evaluation, which does.
fn takes_over(fol: &Fol) -> bool {
    fol.tape.watch.iter().any(|(leaf, holders)| {
        let Some(weak) = fol.leaves.get(*leaf as usize) else { return false };
        let internal: u32 = holders.iter().filter(|(at, _)| fol.nodes.get(*at as usize).and_then(Weak::upgrade).is_some_and(|node| node.is_pending())).map(|(_, slots)| slots).sum();
        weak.strong_count() <= internal as usize
    })
}

/// What a replay of the plan rebinds: the buffers, scalars and nodes of the step being followed.
struct View<'a> {
    leaves: &'a [Weak<Storage>],
    classes: &'a [(ElemTy, u64)],
    nodes: &'a [Weak<Node>],
}

impl Rebind for View<'_> {
    fn leaf(&self, at: u32) -> Option<Arc<Storage>> {
        self.leaves.get(at as usize)?.upgrade()
    }

    fn word(&self, class: u32) -> u32 {
        self.classes[class as usize].1 as u32
    }

    fn node(&self, at: u32) -> Option<Arc<Node>> {
        self.nodes.get(at as usize)?.upgrade()
    }
}

/// Replays the launches of the plan up to (not including) `upto`.
fn run_steps(ctx: &Context, fol: &mut Fol, upto: usize) -> Result<(), ()> {
    let tape = Rc::clone(&fol.tape);
    let jit = ctx.jit();
    let recording = jit.recording_decisions();
    if recording {
        jit.pause_decisions(true);
    }
    let mut outcome = Ok(());
    while fol.outs.len() < upto {
        let step = &tape.plan.steps[fol.outs.len()];
        let view = View { leaves: &fol.leaves, classes: &fol.classes, nodes: &fol.nodes };
        outcome = plancache::run_step(ctx, step, &mut fol.outs, &view);
        if outcome.is_err() {
            break;
        }
    }
    if recording {
        jit.pause_decisions(false);
    }
    outcome
}

// -- Recording --------------------------------------------------------------------------------------

impl Rec {
    fn poison(&mut self, why: &'static str) {
        self.poisoned.get_or_insert(why);
    }

    fn describe(&mut self, args: &Args, arity: usize) -> Option<[Desc; 3]> {
        let mut out = [Desc::None; 3];
        for i in 0..arity {
            out[i] = match &args[i] {
                Src::Node(node) => {
                    if node.looped.get().is_some() || node.deferred.get().is_some() {
                        self.poison("a loop leaf or a deferred value as an operand");
                        return None;
                    }
                    match plancache::with_tape(|recorder| recorder.node_index(node.id)).flatten() {
                        Some(j) => Desc::Node(j),
                        None => {
                            note(&format!("operand node {} ({}) of entry {} is not of this step: pending {}, looped {}", node.id, node.name, self.entries.len(), node.is_pending(), node.looped.get().is_some()));
                            self.poison("an operand that no entry of this step made");
                            return None;
                        }
                    }
                }
                Src::Buffer(storage) => {
                    let (made, leaf) = plancache::with_tape(|recorder| (recorder.produced_by(storage.id()), recorder.leaf_index(storage.id()))).unwrap_or((None, None));
                    match (made, leaf) {
                        (Some((step, out)), _) => Desc::Out(step, out),
                        (None, Some(k)) => Desc::Leaf(k),
                        (None, None) => {
                            if storage.segments() != 1 || storage.is_alias() || storage.is_window() {
                                self.poison("a leaf a replay cannot rebind");
                                return None;
                            }
                            let k = plancache::with_tape(|recorder| recorder.note_leaf(storage.id())).unwrap_or(0);
                            self.leaves.push(Arc::downgrade(storage));
                            self.leaf_sigs.push(LeafSig { len: storage.len(), layout: storage.layout() });
                            Desc::Leaf(k)
                        }
                    }
                }
                Src::Scalar(scalar) => {
                    if scalar.ty.slots() != 1 {
                        self.poison("a wide scalar");
                        return None;
                    }
                    let key = (scalar.ty, scalar.bits);
                    let at = match self.class_at.get(&key) {
                        Some(&at) => at,
                        None => {
                            let at = plancache::with_tape(|recorder| recorder.note_class(key)).unwrap_or(0);
                            self.class_at.insert(key, at);
                            self.classes.push(key);
                            self.class_sigs.push(ClassSig { ty: scalar.ty, special: false, sensitive: simplify::sensitive(scalar.ty, scalar.bits) });
                            at
                        }
                    };
                    Desc::Scalar(at, scalar.ty)
                }
            };
        }
        Some(out)
    }

    /// A call arrives while recording: its operands are described now, what it made is told to
    /// [`finish_created`] or [`finish_hit`].
    fn begin(&mut self, len: usize, op: &Op, args: &Args, arity: usize) -> Begun {
        if self.poisoned.is_none() && self.entries.len() >= MAX_ENTRIES {
            self.poison("too long");
        }
        self.call = None;
        if self.poisoned.is_some() {
            return Begun::Go(Go::Skip);
        }
        match self.describe(args, arity) {
            Some(raw) => {
                let steps_at_start = plancache::with_tape(|recorder| recorder.steps_len()).unwrap_or(0);
                self.call = Some(Call { op: *op, len, raw, steps_at_start, reason: None });
                Begun::Go(Go::Record)
            }
            None => Begun::Go(Go::Skip),
        }
    }

    /// The final form of the call that made (or was answered with) `node`, if it differs from
    /// what the call asked for in its operands: `None` as a whole if it cannot be described.
    fn final_of(&mut self, node: &Node, call: &Call) -> Option<Option<Box<Final>>> {
        let fop = node.op;
        let farity = fop.kind.arity();
        let Some(fargs) = node.operands() else {
            self.poison("a node without operands");
            return None;
        };
        let fdesc = self.describe(&fargs, farity)?;
        Some((fop != call.op || fdesc != call.raw).then(|| Box::new(Final { op: fop, arity: farity as u8, desc: fdesc })))
    }

    /// Whether following has to try the rewrites again for a call like this one.
    fn rewrites_matter(&self, call: &Call, rounds: u32) -> bool {
        rounds > 0 || call.raw.iter().any(|desc| matches!(desc, Desc::Scalar(c, _) if self.class_sigs[*c as usize].sensitive))
    }
}

/// `create` found why it cuts (called with the reason).
pub(super) fn cut_reason(reason: SplitReason) {
    let _ = STATE.try_with(|state| {
        if let Ok(mut state) = state.try_borrow_mut()
            && let Phase::Rec(rec) = &mut state.phase
            && let Some(call) = &mut rec.call
        {
            call.reason = Some(reason);
        }
    });
}

/// Hash-consing handed out `found` for the call.
pub(super) fn finish_hit(go: Go, rounds: u32, found: &Arc<Node>) {
    if !matches!(go, Go::Record) {
        return;
    }
    let _ = STATE.try_with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else { return };
        let Phase::Rec(rec) = &mut state.phase else { return };
        let Some(call) = rec.call.take() else { return };
        let steps = plancache::with_tape(|recorder| recorder.steps_len()).unwrap_or(0);
        if steps != call.steps_at_start {
            rec.poison("launches in a call that made no node");
        }
        let hit = plancache::with_tape(|recorder| recorder.node_index(found.id)).flatten();
        let fin = if rec.poisoned.is_none() { rec.final_of(found, &call) } else { None };
        match (hit, fin) {
            (Some(j), Some(fin)) => {
                let rewrite = rec.rewrites_matter(&call, rounds);
                plancache::with_tape(|recorder| recorder.note_no_node());
                rec.entries.push(Entry {
                    op: call.op,
                    len: call.len,
                    arity: call.op.kind.arity() as u8,
                    raw: call.raw,
                    kind: Kind::Hit(j),
                    fin,
                    rounds: rounds as u8,
                    rewrite,
                    cand: NONE,
                    below: 0,
                    depth: 0,
                    shift_depth: 0,
                    weight: 0.0,
                    cut: None,
                });
            }
            (None, _) => rec.poison("hash-consing handed out a node of another step"),
            _ => {}
        }
        if let Some(why) = rec.poisoned {
            state.stop(why);
        }
    });
}

/// `create` made `node` for the call.
pub(super) fn finish_created(go: Go, rounds: u32, cut: bool, node: &Arc<Node>) {
    if !matches!(go, Go::Record) {
        return;
    }
    let _ = STATE.try_with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else { return };
        let Phase::Rec(rec) = &mut state.phase else { return };
        let Some(call) = rec.call.take() else { return };
        let at = rec.entries.len() as u32;
        let steps = plancache::with_tape(|recorder| recorder.steps_len()).unwrap_or(0);
        // A cut may launch nothing: the operand it evaluates was computed by an earlier launch of the call.
        let mut cut_info = None;
        if cut {
            cut_info = Some(Box::new(Cut { steps: call.steps_at_start as u32..steps as u32, reason: call.reason, strong: Vec::new() }));
        } else if steps != call.steps_at_start {
            rec.poison("launches in a call that did not cut");
        }
        let fin = if rec.poisoned.is_none() { rec.final_of(node, &call) } else { None };
        if let Some(fin) = fin {
            let (fop, fdesc) = match &fin {
                Some(fin) => (fin.op, fin.desc),
                None => (node.op, call.raw),
            };
            let mut cand = NONE;
            if !matches!(fop.kind, OpKind::Unary(UnOp::Id)) {
                let key = (fop.fingerprint, call.len, fdesc);
                if let Some(&earlier) = rec.cons.get(&key) {
                    cand = earlier;
                }
                rec.cons.insert(key, at);
            }
            if let Some(cut) = &mut cut_info {
                cut.strong = plancache::with_tape(|recorder| recorder.strong_counts()).unwrap_or_default();
            }
            plancache::with_tape(|recorder| recorder.note_node(node));
            let rewrite = rec.rewrites_matter(&call, rounds);
            rec.entries.push(Entry {
                op: call.op,
                len: call.len,
                arity: call.op.kind.arity() as u8,
                raw: call.raw,
                kind: Kind::Create,
                fin,
                rounds: rounds as u8,
                rewrite,
                cand,
                below: node.below_len,
                depth: node.depth,
                shift_depth: node.shift_depth,
                weight: node.weight,
                cut: cut_info,
            });
        }
        if let Some(why) = rec.poisoned {
            state.stop(why);
        }
    });
}

// -- The evaluation that ends a step ----------------------------------------------------------------

/// What `plancache::evaluate` does about the tape.
pub(super) enum Eval {
    /// Not following: evaluate as ever; `Some` if the step was recorded and its tape is wanted.
    Normal(Option<Box<Recording>>),
    /// The plan ran from the tape.
    Replayed,
    /// The plan started and failed: evaluate what is left as ever.
    Failed,
}

/// A step that was recorded, waiting for its evaluation to end.
pub(super) struct Recording {
    rec: Option<Box<Rec>>,
    roots: Vec<u32>,
    pending_eval: i64,
    eval_from: usize,
    strong: Vec<u32>,
    key: Option<[u64; 2]>,
}

/// Called by `plancache::evaluate` with the roots it is about to evaluate.
pub(super) fn eval_begin(jit: &JitState, pending: &[Arc<Node>]) -> Eval {
    jit.follow.foreign_use();
    let fol = STATE.with_borrow_mut(|state| {
        if state.jit != jit.id {
            return Err(Eval::Normal(None));
        }
        if pending.is_empty() {
            // Nothing to compute (an `eval` of arrays that are evaluated): not the end of a step.
            return Err(Eval::Normal(None));
        }
        match std::mem::replace(&mut state.phase, Phase::Idle) {
            Phase::Idle | Phase::Skip => {
                state.set(Phase::Idle);
                Err(Eval::Normal(None))
            }
            Phase::Rec(rec) => {
                state.set(Phase::Skip);
                Err(state.record_end(jit, rec, pending))
            }
            Phase::Fol(fol) => {
                state.set(Phase::Skip);
                Ok(fol)
            }
        }
    })
    ;
    let mut fol = match fol {
        Ok(fol) => fol,
        Err(eval) => return eval,
    };
    let ctx = pending[0].context().clone();
    let tape = Rc::clone(&fol.tape);
    let kind = fol.kind;
    let usable = plancache::usable(jit, &ctx) && jit.plan_skip.load(Ordering::Relaxed) == 0;
    let roots_match = pending.len() == tape.roots.len()
        && pending.iter().zip(&tape.roots).all(|(root, &j)| fol.nodes.get(j as usize).is_some_and(|weak| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(root))));
    let full = fol.pos == tape.entries.len()
        && usable
        && roots_match
        && fol.next_step() == tape.eval_from
        && jit.pending_nodes.load(Ordering::Relaxed) == tape.pending_eval
        && fol.write_epoch == jit.write_epoch.load(Ordering::Relaxed)
        && !jit.follow.shared.load(Ordering::Relaxed)
        && fol.leaves.len() == tape.leaves.len()
        && fol.classes.len() == tape.classes.len()
        && fol.nodes.iter().enumerate().all(|(at, weak)| tape.roots.contains(&(at as u32)) || weak.strong_count() == tape.strong[at] as usize)
        && !takes_over(&fol);
    if !full {
        if tracing() {
            note(&format!(
                "the evaluation does not replay: pos {}/{}, usable {usable}, roots {roots_match}, step {} vs {}, pending {} vs {}, epoch {}, leaves {}/{}, classes {}/{}, strong {:?} vs {:?}",
                fol.pos,
                tape.entries.len(),
                fol.next_step(),
                tape.eval_from,
                jit.pending_nodes.load(Ordering::Relaxed),
                tape.pending_eval,
                fol.write_epoch == jit.write_epoch.load(Ordering::Relaxed),
                fol.leaves.len(),
                tape.leaves.len(),
                fol.classes.len(),
                tape.classes.len(),
                fol.nodes.iter().map(Weak::strong_count).collect::<Vec<_>>(),
                tape.strong
            ));
        }
        fol.register_all();
        jit.follow.stats.diverged.fetch_add(1, Ordering::Relaxed);
        STATE.with_borrow_mut(|state| {
            state.failed(kind, false);
            state.set(Phase::Idle);
        });
        return Eval::Normal(None);
    }
    if verifying() {
        fol.register_all();
        match plancache::walk_key(jit, &ctx, pending) {
            Some(key) if tape.key.is_some_and(|recorded| recorded != key) => {
                panic!("trace following: the graph of a followed step differs from its tape (walk key {key:?} vs {:?})", tape.key)
            }
            _ => {}
        }
    }
    let recording = jit.recording_decisions();
    let replayed = run_steps(&ctx, &mut fol, tape.plan.steps.len());
    if recording && replayed.is_ok() {
        let first = &pending[0];
        jit.note_decision(Decision::new(first.name, first.location, DecisionSite::PlanReplay, "replay"));
    }
    // What is alive and was not finished is a node like any other from here on.
    fol.register_all();
    drop(fol);
    STATE.with_borrow_mut(|state| {
        state.set(Phase::Idle);
        if replayed.is_ok() {
            state.replayed(kind);
        } else {
            state.failed(kind, true);
        }
    });
    match replayed {
        Ok(()) => {
            jit.plan_stats.hits.fetch_add(1, Ordering::Relaxed);
            jit.follow.stats.replayed.fetch_add(1, Ordering::Relaxed);
            Eval::Replayed
        }
        Err(()) => Eval::Failed,
    }
}

impl State {
    /// The evaluation of a recorded step begins.
    fn record_end(&mut self, jit: &JitState, rec: Box<Rec>, pending: &[Arc<Node>]) -> Eval {
        let kind = rec.kind;
        let abandon = move |state: &mut State, jit: &JitState| {
            jit.follow.stats.abandoned.fetch_add(1, Ordering::Relaxed);
            state.failed(kind, true);
            state.set(Phase::Idle);
            Eval::Normal(None)
        };
        if rec.poisoned.is_some() || rec.entries.is_empty() || rec.call.is_some() {
            note("a recorded step that is empty or unfinished at its evaluation");
            return abandon(self, jit);
        }
        let ctx = pending[0].context();
        if !plancache::usable(jit, ctx) || jit.plan_skip.load(Ordering::Relaxed) > 0 || jit.follow.shared.load(Ordering::Relaxed) || rec.write_epoch != jit.write_epoch.load(Ordering::Relaxed) {
            note("the plan cache cannot be used at the evaluation, or a write was made");
            return abandon(self, jit);
        }
        let mut roots = Vec::with_capacity(pending.len());
        for root in pending {
            match plancache::with_tape(|recorder| recorder.node_index(root.id)).flatten() {
                Some(j) => roots.push(j),
                None => {
                    note("a root of another step");
                    return abandon(self, jit);
                }
            }
        }
        let (eval_from, mut strong) = plancache::with_tape(|recorder| (recorder.steps_len(), recorder.strong_counts())).unwrap_or_default();
        // What references a root depends on how the step ends (an evaluation, a read): the arrays
        // that hold it are all that matters, and the roots are computed anyway.
        for &j in &roots {
            if let Some(count) = strong.get_mut(j as usize) {
                *count = 0;
            }
        }
        let key = if verifying() { plancache::walk_key(jit, ctx, pending) } else { None };
        Eval::Normal(Some(Box::new(Recording { rec: Some(rec), roots, pending_eval: jit.pending_nodes.load(Ordering::Relaxed), eval_from, strong, key })))
    }
}

/// The evaluation of a step ended (`recording` is what [`eval_begin`] made of the step, if it was recorded).
pub(super) fn eval_end(jit: &JitState, recording: Option<Box<Recording>>) {
    let Some(mut recording) = recording else { return };
    let Some(rec) = recording.rec.take() else { return };
    let recorder = plancache::tape_end();
    let rec = *rec;
    STATE.with_borrow_mut(|state| {
        state.set(Phase::Idle);
        let plan = match recorder.map(plancache::Recorder::into_plan) {
            Some(Ok(plan)) => plan,
            other => {
                note(&format!("the recording left no plan: {}", other.and_then(Result::err).unwrap_or("none")));
                jit.follow.stats.abandoned.fetch_add(1, Ordering::Relaxed);
                state.failed(rec.kind, false);
                return;
            }
        };
        let mut lens: Vec<u64> = rec.entries.iter().map(|e| e.len as u64).collect();
        lens.sort_unstable();
        lens.dedup();
        let mut rec = rec;
        let classes: Vec<ClassSig> = std::mem::take(&mut rec.class_sigs)
            .into_iter()
            .zip(&rec.classes)
            .map(|(sig, &(ty, bits))| ClassSig { ty: sig.ty, special: ty == ElemTy::U32 && (bits == 0 || lens.binary_search(&bits).is_ok()), sensitive: sig.sensitive })
            .collect();
        let leaves = std::mem::take(&mut rec.leaf_sigs);
        let entries = std::mem::take(&mut rec.entries);
        let boundary = rec.boundary.take();
        let watch = plan
            .takeover_leaves()
            .into_iter()
            .map(|leaf| {
                let holders = entries
                    .iter()
                    .enumerate()
                    .filter_map(|(at, e)| {
                        let (arity, desc) = e.fin.as_deref().map_or((e.arity, &e.raw), |fin| (fin.arity, &fin.desc));
                        let slots = desc.iter().take(arity as usize).filter(|d| **d == Desc::Leaf(leaf)).count() as u32;
                        (slots > 0 && e.kind == Kind::Create).then_some((at as u32, slots))
                    })
                    .collect();
                (leaf, holders)
            })
            .collect();
        let tape = Rc::new(Tape {
            jit: rec.jit,
            stamp: rec.stamp,
            pending_start: rec.pending_start,
            pending_eval: recording.pending_eval,
            entries,
            leaves,
            classes,
            lens,
            roots: std::mem::take(&mut recording.roots),
            plan,
            eval_from: recording.eval_from,
            strong: std::mem::take(&mut recording.strong),
            key: recording.key,
            watch,
            boundary: RefCell::new(boundary),
        });
        state.keep(rec.kind, tape);
        jit.follow.stats.recorded.fetch_add(1, Ordering::Relaxed);
    });
}

impl Drop for Recording {
    /// A recording whose evaluation did not end makes no tape.
    fn drop(&mut self) {
        if let Some(rec) = &self.rec {
            let kind = rec.kind;
            plancache::tape_abort();
            let _ = STATE.try_with(|state| {
                if let Ok(mut state) = state.try_borrow_mut() {
                    state.failed(kind, true);
                    state.set(Phase::Idle);
                }
            });
        }
    }
}

// -- Other ways a step ends -------------------------------------------------------------------------

thread_local! {
    /// How many `settle`s (the cuts inside `create`) this thread is in.
    static SETTLING: Cell<u32> = const { Cell::new(0) };
}

/// Marks the cut inside a `create` that is under way: what it evaluates is not the end of a step.
pub(super) struct Settling;

impl Settling {
    pub(super) fn enter() -> Settling {
        SETTLING.with(|settling| settling.set(settling.get() + 1));
        Settling
    }
}

impl Drop for Settling {
    fn drop(&mut self) {
        SETTLING.with(|settling| settling.set(settling.get() - 1));
    }
}

/// Something outside the tape is about to evaluate `node` by itself (a read of the array): the end
/// of the step. A followed step replays when `node` is all it computes, and goes on as ever
/// otherwise; a recorded one is left unfinished. `true` if the node was evaluated.
pub(super) fn escape(node: &Arc<Node>) -> bool {
    let jit = node.context().jit();
    if SETTLING.with(Cell::get) > 0 {
        return false;
    }
    let following = STATE
        .try_with(|state| {
            let Ok(mut state) = state.try_borrow_mut() else { return false };
            if state.jit != jit.id {
                return false;
            }
            match &state.phase {
                Phase::Fol(_) => true,
                Phase::Rec(rec) => {
                    let kind = rec.kind;
                    jit.follow.stats.abandoned.fetch_add(1, Ordering::Relaxed);
                    state.failed(kind, true);
                    state.set(Phase::Idle);
                    false
                }
                Phase::Skip => {
                    state.set(Phase::Idle);
                    false
                }
                Phase::Idle => false,
            }
        })
        .unwrap_or(false);
    if !following {
        return false;
    }
    matches!(eval_begin(jit, std::slice::from_ref(node)), Eval::Replayed)
}

/// Something that reads the lists phantoms are not on is about to run (a write, a sweep of the
/// live arrays, a reduction, an assignment): the step ends here.
pub(super) fn interrupt(jit: &JitState) {
    if !jit.follow_on() {
        return;
    }
    interrupt_thread();
}

/// [`interrupt`] for a caller that has no context at hand (and so does not look whether following is on).
pub(super) fn interrupt_thread() {
    let _ = STATE.try_with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else { return };
        if matches!(state.phase, Phase::Fol(_) | Phase::Rec(_)) {
            if tracing() {
                note(&format!("interrupt from {}", std::backtrace::Backtrace::force_capture()));
            }
            state.stop("something reads what a phantom is not on");
        }
    });
}

/// `submit_requested` postponed the evaluation: the step has no end to record or replay.
pub(super) fn postponed(jit: &JitState) {
    interrupt(jit);
}

// -- step_boundary ----------------------------------------------------------------------------------

/// The pattern of how the arrays of the step before relate to the leaves of this one.
fn boundary_pattern(prev: &[(Option<Arc<Node>>, u64)], leaves: &[Weak<Storage>]) -> Option<Pattern> {
    let mut pattern = Vec::with_capacity(prev.len());
    for (node, id) in prev {
        let (alive, state) = match node {
            Some(node) => match node.result.get() {
                Some(Ok(storage)) => (true, Some(storage.id())),
                _ => return None,
            },
            None => (false, None),
        };
        let slot = leaves.iter().position(|leaf| leaf.upgrade().is_some_and(|leaf| state.is_some_and(|state| leaf.id() == state) || leaf.producer() == *id));
        pattern.push((alive, slot.map(|slot| slot as u32)));
    }
    Some(pattern)
}

/// `step_boundary` ended without a loop (`exit`) for the step being recorded or followed.
pub(super) fn note_boundary(jit: &JitState, prev: &[(Option<Arc<Node>>, u64)], exit: Exit) {
    if !jit.follow_on() {
        return;
    }
    let _ = STATE.try_with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else { return };
        if state.jit != jit.id {
            return;
        }
        match &mut state.phase {
            Phase::Rec(rec) if rec.poisoned.is_none() => {
                if let Some(pattern) = boundary_pattern(prev, &rec.leaves) {
                    rec.boundary = Some((exit, pattern));
                }
            }
            // The step recorded often is the first one after a reset, which ends in another way: the
            // first followed step teaches the tape what a step like it ends in.
            Phase::Fol(fol) if fol.pos == fol.tape.entries.len() && fol.tape.boundary.borrow().is_none() => {
                if let Some(pattern) = boundary_pattern(prev, &fol.leaves) {
                    *fol.tape.boundary.borrow_mut() = Some((exit, pattern));
                }
            }
            _ => {}
        }
    });
}

/// What `step_boundary` of the evaluation that ends the step being followed would do: what it did
/// for a step like it, if the arrays of the step before relate to the leaves of this one as they
/// did. `prev` is the history of the step before.
pub(super) fn boundary_exit(jit: &JitState, prev: &[(Option<Arc<Node>>, u64)]) -> Option<Exit> {
    if !jit.follow_on() {
        return None;
    }
    STATE
        .try_with(|state| {
            let state = state.try_borrow().ok()?;
            if state.jit != jit.id {
                return None;
            }
            let Phase::Fol(fol) = &state.phase else { return None };
            if fol.pos != fol.tape.entries.len() {
                return None;
            }
            let recorded = fol.tape.boundary.borrow();
            let (exit, pattern) = recorded.as_ref()?;
            (boundary_pattern(prev, &fol.leaves).as_ref() == Some(pattern)).then_some(*exit)
        })
        .ok()
        .flatten()
}
