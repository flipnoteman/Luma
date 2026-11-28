//! Schedules: kernels over arrays of several segments, run segment by segment instead of kernel by kernel.
//!
//! When an array is streamed (it has several segments, or a home) and a context's [`SplitMode`] is not `Off`, a store kernel is not
//! launched at the call: its [`Step`] is recorded in the calling thread's open [`Schedule`], and the output segments are marked
//! *planned* (a cell next to their state, see [`crate::gpu::pager::Segment::settle`]). Every way into a planned segment (a pin, a
//! host read, a host write, the writers of a buffer a step reads) *commits* the schedule first: its steps are compiled into stages
//! and run (`run`). A step's launches are what the eager path launches, with the same arguments, so results are bit-identical.
//!
//! Lineage (P1c, evict-by-recompute) is not set for deferred steps, on purpose: a lineage holds the `Arc<Storage>` inputs of the
//! kernel, and a schedule keeps segments, not storages (a temporary whose handles are gone must not be kept alive by it). What it
//! would buy is evicting a retained output without a readback; the outputs that matter, those of an above-budget stage, are streamed
//! to host blocks as they are computed, so there is nothing left to evict by recompute.
//!
//! Streamed stages (`ring`): arrays that do not fit beside each other are not made resident; inputs go through a ring of device
//! buffers filled by the transfer lanes and outputs to host blocks. What plugs in later: `Origin::Structured` steps and
//! `Sink::Schedule` of `Emit` (canonical-block reductions) and the re-selection hook in `compile`.

mod plan;
mod reduce;
mod ring;
mod run;
mod seg;
#[cfg(test)]
mod tests;

pub(crate) use reduce::{applies as reduces_in_blocks, record as record_reduction, Spec};

use std::cell::{Cell, RefCell};
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::ThreadId;

use super::launch::Shifted;
use super::program::{FusedKey, Ref, Tail};
use super::{lock, OpKind, ShiftKind};
use crate::array::Storage;
use crate::context::Context;
use crate::error::LumaError;
use crate::gpu::hash::FxMap;
use crate::gpu::pager::Segment;
use crate::kernel::KernelCore;
use super::structured::strategy::{Params, SplitClass, StrategyId};
use super::structured::Structured;

pub(crate) use seg::{AsSeg, Held, SegMeta, SegRef};

/// Whether (and how) kernels over streamed arrays are deferred into schedules (`LUMA_SPLIT`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SplitMode {
    /// Kernel-major launches at the call, as in 0.5.2.
    #[default]
    Off,
    /// Schedules with slabs equal to segments.
    Segment,
    /// Whatever the planner picks (the same as `Segment` until slabs shrink below segments).
    Auto,
}

impl SplitMode {
    /// `LUMA_SPLIT`: `off`, `segment` or `auto`.
    pub(crate) fn parse(text: &str) -> Option<SplitMode> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(SplitMode::Off),
            "segment" => Some(SplitMode::Segment),
            "auto" | "on" | "1" | "true" => Some(SplitMode::Auto),
            _ => None,
        }
    }
}

/// What schedules did on a context (see `Context::stream_stats`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamStats {
    /// Commits that ran at least one step.
    pub schedules: u64,
    /// Steps run.
    pub steps: u64,
    /// Stages run.
    pub stages: u64,
    /// Slabs run, summed over stages.
    pub slabs: u64,
    /// Bytes of intermediates that never got a buffer (transients).
    pub transient_bytes_elided: u64,
    /// Arrays that were transients.
    pub transients: u64,
    /// Steps whose strategy the re-selection hook replaced because their stage did not fit.
    pub reselections: u64,
    /// Bytes brought from host memory to the device while schedules ran (every cause: a stage's inputs, evicted segments used again).
    pub upload_bytes: u64,
    /// Bytes moved from the device to host memory while schedules ran (evictions).
    pub download_bytes: u64,
    /// Most device bytes the rings of one stage held.
    pub ring_bytes_peak: u64,
    /// Arrays streamed through a ring (inputs uploaded and outputs downloaded slab by slab), summed over stages.
    pub streamed: u64,
    /// Stages that ran with a window of one slab because two did not fit.
    pub window1_fallbacks: u64,
    /// Stages cut in two because one pass over the slabs did not fit.
    pub cuts: u64,
}

/// The counters behind [`StreamStats`].
#[derive(Debug, Default)]
pub(crate) struct StreamCounters {
    schedules: AtomicU64,
    steps: AtomicU64,
    stages: AtomicU64,
    slabs: AtomicU64,
    transient_bytes_elided: AtomicU64,
    transients: AtomicU64,
    reselections: AtomicU64,
    upload_bytes: AtomicU64,
    download_bytes: AtomicU64,
    pub(super) ring_bytes_peak: AtomicU64,
    pub(super) streamed: AtomicU64,
    pub(super) window1_fallbacks: AtomicU64,
    pub(super) cuts: AtomicU64,
}

impl StreamCounters {
    pub(crate) fn snapshot(&self) -> StreamStats {
        let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        StreamStats {
            schedules: get(&self.schedules),
            steps: get(&self.steps),
            stages: get(&self.stages),
            slabs: get(&self.slabs),
            transient_bytes_elided: get(&self.transient_bytes_elided),
            transients: get(&self.transients),
            reselections: get(&self.reselections),
            upload_bytes: get(&self.upload_bytes),
            download_bytes: get(&self.download_bytes),
            ring_bytes_peak: get(&self.ring_bytes_peak),
            streamed: get(&self.streamed),
            window1_fallbacks: get(&self.window1_fallbacks),
            cuts: get(&self.cuts),
        }
    }
}

/// A schedule that has this many steps commits (see [`record_store`]).
const MAX_STEPS: usize = 512;

/// A schedule whose steps write this many times the memory budget commits. Deferring holds no device memory (planned segments have no
/// buffer until the steps run), so the bound only keeps what a thread defers finite; a bound near the budget would commit after every
/// step over an array larger than the budget, which is the case schedules exist for.
const MAX_OUT_BUDGETS: u64 = 64;

/// How a step uses an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
    ReadWrite,
}

/// How a step reads an array, for stage formation: only `Point` reads are local to a slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Reach {
    /// Element `i` of the output reads element `i`.
    Point,
    /// Through a shift or a view: neighbouring segments are read (a halo).
    Halo,
    /// A single buffer of another length, bound in full for every segment of the output.
    Whole,
    /// Data-dependent indices (`take`) or a sub-program: any element.
    Data,
}

/// The iteration space of a step: two steps are in one space iff both fields are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Space {
    pub(crate) len: usize,
    pub(crate) seg_len: usize,
}

impl Space {
    /// Slabs (= segments) of the space.
    pub(crate) fn slabs(&self) -> usize {
        self.len.div_ceil(self.seg_len.max(1))
    }
}

/// One array of a step, by index in the schedule's array table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bind {
    pub(crate) array: u32,
    pub(crate) access: Access,
    pub(crate) reach: Reach,
}

/// What runs per slab.
pub(crate) enum StepKernel {
    /// A store kernel (the outputs of `binds[..n_out]`, then the inputs in the kernel's binding order).
    Store { core: Arc<KernelCore>, key: Arc<FusedKey> },
    /// The canonical-block first pass of a float reduction (`binds[0]` is its one-segment result), the merge of each slab's block
    /// states into its slot of the stage's carry, and after the last slab the fold of the carry (see `reduce`).
    ReduceBlocks(Box<reduce::Blocks>),
    /// A strategy chosen by the re-selection hook (D9), run whole when its stage is reached.
    Relowered,
}

/// What a step came from.
pub(crate) enum Origin {
    /// An elementwise kernel or the fused first pass of a reduction: its program is in the kernel.
    Fused,
    /// A structured value lowered by a strategy of the registry (06 section 3.3). `value` is `None` for a reduction that is not
    /// registered (an immediate one), which re-selection leaves alone.
    Structured { value: Option<Arc<Structured>>, strategy: StrategyId, params: Params },
}

/// One deferred launch.
pub(crate) struct Step {
    pub(crate) origin: Origin,
    /// How the step splits over slabs: `Map` for a store without shifts, `Halo` with shifts (the radius is not tracked), `Reduce` for a
    /// first pass.
    pub(crate) class: SplitClass,
    pub(crate) kernel: StepKernel,
    pub(crate) space: Space,
    /// Outputs first (`n_out` of them), then the inputs (in-place inputs excluded, as in `launch::pin_inputs`).
    pub(crate) binds: Box<[Bind]>,
    pub(crate) n_out: usize,
    pub(crate) scalars: Box<[u32]>,
    pub(crate) shifted: Option<Shifted>,
    /// Whether the re-selection hook already replaced this step's strategy (at most once per step and commit, see `plan::reselect`).
    pub(crate) reselected: bool,
}

/// What a schedule knows of an array: its segments, not the `Storage` (a temporary whose handles are all dropped still has its data,
/// or its place, until the commit; and `Arc::strong_count(storage) == 1` keeps meaning "nobody else holds it" for the in-place takeover).
pub(crate) struct ArrayRef {
    pub(crate) meta: SegMeta,
    pub(crate) segments: Box<[Arc<Segment>]>,
    windowed: bool,
    /// The first step that uses it.
    pub(crate) first_use: u32,
    /// The first step that writes it, if any.
    pub(crate) first_write: Option<u32>,
    /// Whether a step reads what it wrote over (an in-place step).
    pub(crate) read_write: bool,
    /// The last step that uses it.
    pub(crate) last_use: u32,
    /// Name and place of the step that wrote it first (for the error of a failed commit).
    pub(crate) writer: Option<(&'static str, &'static Location<'static>)>,
}

impl ArrayRef {
    pub(crate) fn as_seg<'a>(&'a self, ctx: &'a Context) -> SegRef<'a> {
        SegRef::new(ctx, self.meta, &self.segments, None, self.windowed)
    }
}

/// The steps and arrays of a schedule until it commits.
pub(crate) struct Open {
    pub(crate) arrays: Vec<ArrayRef>,
    /// Array index by the address of its first segment (aliases of a storage share segments, and so an array).
    by_segments: FxMap<usize, u32>,
    pub(crate) steps: Vec<Step>,
    out_bytes: u64,
}

impl Open {
    fn array(&mut self, seg: SegRef<'_>, at: u32) -> u32 {
        let key = Arc::as_ptr(&seg.segments()[0]) as usize;
        let arrays = &mut self.arrays;
        *self.by_segments.entry(key).or_insert_with(|| {
            arrays.push(ArrayRef {
                meta: seg.meta(),
                segments: seg.segments().to_vec().into_boxed_slice(),
                windowed: seg.windowed(),
                first_use: at,
                first_write: None,
                read_write: false,
                last_use: at,
                writer: None,
            });
            (arrays.len() - 1) as u32
        })
    }
}

enum Phase {
    Open,
    Committing,
    Done(Result<(), Arc<LumaError>>),
}

/// The steps one thread deferred on a context, and what became of them (see the module documentation).
pub(crate) struct Schedule {
    id: u64,
    #[allow(dead_code)]
    owner: ThreadId,
    pub(crate) ctx: Context,
    /// `None` once a commit took the steps.
    open: Mutex<Option<Open>>,
    phase: Mutex<Phase>,
    done: Condvar,
}

static NEXT_SCHEDULE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The id of the schedule this thread is executing, 0 if none: its segments are not settled (they are being written by it).
    static EXECUTING: Cell<u64> = const { Cell::new(0) };
    /// The open schedule of this thread on each context it deferred on.
    static OPEN: RefCell<OpenSchedules> = const { RefCell::new(OpenSchedules(Vec::new())) };
}

/// The open schedules of a thread. A thread that ends commits them (and logs a failure), as dropping a context does for pending work.
struct OpenSchedules(Vec<(u64, Arc<Schedule>)>);

impl Drop for OpenSchedules {
    fn drop(&mut self) {
        for (_, schedule) in self.0.drain(..) {
            if let Err(error) = schedule.commit() {
                log::warn!("luma: a deferred schedule failed when its thread ended: {error}");
            }
        }
    }
}

impl Schedule {
    fn new(ctx: &Context) -> Arc<Schedule> {
        ctx.schedule_opened();
        Arc::new(Schedule {
            id: NEXT_SCHEDULE_ID.fetch_add(1, Ordering::Relaxed),
            owner: std::thread::current().id(),
            ctx: ctx.clone(),
            open: Mutex::new(Some(Open::private())),
            phase: Mutex::new(Phase::Open),
            done: Condvar::new(),
        })
    }

    /// Whether the schedule committed (or fails to have anything left to commit).
    pub(crate) fn is_finished(&self) -> bool {
        matches!(*lock(&self.phase), Phase::Done(_))
    }

    /// Runs the schedule's steps if nobody has yet, and waits for the thread that is doing it. The error of a failed commit is
    /// returned to everyone who asks.
    pub(crate) fn commit(&self) -> Result<(), LumaError> {
        // The thread running the steps pins what they write: nothing to wait for.
        if EXECUTING.with(Cell::get) == self.id {
            return Ok(());
        }
        {
            let mut phase = lock(&self.phase);
            loop {
                match &*phase {
                    Phase::Done(result) => return result.clone().map_err(|error| LumaError::clone(&error)),
                    Phase::Committing => phase = self.done.wait(phase).unwrap_or_else(PoisonError::into_inner),
                    Phase::Open => {
                        *phase = Phase::Committing;
                        break;
                    }
                }
            }
        }
        let open = lock(&self.open).take();
        let result = match open {
            Some(mut open) if !open.steps.is_empty() => self.execute(&mut open),
            _ => Ok(()),
        };
        let shared = result.map_err(Arc::new);
        *lock(&self.phase) = Phase::Done(shared.clone());
        self.done.notify_all();
        self.ctx.schedule_closed();
        shared.map_err(|error| LumaError::clone(&error))
    }

    /// [`Schedule::commit`], as the pin paths of a planned segment call it.
    pub(crate) fn settle(&self) -> Result<(), LumaError> {
        self.commit()
    }

    fn execute(&self, open: &mut Open) -> Result<(), LumaError> {
        struct Executing(u64);
        impl Drop for Executing {
            fn drop(&mut self) {
                EXECUTING.with(|executing| executing.set(self.0));
            }
        }
        // One schedule runs at a time on a context. A thread that is already running one (a nested commit) does not wait for itself.
        let nested = EXECUTING.with(Cell::get) != 0;
        let _lease = (!nested).then(|| lock(self.ctx.schedule_lease()));
        let _executing = Executing(EXECUTING.with(|executing| executing.replace(self.id)));
        let result = self.ctx.abort_recording_for("a schedule committed").and_then(|()| run::run(&self.ctx, open));
        match &result {
            Ok(()) => open.written().for_each(|(array, _)| array.segments.iter().for_each(|segment| segment.clear_planned())),
            Err(error) => {
                let error = Arc::new(error.clone());
                for (array, _) in open.written() {
                    let (op, location) = array.writer.unwrap_or(("elementwise", Location::caller()));
                    let poison = Arc::new(LumaError::Op { op, location, source: Arc::clone(&error) });
                    array.segments.iter().for_each(|segment| segment.poison(&poison));
                }
            }
        }
        result
    }
}

impl Open {
    /// The arrays a step writes, with the index of the first one that does.
    fn written(&self) -> impl Iterator<Item = (&ArrayRef, u32)> {
        self.arrays.iter().filter_map(|array| array.first_write.map(|at| (array, at)))
    }
}

/// A store kernel to defer: what `launch::launch_store` would have launched.
pub(super) struct Recorded<'a> {
    pub(super) core: Arc<KernelCore>,
    pub(super) key: FusedKey,
    pub(super) outputs: &'a [&'a Storage],
    /// Without the in-place input of the key.
    pub(super) inputs: &'a [Arc<Storage>],
    pub(super) scalars: Vec<u32>,
    pub(super) shifted: Option<Shifted>,
    pub(super) name: &'static str,
    pub(super) location: &'static Location<'static>,
}

/// Whether a store kernel of `key` over these arrays is deferred (see the module documentation): the context's mode is not `Off`, no
/// finite check reads each result (it needs it at once), this thread records no graph, and the arrays are streamed.
#[inline]
pub(super) fn deferred(ctx: &Context, key: &FusedKey, outputs: &[&Storage], inputs: &[Arc<Storage>]) -> bool {
    ctx.split_mode() != SplitMode::Off && deferred_here(ctx, key, outputs, inputs)
}

fn deferred_here(ctx: &Context, key: &FusedKey, outputs: &[&Storage], inputs: &[Arc<Storage>]) -> bool {
    key.tail == Tail::Store
        && !outputs.is_empty()
        && outputs.iter().all(|out| out.segments() > 0)
        && inputs.iter().all(|input| input.segments() > 0)
        && (outputs.iter().any(|out| out.segments() > 1) || inputs.iter().any(|input| input.is_streamed()))
        && !ctx.check_finite()
        && !ctx.recording_here()
}

/// Whether a canonical-block reduction recorded now waits in this thread's schedule (see [`deferred`]; a reduction has no key): the
/// context splits, the finite check does not read each result, and this thread records no graph. Otherwise it runs before its
/// call returns, with the same blocks and the same tree.
pub(crate) fn deferred_reduction(ctx: &Context) -> bool {
    ctx.split_mode() != SplitMode::Off && !ctx.check_finite() && !ctx.recording_here()
}

/// How the program reads input `k`.
fn reach_of(key: &FusedKey, k: u32) -> Reach {
    let input = Ref::Input(k);
    if key.subs.iter().flatten().any(|instr| instr.args.contains(&input)) {
        return Reach::Data;
    }
    let mut reach = Reach::Point;
    for instr in key.instrs.iter() {
        if let OpKind::Shift(kind) = instr.op.kind
            && instr.args[0] == input
        {
            reach = reach.max(match kind {
                ShiftKind::Linear | ShiftKind::View { .. } => Reach::Halo,
                _ => Reach::Data,
            });
        }
    }
    reach
}

/// The open schedule of this thread on `ctx`, a new one if there is none or it was taken by a commit.
fn open_schedule(ctx: &Context) -> Result<Arc<Schedule>, LumaError> {
    OPEN.try_with(|all| {
        let mut all = all.borrow_mut();
        if let Some(at) = all.0.iter().position(|(id, _)| *id == ctx.id()) {
            if !matches!(*lock(&all.0[at].1.phase), Phase::Done(_) | Phase::Committing) {
                return Arc::clone(&all.0[at].1);
            }
            all.0.remove(at);
        }
        let schedule = Schedule::new(ctx);
        all.0.push((ctx.id(), Arc::clone(&schedule)));
        schedule
    })
    .map_err(|_| LumaError::Gpu("a kernel was deferred while its thread was ending".into()))
}

/// Whether the calling thread is running a schedule's steps: its planned segments are not settled, it is writing them.
pub(crate) fn executing() -> bool {
    EXECUTING.try_with(|executing| executing.get() != 0).unwrap_or(false)
}

/// Tests: schedules committed because the last handle of their context was dropped.
#[cfg(test)]
pub(crate) static COMMITS_ON_DROP: AtomicU64 = AtomicU64::new(0);

/// A handle of `ctx` is being dropped: commits this thread's open schedule on it if the handles that remain are only the ones the
/// schedule itself holds (its own, and one per segment it remembers), so the context is not kept alive by steps nobody can ask for.
pub(crate) fn context_dropped(ctx: &Context) {
    let held = OPEN
        .try_with(|all| {
            let all = all.try_borrow().ok()?;
            let (_, schedule) = all.0.iter().find(|(id, _)| *id == ctx.id())?;
            let open = schedule.open.try_lock().ok()?;
            Some(1 + open.as_ref().map_or(0, |open| open.arrays.iter().map(|array| array.segments.len()).sum::<usize>()))
        })
        .ok()
        .flatten();
    if let Some(held) = held
        && ctx.handles() <= held + 1
    {
        #[cfg(test)]
        COMMITS_ON_DROP.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = commit_here(ctx) {
            log::warn!("luma: a deferred schedule failed when its context was dropped: {error}");
        }
    }
}

/// Commits this thread's open schedule on `ctx`, if it has one.
pub(crate) fn commit_here(ctx: &Context) -> Result<(), LumaError> {
    let taken = OPEN
        .try_with(|all| {
            let mut all = all.borrow_mut();
            all.0.iter().position(|(id, _)| *id == ctx.id()).map(|at| all.0.remove(at).1)
        })
        .ok()
        .flatten();
    match taken {
        Some(schedule) => schedule.commit(),
        None => Ok(()),
    }
}

/// Defers a store kernel into this thread's open schedule: registers its arrays and marks what it writes as planned.
pub(super) fn record_store(ctx: &Context, step: Recorded<'_>) -> Result<(), LumaError> {
    let Recorded { core, key, outputs, inputs, scalars, shifted, name, location } = step;
    if super::launch::store_tiles(&key).is_some() && outputs[0].segments() != 1 {
        return Err(LumaError::Gpu("tiled program on an array of several segments".into()));
    }
    let first = outputs[0].as_seg();
    let space = Space { len: first.len(), seg_len: first.meta().seg_len };
    let mut reaches = Vec::with_capacity(inputs.len());
    for k in 0..key.input_tys.len() {
        if !key.in_place.iter().any(|&(paired, _)| paired as usize == k) {
            reaches.push(reach_of(&key, k as u32));
        }
    }
    let class = if key.spans.is_empty() { SplitClass::Map } else { SplitClass::Halo { flat_radius: 0 } };
    let key = Arc::new(key);
    let in_place = |o: usize| key.in_place.iter().any(|&(_, output)| output as usize == o);
    let step = Step {
        origin: Origin::Fused,
        class,
        kernel: StepKernel::Store { core, key: Arc::clone(&key) },
        space,
        binds: Box::default(),
        n_out: outputs.len(),
        scalars: scalars.into_boxed_slice(),
        shifted,
        reselected: false,
    };
    defer(ctx, step, outputs, inputs, &reaches, &in_place, first.segment_count(), (name, location))
}

/// Appends `step` (its `binds` are made here) to this thread's open schedule: `outputs` are what it writes (the first `n_out`
/// binds), `inputs` what it reads, each with how it reads it (`reaches`); `in_place(o)` says whether output `o` is read too, and
/// `segments` is the segments of the iteration space (an input of one segment is read whole).
#[allow(clippy::too_many_arguments)]
pub(super) fn defer(
    ctx: &Context,
    step: Step,
    outputs: &[&Storage],
    inputs: &[Arc<Storage>],
    reaches: &[Reach],
    in_place: &dyn Fn(usize) -> bool,
    segments: usize,
    writer: (&'static str, &'static Location<'static>),
) -> Result<(), LumaError> {
    super::plancache::taint("a launch was deferred into a schedule");
    let out_bytes: u64 = outputs.iter().map(|out| out.byte_len()).sum();
    let mut step = Some(step);
    loop {
        let schedule = open_schedule(ctx)?;
        // What another schedule plans must have run before this step is recorded after it.
        for input in inputs {
            input.as_seg().segments().iter().try_for_each(|segment| segment.settle_other(&schedule))?;
        }
        for out in outputs {
            out.as_seg().segments().iter().try_for_each(|segment| segment.plan_for(&schedule))?;
        }
        let full = {
            let mut guard = lock(&schedule.open);
            let Some(open) = guard.as_mut() else {
                // A commit on another thread took the steps between `open_schedule` and here: record into a new one.
                drop(guard);
                OPEN.try_with(|all| all.borrow_mut().0.retain(|(_, known)| !Arc::ptr_eq(known, &schedule))).ok();
                continue;
            };
            open.push(step.take().expect("a step is recorded once"), outputs, inputs, reaches, in_place, segments, writer, out_bytes);
            open.steps.len() >= MAX_STEPS || ctx.pager().budget().is_some_and(|budget| open.out_bytes / MAX_OUT_BUDGETS >= budget)
        };
        for input in inputs {
            input.readers.add_schedule(&schedule);
        }
        for (o, out) in outputs.iter().enumerate() {
            if in_place(o) {
                out.readers.add_schedule(&schedule);
            }
        }
        return if full { commit_here(ctx) } else { Ok(()) };
    }
}

impl Open {
    /// Appends `step` and registers its arrays (see [`defer`]).
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        mut step: Step,
        outputs: &[&Storage],
        inputs: &[Arc<Storage>],
        reaches: &[Reach],
        in_place: &dyn Fn(usize) -> bool,
        segments: usize,
        writer: (&'static str, &'static Location<'static>),
        out_bytes: u64,
    ) {
        let at = self.steps.len() as u32;
        let mut binds = Vec::with_capacity(outputs.len() + inputs.len());
        for (o, out) in outputs.iter().enumerate() {
            let array = self.array(out.as_seg(), at);
            let entry = &mut self.arrays[array as usize];
            entry.first_write.get_or_insert(at);
            entry.read_write |= in_place(o);
            entry.last_use = at;
            entry.writer.get_or_insert(writer);
            binds.push(Bind { array, access: if in_place(o) { Access::ReadWrite } else { Access::Write }, reach: Reach::Point });
        }
        for (input, &reach) in inputs.iter().zip(reaches) {
            let seg = input.as_seg();
            let array = self.array(seg, at);
            self.arrays[array as usize].last_use = at;
            let reach = if seg.segment_count() == 1 && segments > 1 { Reach::Whole } else { reach };
            binds.push(Bind { array, access: Access::Read, reach });
        }
        step.binds = binds.into_boxed_slice();
        self.steps.push(step);
        self.out_bytes += out_bytes;
    }

    /// An open schedule that is not one of a thread's (see `reduce::run_now`).
    fn private() -> Open {
        Open { arrays: Vec::new(), by_segments: FxMap::default(), steps: Vec::new(), out_bytes: 0 }
    }
}

impl Context {
    /// Notes a schedule that is open (to be committed) on this context.
    fn schedule_opened(&self) {
        self.schedules_open().fetch_add(1, Ordering::Relaxed);
    }

    fn schedule_closed(&self) {
        self.schedules_open().fetch_sub(1, Ordering::Relaxed);
    }

    /// Commits the open schedule of the calling thread on this context, if there is one: the trigger of `flush`, `sync`, the
    /// submission of requested arrays and the start of a recording.
    #[inline]
    pub(crate) fn commit_schedules(&self) -> Result<(), LumaError> {
        if self.schedules_open().load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        commit_here(self)
    }
}
