//! A factorisation as a launch list: what a plan cache would key and replay, and what the launches-per-factorisation measurement reads.
//!
//! `capture` records the launches of one call (`lu`, `cholesky`, `solve`, ...) with the context's recording, the way a traced step is
//! recorded (`jit::replay`), and keeps the list with the numbers that explain the call's host time. **Nothing is replayed here** and
//! the context keeps no cache: replay is W3-U5, which is the first consumer. The key says what a replay would have to match: the
//! kind, the shape, the batch, the strategy and its parameters, the device.
//!
//! A recording holds plain launches only. A `clear` or a `copy` of a buffer, a readback, an upload or a sync aborts it
//! ([`Context::abort_recording_for`]), so a factorisation that issues one is not capturable: `capture` returns the reason, which is
//! what U5 needs to know about it. `svd` and `eigh` of a matrix that does not fit workgroup memory read back (the sweeps are recorded in
//! chunks and the convergence flag is read), and the refined solves read back by design: `capture` refuses all of these before it starts.
//!
//! **Replay (W3-U5 part B):** [`FactorPlans`], kept by the context's `JitState`, caches the recording of a call per key and runs it again
//! over the buffers of the next call instead of running the call (its host work is the recursion of lazy `Array` ops, selections and
//! launches; a replay is the encode loop). See [`FactorPlans::call`].

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use std::panic::Location;

use super::svd::{self, Kind};
use super::{panel_width, tri, Dims};
use luma_core::Buffer;

use crate::array::{Array, Storage};
use crate::context::{CmdArg, Recording};
use crate::context::Context;
use crate::error::{LumaError, Result};
use crate::gpu::hash::{FxHasher, FxMap};
use crate::gpu::lock;
use crate::gpu::pager::Accounted;
use crate::jit::replay::Plan;
use crate::jit::structured::boundary::FactorKind;
use crate::jit::structured::families::factor::{boundary_flags, chol_stage, eigh_stage, lu_stage, qr_stage, refined_stage, svd_stage, tri_stage};
use crate::jit::structured::strategy::{first_candidate_in, StrategyId};

/// What a replay of a factorisation must match.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FactorKey {
    pub kind: FactorKind,
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
    /// Right-hand-side columns of a solve.
    pub rhs: u32,
    /// The strategy that ran the top of the call (`lu_recursive`, `chol_blocked`, ...) and the sizes of its parameters.
    pub strategy: StrategyId,
    pub nb: u16,
    pub panel: u16,
    pub tf32: bool,
    /// A hash of the device and what it can do (`Caps`, name, backend, compute units): the `Tuning` and the capabilities enter the
    /// key through it.
    pub device: u64,
}

#[allow(dead_code, reason = "the measurement of W2-U3b: the cache records its own plans")]
impl FactorKey {
    /// The key of a `kind` call on the `d` matrices (`rhs` right-hand-side columns for the solves) on `ctx`: the strategy is the one the
    /// top of the call selects (its stage's first candidate), `nb` the block width of the blocked Cholesky and triangular solves
    /// (`LUMA_CHOL_NB`) and `panel` the width of the LU and QR panels (`LUMA_LINALG_PANEL`), 0 where the kind has none.
    pub(crate) fn for_call(ctx: &Context, kind: FactorKind, d: Dims, rhs: usize) -> FactorKey {
        FactorKey::for_call_on(ctx, kind, d, rhs, device_hash(ctx))
    }

    /// [`FactorKey::for_call`] with the device hash the caller keeps (it formats the backend's name).
    pub(crate) fn for_call_on(ctx: &Context, kind: FactorKind, d: Dims, rhs: usize, device: u64) -> FactorKey {
        let stage = match kind {
            FactorKind::Lu | FactorKind::Solve | FactorKind::Inverse => lu_stage(),
            FactorKind::Cholesky | FactorKind::SolveSpd => chol_stage(),
            FactorKind::Qr | FactorKind::Lstsq => qr_stage(),
            FactorKind::Svd => svd_stage(),
            FactorKind::Eigh => eigh_stage(),
            FactorKind::SolveTriangular => tri_stage(),
            FactorKind::SolveRefined | FactorKind::SolveSpdRefined => refined_stage(),
        };
        let stage_kind = match kind {
            FactorKind::Solve | FactorKind::Inverse => FactorKind::Lu,
            FactorKind::SolveSpd => FactorKind::Cholesky,
            FactorKind::Lstsq => FactorKind::Qr,
            other => other,
        };
        let boundary = boundary_flags(stage_kind, "factor_graph", Location::caller(), d, rhs, false, true, true);
        let strategy = first_candidate_in(stage, ctx, &boundary).map_or(StrategyId::LU_RECURSIVE, |(id, _)| id);
        let blocked = matches!(strategy, StrategyId::CHOL_BLOCKED | StrategyId::TRI_BLOCKED) || matches!(kind, FactorKind::Solve | FactorKind::SolveSpd | FactorKind::Inverse | FactorKind::SolveTriangular);
        let dim = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        FactorKey {
            kind,
            rows: dim(d.rows),
            cols: dim(d.cols),
            batch: dim(d.batch),
            rhs: dim(rhs),
            strategy,
            nb: if blocked { tri::block_size(ctx) as u16 } else { 0 },
            panel: if matches!(stage_kind, FactorKind::Lu | FactorKind::Qr) { panel_width() as u16 } else { 0 },
            tf32: ctx.numerics().tf32_linalg(),
            device,
        }
    }
}

/// A hash of the device and its capabilities.
fn device_hash(ctx: &Context) -> u64 {
    let mut hasher = FxHasher::default();
    ctx.caps().bits().hash(&mut hasher);
    ctx.device_name().hash(&mut hasher);
    format!("{:?}", ctx.backend()).hash(&mut hasher);
    ctx.info().compute_units.hash(&mut hasher);
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    hasher.finish()
}

/// The recorded launch list of one factorisation call.
#[allow(dead_code, reason = "the measurement of W2-U3b: the cache records its own plans")]
pub(crate) struct FactorGraph {
    pub key: FactorKey,
    pub recording: Recording,
    /// Kernel launches in the list.
    pub launches: u32,
    /// Launches per kernel label, in first-use order.
    pub by_kernel: Vec<(Arc<str>, u32)>,
    /// The sum of the bound storage bytes of every launch: what a replay moves.
    pub arg_bytes: u64,
}

#[allow(dead_code, reason = "the measurement of W2-U3b: the cache records its own plans")]
impl FactorGraph {
    /// Records `run()` (a factorisation call on `inputs`) and forces its outputs. `Err` if the kind reads back, if another recording
    /// is open on this thread, or if the call does something a recording cannot hold (the error names it).
    pub(crate) fn capture(ctx: &Context, key: FactorKey, inputs: &[&Array<f32>], run: impl FnOnce() -> Vec<Array<f32>>) -> Result<FactorGraph> {
        refuse_readbacks(ctx, &key)?;
        if ctx.recording_open() {
            return Err(LumaError::Unsupported("a factorisation cannot be captured inside another recording".into()));
        }
        // The operands exist before the recording starts: computing them is not part of the factorisation.
        for input in inputs {
            input.eval()?;
        }
        ctx.sync()?;
        if !ctx.begin_recording(Vec::new()) {
            return Err(LumaError::Unsupported("a factorisation cannot be captured inside another recording".into()));
        }
        let outputs = run();
        let forced = outputs.iter().try_for_each(Array::request);
        ctx.jit().evaluate_requested();
        let reason = ctx.abort_reason();
        let recording = ctx.end_recording();
        forced?;
        match recording {
            Some(recording) => Ok(FactorGraph::of(key, recording)),
            None => Err(LumaError::Unsupported(format!(
                "a factorisation that does a {} cannot be recorded",
                reason.unwrap_or("non-launch operation")
            ))),
        }
    }

    fn of(key: FactorKey, recording: Recording) -> FactorGraph {
        let mut by_kernel: Vec<(Arc<str>, u32)> = Vec::new();
        let mut arg_bytes = 0u64;
        for cmd in &recording.cmds {
            match by_kernel.iter_mut().find(|(label, _)| *label == cmd.kernel.label) {
                Some((_, count)) => *count += 1,
                None => by_kernel.push((Arc::clone(&cmd.kernel.label), 1)),
            }
            arg_bytes += cmd.args.iter().map(|arg| if let CmdArg::Storage { bytes, .. } = arg { *bytes } else { 0 }).sum::<u64>();
        }
        FactorGraph { key, launches: recording.cmds.len() as u32, by_kernel, arg_bytes, recording }
    }
}

/// Refuses what reads back before a recording starts, not mid-way: the SVD and eigh of a matrix the workgroup-memory kernel does not
/// take (their sweeps read the convergence flag), and the refined solves.
fn refuse_readbacks(ctx: &Context, key: &FactorKey) -> Result<()> {
    let refused = |what: &str| Err(LumaError::Unsupported(format!("{what} reads back and cannot be captured")));
    match key.kind {
        FactorKind::SolveRefined | FactorKind::SolveSpdRefined => refused("an adaptive refined solve"),
        FactorKind::Svd | FactorKind::Eigh => {
            let (rr, cc) = (key.rows.max(key.cols) as usize, key.rows.min(key.cols) as usize);
            let kind = if key.kind == FactorKind::Svd { Kind::Svd } else { Kind::Eigh };
            if svd::plan(ctx, kind, true, rr, cc).is_none() {
                return refused("a decomposition of a matrix that does not fit workgroup memory");
            }
            Ok(())
        }
        _ => Ok(()),
    }
}


// -- replay (W3-U5 part B) --

/// Environment variable that switches the replay of factorisations: `off`, `seq` (every backend, one encode per launch; the default
/// everywhere for now), `auto` (as `seq`).
pub(crate) const FACTOR_GRAPHS_ENV: &str = "LUMA_FACTOR_GRAPHS";

/// Plans kept per context (the least recently used goes first).
const MAX_PLANS: usize = 8;

/// The most workspace the plans of a context hold: `min(256 MiB, budget / 8)`.
fn workspace_cap(ctx: &Context) -> u64 {
    let cap = 256u64 << 20;
    ctx.pager().budget().map_or(cap, |budget| cap.min(budget / 8))
}

/// An output of a factorisation call, as a value of its type.
pub(crate) enum Out {
    F32(Array<f32>),
    U32(Array<u32>),
}

impl Out {
    fn request(&self) -> Result<()> {
        match self {
            Out::F32(array) => array.request(),
            Out::U32(array) => array.request(),
        }
    }

    /// The buffer of the evaluated output, if it is a whole buffer of its own.
    fn buffer(&self) -> Option<Arc<Storage>> {
        let (storage, len) = match self {
            Out::F32(array) => (array.storage().ok()?, array.len()),
            Out::U32(array) => (array.storage().ok()?, array.len()),
        };
        (storage.segments() == 1 && !storage.is_alias() && !storage.is_window() && storage.len() == len).then(|| Arc::clone(storage))
    }

    /// The type and shape of the output (an array holds its context, which the plan the context keeps must not).
    fn kind(&self) -> (OutKind, crate::shape::Shape) {
        match self {
            Out::F32(array) => (OutKind::F32, array.shape()),
            Out::U32(array) => (OutKind::U32, array.shape()),
        }
    }
}

#[derive(Clone, Copy)]
enum OutKind {
    F32,
    U32,
}

impl OutKind {
    fn array(self, ctx: &Context, shape: crate::shape::Shape, storage: Arc<Storage>) -> Out {
        match self {
            OutKind::F32 => Out::F32(Array::from_storage(ctx, shape, storage)),
            OutKind::U32 => Out::U32(Array::from_storage(ctx, shape, storage)),
        }
    }
}

/// What a result of a recorded call looks like: a replay allocates a buffer of it and wraps it as the array the call returns.
struct OutSpec {
    kind: OutKind,
    shape: crate::shape::Shape,
    len: usize,
    layout: crate::gpu_type::ElemLayout,
    /// What the commands call the result.
    name: Buffer,
}

/// The replayable form of a [`FactorGraph`]: the commands of one call over placeholders for its inputs and results.
pub(crate) struct FactorPlan {
    plan: Plan,
    /// What the commands call each input.
    inputs: Vec<Buffer>,
    outs: Vec<OutSpec>,
    /// The workspace the plan holds (its transient buffers and the placeholders).
    bytes: u64,
    /// The plan's transient buffers are shared by its commands: one replay at a time.
    running: Mutex<()>,
    /// Counts the placeholders against the memory budget.
    #[allow(dead_code, reason = "held for its drop")]
    placeholders: Accounted,
}

enum Slot {
    Ready { plan: Arc<FactorPlan>, used: u64 },
    /// The call cannot be recorded (it reads back, clears a buffer, returns something that is not a buffer of its own, ...).
    NotRecordable,
}

/// The plans of a context by key: recorded the second time a key is asked for, replayed from then on (see [`FactorPlans::call`]).
pub(crate) struct FactorPlans {
    state: Mutex<PlanTable>,
    device: OnceLock<u64>,
    /// `LUMA_FACTOR_GRAPHS` is not `off`.
    on: AtomicBool,
    hits: AtomicU64,
    recorded: AtomicU64,
}

impl Default for FactorPlans {
    fn default() -> Self {
        let on = !std::env::var(FACTOR_GRAPHS_ENV).is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no"));
        FactorPlans { state: Mutex::default(), device: OnceLock::new(), on: AtomicBool::new(on), hits: AtomicU64::new(0), recorded: AtomicU64::new(0) }
    }
}

impl std::fmt::Debug for FactorPlans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FactorPlans")
    }
}

#[derive(Default)]
struct PlanTable {
    slots: FxMap<(FactorKey, u64), Slot>,
    asked: FxMap<(FactorKey, u64), u32>,
    clock: u64,
    /// The `plan_epoch` the slots were made under: a test hook that changes the planning clears them.
    epoch: u64,
}

impl FactorPlans {
    /// Whether factorisations are replayed (`LUMA_FACTOR_GRAPHS` is not `off`).
    #[inline]
    pub(crate) fn enabled(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// Switches the replay on or off, dropping every plan.
    #[cfg(test)]
    pub(crate) fn set_enabled(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
        self.clear();
    }

    /// What the cache did: (replays, recordings).
    #[cfg(test)]
    pub(crate) fn stats(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.recorded.load(Ordering::Relaxed))
    }

    /// The key of a call on this context.
    pub(crate) fn key(&self, ctx: &Context, kind: FactorKind, d: Dims, rhs: usize) -> FactorKey {
        FactorKey::for_call_on(ctx, kind, d, rhs, *self.device.get_or_init(|| device_hash(ctx)))
    }

    /// The results of the call `key` describes on `inputs` (evaluated buffers of their own), from a plan: `None` where the caller runs the
    /// call itself (the first time a key is asked for, a call that cannot be recorded or replayed, a context the cache does not serve).
    /// The second time a key is asked for, `body` runs once over probe arrays, recording its launches (nothing runs), and the plan is
    /// replayed over the real inputs, so that the call's work is the replay. The results are bit for bit the call's own: the same
    /// kernels run over the same words, in the same order.
    pub(crate) fn call(&self, ctx: &Context, key: FactorKey, inputs: &[&Array<f32>], body: &dyn Fn(&[Array<f32>]) -> Vec<Out>) -> Option<Vec<Out>> {
        if ctx.recording_open() || ctx.is_profiling() || ctx.check_finite() || ctx.can_run_on_host() {
            return None;
        }
        let storages: Vec<Arc<Storage>> = inputs.iter().map(|input| plain(input)).collect::<Option<_>>()?;
        let epoch = ctx.jit().plan_epoch();
        let at = (key, epoch);
        let plan = {
            let mut table = lock(&self.state);
            if table.epoch != epoch {
                *table = PlanTable { epoch, ..PlanTable::default() };
            }
            table.clock += 1;
            let clock = table.clock;
            match table.slots.get_mut(&at) {
                Some(Slot::NotRecordable) => return None,
                Some(Slot::Ready { plan, used }) => {
                    *used = clock;
                    Some(Arc::clone(plan))
                }
                None => {
                    let asked = table.asked.entry(at).or_insert(0);
                    *asked += 1;
                    if *asked < 2 {
                        return None;
                    }
                    None
                }
            }
        };
        let plan = match plan {
            Some(plan) => plan,
            None => {
                // Recorded outside the lock: the call runs the whole library.
                let made = FactorPlan::record(ctx, &key, inputs, body);
                let mut table = lock(&self.state);
                match made {
                    // A plan that alone holds more than the cap would be evicted by the next record and made again: it is replayed
                    // this once and the key is left to the call.
                    Ok(plan) if plan.bytes > workspace_cap(ctx) => {
                        table.slots.insert(at, Slot::NotRecordable);
                        let plan = Arc::new(plan);
                        let outs = plan.replay(ctx, &storages).ok();
                        return outs;
                    }
                    Ok(plan) => {
                        let plan = Arc::new(plan);
                        self.recorded.fetch_add(1, Ordering::Relaxed);
                        let clock = table.clock;
                        table.slots.insert(at, Slot::Ready { plan: Arc::clone(&plan), used: clock });
                        self.evict(&mut table, workspace_cap(ctx));
                        plan
                    }
                    Err(_) => {
                        table.slots.insert(at, Slot::NotRecordable);
                        return None;
                    }
                }
            }
        };
        let outs = plan.replay(ctx, &storages).ok()?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(outs)
    }

    /// Drops the least recently used plans while there are more than [`MAX_PLANS`] or they hold more than `cap` bytes.
    fn evict(&self, table: &mut PlanTable, cap: u64) {
        loop {
            let ready = table.slots.values().filter(|slot| matches!(slot, Slot::Ready { .. })).count();
            let bytes: u64 = table.slots.values().map(|slot| if let Slot::Ready { plan, .. } = slot { plan.bytes } else { 0 }).sum();
            if ready <= MAX_PLANS && bytes <= cap {
                return;
            }
            let oldest = table.slots.iter().filter_map(|(key, slot)| if let Slot::Ready { used, .. } = slot { Some((*used, *key)) } else { None }).min_by_key(|(used, _)| *used);
            match oldest {
                Some((_, key)) => {
                    table.slots.remove(&key);
                    // Asked again from the start: a plan that was too big for the cap is not made over and over.
                    table.asked.remove(&key);
                }
                None => return,
            }
        }
    }

    /// Drops every plan and what was asked.
    #[cfg(test)]
    pub(crate) fn clear(&self) {
        *lock(&self.state) = PlanTable::default();
    }
}

/// The buffer of `input` if it is an evaluated array of one whole buffer of its own.
fn plain(input: &Array<f32>) -> Option<Arc<Storage>> {
    let storage = input.storage_if_valid()?;
    (storage.segments() == 1 && !storage.is_alias() && !storage.is_window() && storage.len() == input.len()).then(|| Arc::clone(storage))
}

impl FactorPlan {
    /// Records `body` over probes of the inputs. `Err` says why the call cannot be recorded.
    fn record(ctx: &Context, key: &FactorKey, inputs: &[&Array<f32>], body: &dyn Fn(&[Array<f32>]) -> Vec<Out>) -> std::result::Result<FactorPlan, &'static str> {
        refuse_readbacks(ctx, key).map_err(|_| "the call reads back")?;
        let probes: Vec<Array<f32>> = inputs.iter().map(|input| Array::<f32>::alloc(ctx, input.shape())).collect::<Result<_>>().map_err(|_| "no memory for the probes")?;
        if !ctx.begin_recording(Vec::new()) {
            return Err("another recording is open");
        }
        let outs = body(&probes);
        let forced = outs.iter().try_for_each(Out::request);
        ctx.jit().evaluate_requested();
        let recording = ctx.end_recording();
        forced.map_err(|_| "a result failed")?;
        let recording = recording.ok_or("the call does something a recording cannot hold")?;
        let plan = Plan::new(ctx, recording);
        // The commands name probe buffers: the call's inputs and results are what a replay swaps.
        let mut names = Vec::with_capacity(probes.len());
        for probe in &probes {
            let storage = probe.storage_if_valid().ok_or("a probe was replaced")?;
            names.push(storage.pin_quiet(0).map_err(|_| "a probe cannot be pinned")?);
        }
        let mut out_pins = Vec::with_capacity(outs.len());
        let mut out_storages = Vec::with_capacity(outs.len());
        for out in &outs {
            let storage = out.buffer().ok_or("a result is not a buffer of its own")?;
            out_pins.push(storage.pin_quiet(0).map_err(|_| "a result cannot be pinned")?);
            out_storages.push(storage);
        }
        let in_buffers: Vec<Buffer> = names.iter().map(|pin| pin.buffer().clone()).collect();
        let out_buffers: Vec<Buffer> = out_pins.iter().map(|pin| pin.buffer().clone()).collect();
        let distinct = out_buffers.iter().enumerate().all(|(j, out)| out_buffers[..j].iter().all(|other| other != out) && in_buffers.iter().all(|other| other != out));
        let written = out_buffers.iter().all(|out| plan.writes(out) && !plan.is_transient(out));
        if !distinct || !written || in_buffers.iter().any(|input| plan.writes(input) || plan.is_transient(input)) {
            return Err("the results are not buffers the commands write, or an input is written");
        }
        // Placeholders take the probes' names, and the probes' memory goes with the arrays that hold them.
        let mut counted = Accounted::new(ctx, 0);
        let mut placeholder_bytes = 0u64;
        let mut make = |buffer: &Buffer| -> std::result::Result<Buffer, &'static str> {
            let placeholder = ctx.alloc_buffer(16, buffer.usage()).map_err(|_| "no memory for a placeholder")?;
            placeholder_bytes += placeholder.size();
            counted.add(placeholder.size());
            Ok(placeholder)
        };
        let in_names: Vec<Buffer> = in_buffers.iter().map(&mut make).collect::<std::result::Result<_, _>>()?;
        let out_names: Vec<Buffer> = out_buffers.iter().map(&mut make).collect::<std::result::Result<_, _>>()?;
        let mut plan = plan;
        plan.forget_probes(&in_buffers, &in_names);
        plan.forget_probes(&out_buffers, &out_names);
        // What else the plan pins (an array that existed before the recording) would keep the context open.
        if plan.pinned_bytes() != 0 {
            return Err("the call reads arrays that existed before it");
        }
        let outs: Vec<OutSpec> = outs.into_iter().zip(&out_storages).zip(out_names).map(|((model, storage), name)| {
            let (kind, shape) = model.kind();
            OutSpec { len: storage.len(), layout: storage.layout(), kind, shape, name }
        }).collect();
        let bytes = plan.transient_bytes() + placeholder_bytes;
        Ok(FactorPlan { plan, inputs: in_names, outs, bytes, running: Mutex::new(()), placeholders: counted })
    }

    /// Runs the commands over `inputs`, into new buffers: the results of the call.
    fn replay(&self, ctx: &Context, inputs: &[Arc<Storage>]) -> Result<Vec<Out>> {
        if inputs.len() != self.inputs.len() {
            return Err(LumaError::Unsupported("a replay with other inputs than its plan".into()));
        }
        let fresh: Vec<Arc<Storage>> = self.outs.iter().map(|out| Arc::new(Storage::new(ctx, out.len, out.layout))).collect();
        let mut pins = Vec::with_capacity(inputs.len() + fresh.len());
        let mut subst = Vec::with_capacity(inputs.len() + fresh.len());
        for (storage, name) in inputs.iter().zip(&self.inputs) {
            let pin = storage.pin_single_read()?;
            subst.push((name.clone(), pin.buffer().clone()));
            pins.push(pin);
        }
        for (storage, out) in fresh.iter().zip(&self.outs) {
            let pin = storage.pin_single()?;
            subst.push((out.name.clone(), pin.buffer().clone()));
            pins.push(pin);
        }
        {
            let _one = lock(&self.running);
            self.plan.replay(&subst, None)?;
        }
        drop(pins);
        Ok(self.outs.iter().zip(fresh).map(|(out, storage)| out.kind.array(ctx, out.shape, storage)).collect())
    }
}


// -- the library's calls --

/// The cache the call on `a` (and `b`) may use, with the key: `None` for a context or an input the cache does not serve.
fn served<'a>(a: &'a Array<f32>, b: Option<&Array<f32>>) -> Option<&'a FactorPlans> {
    let plans = &a.context().jit().factor_plans;
    (plans.enabled() && !a.has_props() && plain(a).is_some() && b.is_none_or(|b| plain(b).is_some() && !b.has_props())).then_some(plans)
}

/// `a.lu()` from a plan, if there is one (see [`FactorPlans::call`]).
pub(crate) fn lu_cached(a: &Array<f32>, loc: &'static Location<'static>) -> Option<super::lu::Lu> {
    let plans = served(a, None)?;
    let ctx = a.context();
    let key = plans.key(ctx, FactorKind::Lu, Dims::of(a).ok()?, 0);
    let mut outs = plans.call(ctx, key, &[a], &|ins| {
        let f = super::lu::factor(&ins[0], "lu", loc);
        vec![Out::F32(f.lu), Out::U32(f.perm)]
    })?
    .into_iter();
    match (outs.next()?, outs.next()?) {
        (Out::F32(lu), Out::U32(perm)) => Some(super::lu::Lu { lu, perm }),
        _ => None,
    }
}

/// `a.qr()`.
pub(crate) fn qr_cached(a: &Array<f32>, loc: &'static Location<'static>) -> Option<super::qr::Qr> {
    let plans = served(a, None)?;
    let ctx = a.context();
    let d = Dims::of(a).ok().filter(|d| d.rows >= d.cols)?;
    let key = plans.key(ctx, FactorKind::Qr, d, 0);
    let mut outs = plans.call(ctx, key, &[a], &|ins| {
        let f = super::qr::factor(&ins[0], "qr", loc);
        vec![Out::F32(f.qr), Out::F32(f.tau), Out::F32(f.t)]
    })?
    .into_iter();
    match (outs.next()?, outs.next()?, outs.next()?) {
        (Out::F32(qr), Out::F32(tau), Out::F32(t)) => Some(super::qr::Qr { qr, tau, t }),
        _ => None,
    }
}

/// The lower Cholesky factor of `a` (before its properties are set).
pub(crate) fn cholesky_cached(a: &Array<f32>, loc: &'static Location<'static>) -> Option<Array<f32>> {
    let plans = served(a, None)?;
    let ctx = a.context();
    let key = plans.key(ctx, FactorKind::Cholesky, Dims::square(a).ok()?, 0);
    match plans.call(ctx, key, &[a], &|ins| vec![Out::F32(super::chol::lower(&ins[0], loc))])?.into_iter().next()? {
        Out::F32(l) => Some(l),
        Out::U32(_) => None,
    }
}

/// Whether `b` is a matrix of right-hand sides of the square systems of `d` (not the vector form `rhs_matrix` reshapes): its columns.
fn rhs_columns(b: &Array<f32>, d: Dims) -> Option<usize> {
    let [k, rows, batch, last] = b.shape().dims();
    let vector = rows == 1 && k == d.rows && d.rows > 1;
    (rows == d.rows && batch == d.batch && last == 1 && !vector).then_some(k)
}

/// `a.solve(b)` for a square `a` and a matrix `b`.
pub(crate) fn solve_cached(a: &Array<f32>, b: &Array<f32>, d: Dims, loc: &'static Location<'static>) -> Option<Array<f32>> {
    let plans = served(a, Some(b))?;
    let ctx = a.context();
    let key = plans.key(ctx, FactorKind::Solve, d, rhs_columns(b, d)?);
    match plans.call(ctx, key, &[a, b], &|ins| {
        let f = super::lu::factor(&ins[0], "solve", loc);
        vec![Out::F32(super::lu::solve_with(&f, &ins[1], "solve", loc))]
    })?
    .into_iter()
    .next()?
    {
        Out::F32(x) => Some(x),
        Out::U32(_) => None,
    }
}

/// `a.inverse()`.
pub(crate) fn inverse_cached(a: &Array<f32>, loc: &'static Location<'static>) -> Option<Array<f32>> {
    let plans = served(a, None)?;
    let ctx = a.context();
    let key = plans.key(ctx, FactorKind::Inverse, Dims::square(a).ok()?, 0);
    match plans.call(ctx, key, &[a], &|ins| {
        let f = super::lu::factor(&ins[0], "inverse", loc);
        vec![Out::F32(super::lu::inverse_with(&f, "inverse", loc))]
    })?
    .into_iter()
    .next()?
    {
        Out::F32(x) => Some(x),
        Out::U32(_) => None,
    }
}

/// `a.solve_spd(b)` for a matrix `b`.
pub(crate) fn solve_spd_cached(a: &Array<f32>, b: &Array<f32>, loc: &'static Location<'static>) -> Option<Array<f32>> {
    let plans = served(a, Some(b))?;
    let ctx = a.context();
    let d = Dims::square(a).ok()?;
    let key = plans.key(ctx, FactorKind::SolveSpd, d, rhs_columns(b, d)?);
    match plans.call(ctx, key, &[a, b], &|ins| {
        let l = super::chol::lower(&ins[0], loc);
        vec![Out::F32(super::chol::cholesky_solve(&l, &ins[1], loc))]
    })?
    .into_iter()
    .next()?
    {
        Out::F32(x) => Some(x),
        Out::U32(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextOptions;
    use crate::BackendKind;

    fn cpu() -> Context {
        Context::with_options(ContextOptions::default().backend(BackendKind::Cpu).calibrate(false)).unwrap()
    }

    /// A diagonally dominant symmetric matrix: SPD, and it needs no pivoting.
    fn spd(ctx: &Context, n: usize) -> Array<f32> {
        Array::from_slice_on(ctx, &(0..n * n).map(|i| if i % (n + 1) == 0 { n as f32 } else { ((i * 31) % 7) as f32 * 0.01 }).collect::<Vec<_>>(), [n, n]).unwrap()
    }

    fn square(n: usize) -> Dims {
        Dims { rows: n, cols: n, batch: 1 }
    }

    /// What a call computes.
    type Call<'a> = Box<dyn Fn() -> Vec<Array<f32>> + 'a>;

    /// The calls of a factorisation test: name, kind, and what it computes.
    fn calls<'a>(a: &'a Array<f32>, b: &'a Array<f32>) -> Vec<(&'static str, FactorKind, Call<'a>)> {
        vec![
            ("lu", FactorKind::Lu, Box::new(|| vec![a.lu().lu])),
            ("cholesky", FactorKind::Cholesky, Box::new(|| vec![a.cholesky()])),
            ("qr", FactorKind::Qr, Box::new(|| {
                let f = a.qr();
                vec![f.qr, f.t]
            })),
            ("solve", FactorKind::Solve, Box::new(|| vec![a.solve(b)])),
            ("solve_spd", FactorKind::SolveSpd, Box::new(|| vec![a.solve_spd(b)])),
            ("inverse", FactorKind::Inverse, Box::new(|| vec![a.inverse()])),
        ]
    }

    /// What the profiler counts for the same call is what the graph holds. (The CPU backend has no launch timer, so there the graph's
    /// own count is checked to be non-zero and the profile comparison is the GPU gate's.) A factorisation that clears a buffer (the
    /// blocked Cholesky starts from a zeroed output) is not capturable, and says why.
    #[test]
    fn factor_graph_matches_profile_launches() {
        let ctx = Context::new().unwrap();
        let sizes: &[usize] = if ctx.info().cpu { &[16, 100] } else { &[16, 100, 700] };
        for &n in sizes {
            let a = spd(&ctx, n);
            let b = Array::<f32>::from_slice_on(&ctx, &vec![1.0; n], [1, n]).unwrap();
            a.eval().unwrap();
            b.eval().unwrap();
            for (name, kind, call) in calls(&a, &b) {
                let key = FactorKey::for_call(&ctx, kind, square(n), 1);
                let captured = FactorGraph::capture(&ctx, key, &[&a, &b], &call);
                // The blocked Cholesky starts from a zeroed output, and the look-back triangular solve of a GPU zeroes its ticket buffer.
                let must_clear = matches!(kind, FactorKind::Cholesky | FactorKind::SolveSpd) && n > super::super::chol::leaf(&ctx);
                match captured {
                    Ok(graph) => {
                        assert!(!must_clear, "{name} n={n}: a blocked Cholesky clears its output, which a recording cannot hold");
                        assert!(graph.launches > 0 && graph.launches as usize == graph.recording.cmds.len(), "{name} n={n}");
                        assert_eq!(graph.by_kernel.iter().map(|(_, count)| count).sum::<u32>(), graph.launches);
                        assert!(graph.arg_bytes > 0);
                        if ctx.profiling_supported() {
                            let (_, profile) = ctx.profile(|| call().iter().for_each(|x| x.eval().unwrap())).unwrap();
                            assert_eq!(profile.launches(), graph.launches, "{name} n={n}");
                        }
                        // The recording left nothing behind: the next one starts.
                        assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some());
                    }
                    Err(error) => {
                        assert!(error.to_string().contains("clear"), "{name} n={n}: {error}");
                        assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some(), "an aborted capture ends cleanly");
                    }
                }
            }
        }
    }

    #[test]
    fn factor_graph_is_stable() {
        let ctx = cpu();
        let a = spd(&ctx, 100);
        let b = Array::<f32>::from_slice_on(&ctx, &[1.0; 100], [1, 100]).unwrap();
        for (name, kind, call) in calls(&a, &b).into_iter().filter(|(name, ..)| matches!(*name, "lu" | "qr" | "solve" | "inverse")) {
            let key = FactorKey::for_call(&ctx, kind, square(100), 1);
            let first = FactorGraph::capture(&ctx, key, &[&a, &b], &call).unwrap();
            let second = FactorGraph::capture(&ctx, key, &[&a, &b], &call).unwrap();
            assert_eq!(first.by_kernel, second.by_kernel, "{name}");
            assert_eq!((first.launches, first.arg_bytes), (second.launches, second.arg_bytes), "{name}");
        }
    }

    #[test]
    fn factor_graph_refuses_readbacks() {
        let ctx = cpu();
        // 200 x 200 does not fit the workgroup-memory SVD: its sweeps read the convergence flag back.
        let a = Array::<f32>::from_slice_on(&ctx, &(0..200 * 200).map(|i| ((i * 37) % 11) as f32 * 0.1 + if i % 201 == 0 { 5.0 } else { 0.0 }).collect::<Vec<_>>(), [200, 200]).unwrap();
        a.eval().unwrap();
        for kind in [FactorKind::Svd, FactorKind::Eigh, FactorKind::SolveRefined, FactorKind::SolveSpdRefined] {
            let key = FactorKey::for_call(&ctx, kind, square(200), 1);
            let refused = FactorGraph::capture(&ctx, key, &[&a], || panic!("a refused capture must not run the call")).map(drop).unwrap_err();
            assert!(matches!(refused, LumaError::Unsupported(ref what) if what.contains("reads back")), "{kind:?}: {refused}");
            assert!(ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some(), "{kind:?} left a recording open");
        }
        // A small decomposition (one workgroup, no readback) is captured.
        let small = Array::<f32>::from_slice_on(&ctx, &(0..16 * 16).map(|i| ((i * 37) % 11) as f32 + if i % 17 == 0 { 5.0 } else { 0.0 }).collect::<Vec<_>>(), [16, 16]).unwrap();
        small.eval().unwrap();
        let key = FactorKey::for_call(&ctx, FactorKind::Svd, square(16), 0);
        let graph = FactorGraph::capture(&ctx, key, &[&small], || vec![small.svd().s]).unwrap();
        assert_eq!(graph.launches, 1);
        // Inside another recording a capture does not start.
        assert!(ctx.begin_recording(Vec::new()));
        let key = FactorKey::for_call(&ctx, FactorKind::Lu, square(16), 0);
        assert!(FactorGraph::capture(&ctx, key, &[&small], || vec![small.lu().lu]).is_err());
        assert!(ctx.end_recording().is_some());
    }

    /// The key says which strategy and which sizes ran: another block width (`LUMA_CHOL_NB`) is another key. The environment is set before
    /// a child process starts (see `the_environment_gives_other_keys`).
    #[test]
    #[ignore = "run by the_key_follows_the_environment in a child process with its own environment"]
    fn key_child() {
        let ctx = cpu();
        let key = FactorKey::for_call(&ctx, FactorKind::Cholesky, square(300), 0);
        println!("child-key={key:?}");
    }

    #[test]
    fn the_key_follows_the_environment() {
        let run = |nb: &str| {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["dense::graph::tests::key_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
                .env("LUMA_CHOL_NB", nb)
                .env("LUMA_DEVICE", "cpu")
                .env("LUMA_CALIBRATE", "0")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8_lossy(&output.stdout).lines().find_map(|line| line.split_once("child-key=").map(|(_, key)| key.to_owned())).expect("the child prints its key")
        };
        let (narrow, wide) = (run("32"), run("64"));
        assert!(narrow.contains("nb: 32") && wide.contains("nb: 64"), "{narrow} / {wide}");
        assert!(narrow.contains("strategy: StrategyId(Factor, 6)") && narrow != wide, "{narrow} / {wide}");
        // The kind and the shape are part of it as well.
        let ctx = cpu();
        let key = |kind, n| FactorKey::for_call(&ctx, kind, square(n), 1);
        assert_ne!(key(FactorKind::Lu, 100), key(FactorKind::Solve, 100));
        assert_ne!(key(FactorKind::Lu, 100), key(FactorKind::Lu, 101));
        assert_eq!(key(FactorKind::Lu, 100), key(FactorKind::Lu, 100));
    }

    /// The context of the environment (`LUMA_DEVICE`, `LUMA_BACKEND`): the replay is tested on every device.
    fn device() -> Context {
        Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap()
    }

    // -- replay (W3-U5 part B) --

    fn bits(array: &Array<f32>) -> Vec<u32> {
        array.to_vec().unwrap().iter().map(|v| v.to_bits()).collect()
    }

    /// A well conditioned matrix of any shape: a pseudo-random fill with a heavy diagonal (SPD when square and symmetrised).
    fn matrix(ctx: &Context, rows: usize, cols: usize, batch: usize, seed: u64) -> Array<f32> {
        let mut state = seed;
        let data: Vec<f32> = (0..rows * cols * batch)
            .map(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                if (i % (cols * rows)).is_multiple_of(cols + 1) && rows == cols { v + cols as f32 * 0.5 } else { v }
            })
            .collect();
        let a = Array::from_slice_on(ctx, &data, [cols, rows, batch]).unwrap();
        a.eval().unwrap();
        a
    }

    /// The calls whose results are compared: the bits of every output, by name.
    fn run_call(name: &str, a: &Array<f32>, b: &Array<f32>) -> Vec<Vec<u32>> {
        match name {
            "lu" => {
                let f = a.lu();
                vec![bits(&f.lu), f.perm.to_vec().unwrap()]
            }
            "qr" => {
                let f = a.qr();
                vec![bits(&f.qr), bits(&f.tau), bits(&f.t)]
            }
            "cholesky" => vec![bits(&a.cholesky())],
            "solve" => vec![bits(&a.solve(b))],
            "inverse" => vec![bits(&a.inverse())],
            "solve_spd" => vec![bits(&a.solve_spd(b))],
            _ => unreachable!(),
        }
    }

    /// First, second (records) and third (replays) calls give the bits of the call that no cache is in: each kind, a few sizes, one matrix and a
    /// batch, and the cache replays where the call is recordable.
    #[test]
    fn a_replayed_factorisation_equals_the_call() {
        let ctx = device();
        let plans = &ctx.jit().factor_plans;
        for name in ["lu", "qr", "cholesky", "solve", "inverse", "solve_spd"] {
            for (n, batch) in [(32, 1), (64, 1), (65, 1), (130, 1), (32, 16)] {
                // A symmetric dominant matrix: valid for every kind.
                let raw = matrix(&ctx, n, n, batch, 7 + n as u64);
                let a = (&raw + &raw.transpose()).cache();
                a.eval().unwrap();
                let b = matrix(&ctx, n, 3, batch, 11).reshape([3, n, batch]);
                b.eval().unwrap();
                plans.set_enabled(false);
                let want = run_call(name, &a, &b);
                plans.set_enabled(true);
                let before = plans.stats();
                for call in 0..4 {
                    assert_eq!(run_call(name, &a, &b), want, "{name} n={n} batch={batch} call {call}");
                }
                let after = plans.stats();
                // A key that records replays from its second call on.
                if after.1 > before.1 {
                    assert!(after.0 - before.0 >= 3, "{name} n={n} batch={batch}: {before:?} -> {after:?}");
                }
            }
        }
    }

    #[test]
    fn the_cache_records_what_it_can_and_remembers_what_it_cannot() {
        let ctx = device();
        ctx.jit().factor_plans.set_enabled(true);
        let plans = &ctx.jit().factor_plans;
        let a = matrix(&ctx, 64, 64, 1, 3);
        let b = matrix(&ctx, 64, 2, 1, 5).reshape([2, 64]);
        let before = plans.stats();
        for _ in 0..5 {
            let _ = a.lu().lu.to_vec().unwrap();
            let _ = a.solve(&b).to_vec().unwrap();
        }
        let after = plans.stats();
        assert!(after.1 - before.1 == 2 && after.0 - before.0 == 8, "{before:?} -> {after:?}");
        // Another shape is another plan, and a lazy input bypasses the cache.
        let lazy = &a * 2.0;
        let before = plans.stats();
        for _ in 0..4 {
            let _ = lazy.lu().lu.to_vec().unwrap();
        }
        assert_eq!(plans.stats(), before);
    }

    /// Two inputs through one plan do not leak state: the second result is its own call's.
    #[test]
    fn one_plan_serves_different_inputs() {
        let ctx = device();
        let plans = &ctx.jit().factor_plans;
        let (a, c) = (matrix(&ctx, 64, 64, 1, 21), matrix(&ctx, 64, 64, 1, 22));
        plans.set_enabled(false);
        let (want_a, want_c) = (run_call("lu", &a, &a), run_call("lu", &c, &c));
        assert_ne!(want_a, want_c);
        plans.set_enabled(true);
        for round in 0..4 {
            assert_eq!(run_call("lu", &a, &a), want_a, "round {round}");
            assert_eq!(run_call("lu", &c, &c), want_c, "round {round}");
        }
        assert!(plans.stats().0 >= 6);
    }

    /// Singular and NaN input behave as the call does.
    #[test]
    fn singular_and_nan_inputs_match_the_call() {
        let ctx = device();
        let plans = &ctx.jit().factor_plans;
        let n = 64;
        let mut data = vec![1.0f32; n * n];
        let singular = Array::from_slice_on(&ctx, &data, [n, n]).unwrap();
        data[5] = f32::NAN;
        let nan = Array::from_slice_on(&ctx, &data, [n, n]).unwrap();
        for a in [&singular, &nan] {
            a.eval().unwrap();
            plans.set_enabled(false);
            let want = run_call("lu", a, a);
            plans.set_enabled(true);
            for _ in 0..4 {
                assert_eq!(run_call("lu", a, a), want);
            }
        }
    }

    /// The plans are bounded: more keys than `MAX_PLANS` evict the least recently used, and a plan that is dropped frees its workspace.
    #[test]
    fn the_plans_are_bounded() {
        let ctx = device();
        let plans = &ctx.jit().factor_plans;
        let before = ctx.memory_stats().resident_bytes;
        for n in 40..40 + 2 * MAX_PLANS {
            let a = matrix(&ctx, n, n, 1, n as u64);
            for _ in 0..3 {
                a.lu().lu.eval().unwrap();
            }
        }
        let ready = lock(&plans.state).slots.values().filter(|slot| matches!(slot, Slot::Ready { .. })).count();
        assert!(ready <= MAX_PLANS, "{ready} plans");
        plans.clear();
        ctx.sync().unwrap();
        assert!(ctx.memory_stats().resident_bytes <= before + (1 << 20), "{} -> {}", before, ctx.memory_stats().resident_bytes);
    }

    /// The plans a context keeps do not keep it open.
    #[test]
    fn a_context_with_plans_closes() {
        let ctx = device();
        ctx.jit().factor_plans.set_enabled(true);
        let weak = ctx.downgrade();
        let a = matrix(&ctx, 64, 64, 1, 5);
        for _ in 0..4 {
            a.lu().lu.eval().unwrap();
        }
        assert!(ctx.jit().factor_plans.stats().0 >= 2);
        drop(a);
        drop(ctx);
        assert!(weak.upgrade().is_none(), "the context is still open");
    }
}
