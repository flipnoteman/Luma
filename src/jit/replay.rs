//! Replaying a traced step. A step of a loop is traced once with its state replaced by
//! placeholders (probes) while the context records its launches; the recorded launches are then
//! replayed with the probes' buffers swapped for the real state, step after step.

use std::hash::Hash;
use std::sync::{Arc, Mutex};

use luma_core::{Buffer, Dispatch, ExecGraph, GraphLaunch};

use crate::array::{Array, Failure, Storage};
use crate::context::{Cmd, CmdArg, Context, RawArg, Recording};
use crate::error::Result;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
use crate::gpu::pager::{Accounted, Pin};
use crate::gpu_type::GpuType;
use crate::kernel::KernelCore;

use super::Node;

/// The launches of one traced step, ready to be encoded again.
pub(crate) struct Plan {
    ctx: Context,
    seed: Vec<(Arc<Storage>, Arc<Storage>)>,
    cmds: Vec<Cmd>,
    /// Keep the segments the commands use resident and alive (not those of dead transient
    /// arrays: see `Recording::share_transients`).
    #[allow(dead_code)]
    pins: Vec<Pin>,
    /// Buffers released while recording; given back to the pool when the plan is dropped.
    retired: Vec<Buffer>,
    /// The buffers of `retired` that the plan made up for its intermediates and shares between
    /// them (see `Recording::share_transients`); no state, probe or result is one.
    transient: Vec<Buffer>,
    /// Counts the bytes of `retired` against the memory budget while the plan lives.
    #[allow(dead_code)]
    resident: Accounted,
}

impl Plan {
    pub(crate) fn new(ctx: &Context, recording: Recording) -> Plan {
        Plan {
            ctx: ctx.clone(),
            seed: recording.seed,
            cmds: recording.cmds,
            pins: recording.pins,
            retired: recording.retired,
            transient: recording.transient,
            resident: recording.resident,
        }
    }

    /// The distinct buffers the plan keeps: those its commands name and those it holds.
    #[cfg(test)]
    pub(crate) fn held_buffers(&self) -> usize {
        let named = self.cmds.iter().flat_map(|cmd| &cmd.args).filter_map(|arg| match arg {
            CmdArg::Storage { buffer, .. } => Some(buffer.id()),
            CmdArg::Uniform(_) => None,
        });
        let held = self.retired.iter().map(Buffer::id).chain(self.pins.iter().map(|pin| pin.buffer().id()));
        named.chain(held).collect::<std::collections::HashSet<u64>>().len()
    }

    /// Bytes of the buffers the plan made up for its intermediates.
    #[cfg(test)]
    pub(crate) fn transient_bytes(&self) -> u64 {
        self.transient.iter().map(Buffer::size).sum()
    }

    /// Bytes of the buffers the plan holds that no array owns (its `retired` ones), and the
    /// bytes counted for them.
    #[cfg(test)]
    pub(crate) fn held_bytes(&self) -> (u64, u64) {
        (self.retired.iter().map(Buffer::size).sum(), self.resident.bytes())
    }

    /// Number of recorded launches.
    pub(crate) fn launches(&self) -> usize {
        self.cmds.len()
    }

    /// Whether some command binds `buffer` writable.
    pub(crate) fn writes(&self, buffer: &Buffer) -> bool {
        self.cmds.iter().any(|cmd| {
            cmd.args.iter().any(|arg| matches!(arg, CmdArg::Storage { buffer: bound, writable: true, .. } if bound == buffer))
        })
    }

    /// The workgroup counts of every command, three words each.
    pub(crate) fn template(&self) -> Vec<u32> {
        self.cmds.iter().flat_map(|cmd| cmd.workgroups).collect()
    }

    /// Encodes every command with storage buffers replaced per `subst` (simultaneous, by `==`);
    /// dispatches come from `indirect` at byte offset `12 * l` for command `l` when given, else
    /// from the recorded workgroups.
    pub(crate) fn replay(&self, subst: &[(Buffer, Buffer)], indirect: Option<&Buffer>) -> Result<()> {
        for (l, cmd) in self.cmds.iter().enumerate() {
            cmd.encode(&self.ctx, subst, indirect.map(|buffer| (buffer, 12 * l as u64)))?;
        }
        Ok(())
    }

    /// Makes the traced results real: copies the seeds, then runs the commands once as recorded.
    pub(crate) fn run_once(&self) -> Result<()> {
        for (from, to) in &self.seed {
            self.ctx.copy_storage(from, to)?;
        }
        self.replay(&[], None)
    }

    /// Makes the commands bind `to[j]` wherever they bind `from[j]`, and lets go of what kept
    /// `from` alive here: their pins and the seeds (used only by [`Plan::run_once`], which a plan
    /// that is replayed with swaps never needs). For a stepper's probes, which every replay swaps
    /// for the real state: placeholders take their names and their memory can go.
    fn forget_probes(&mut self, from: &[Buffer], to: &[Buffer]) {
        for cmd in &mut self.cmds {
            for arg in &mut cmd.args {
                if let CmdArg::Storage { buffer, .. } = arg
                    && let Some(j) = from.iter().position(|probe| probe == buffer)
                {
                    *buffer = to[j].clone();
                }
            }
        }
        self.pins.retain(|pin| !from.contains(pin.buffer()));
        self.seed.clear();
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        for buffer in self.retired.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
}

/// The launches of a graph being put together: steps of a plan, and whatever else runs between
/// them. Nothing is captured before [`GraphBuilder::capture`].
pub(crate) struct GraphBuilder<'a> {
    launches: Vec<(&'a KernelCore, Dispatch<'a>, [u32; 3], Vec<RawArg<'a>>)>,
}

impl<'a> GraphBuilder<'a> {
    pub(crate) fn new() -> Self {
        GraphBuilder { launches: Vec::new() }
    }

    /// Appends the launches of `plan` with storage buffers replaced per `subst` (see
    /// [`Plan::replay`]), dispatched from `indirect` when given.
    pub(crate) fn plan(&mut self, plan: &'a Plan, subst: &'a [(Buffer, Buffer)], indirect: Option<&'a Buffer>) {
        for (l, cmd) in plan.cmds.iter().enumerate() {
            let dispatch = cmd.dispatch(indirect.map(|buffer| (buffer, 12 * l as u64)));
            self.launches.push((&*cmd.kernel, dispatch, cmd.size, cmd.raw_args(subst)));
        }
    }

    /// Appends one launch.
    pub(crate) fn launch(&mut self, kernel: &'a KernelCore, dispatch: Dispatch<'a>, size: [u32; 3], args: Vec<RawArg<'a>>) {
        self.launches.push((kernel, dispatch, size, args));
    }

    /// The launches as one graph the backend can replay; `None` if it cannot.
    pub(crate) fn capture(&self, ctx: &Context) -> Option<Arc<dyn ExecGraph>> {
        let launches: Vec<GraphLaunch<'_>> = self
            .launches
            .iter()
            .map(|(kernel, dispatch, size, args)| GraphLaunch { kernel, dispatch: *dispatch, size: *size, args })
            .collect();
        ctx.hal().capture_graph(&launches)
    }
}

/// Replays of a graph within one call that pay for capturing and instantiating it (about a
/// millisecond for a step of a few dozen launches, against about a microsecond saved per launch).
pub(crate) const PAYING_REPLAYS: usize = 8;

/// A captured graph with its device memory counted as resident.
type Counted = (Arc<dyn ExecGraph>, Accounted);

/// Captured graphs by what they replay, `None` for what the backend could not capture (asked
/// only once). A graph names the buffers it was captured with: the owner of a cache clears it
/// when one of them is replaced.
///
/// A graph is captured when it pays: when the caller is about to replay it often enough
/// ([`PAYING_REPLAYS`]), or the second time it is asked for (an owner kept across runs replays it
/// again). Until then the caller replays the launches one by one.
pub(crate) struct GraphCache<K> {
    ctx: Context,
    /// Each graph with its device memory (see [`ExecGraph::device_bytes`]) counted as resident
    /// while the cache holds it.
    graphs: Mutex<FxMap<K, Option<Counted>>>,
    asked: Mutex<FxMap<K, u32>>,
}

impl<K: Hash + Eq + Copy> GraphCache<K> {
    pub(crate) fn new(ctx: &Context) -> Self {
        GraphCache { ctx: ctx.clone(), graphs: Mutex::new(FxMap::default()), asked: Mutex::new(FxMap::default()) }
    }

    /// The graph under `key` for `replays` replays now: the cached one, or one captured by
    /// `build` if that pays; `None` to replay the launches one by one.
    pub(crate) fn get_or_build(&self, key: K, replays: usize, build: impl FnOnce() -> Option<Arc<dyn ExecGraph>>) -> Option<Arc<dyn ExecGraph>> {
        if let Some(graph) = lock(&self.graphs).get(&key) {
            return graph.as_ref().map(|(graph, _)| Arc::clone(graph));
        }
        let asked = {
            let mut asked = lock(&self.asked);
            let n = asked.entry(key).or_insert(0);
            *n += 1;
            *n
        };
        if replays < PAYING_REPLAYS && asked < 2 {
            return None;
        }
        let graph = build();
        let counted = graph.as_ref().map(|graph| (Arc::clone(graph), Accounted::new(&self.ctx, graph.device_bytes())));
        lock(&self.graphs).insert(key, counted);
        graph
    }

    /// Forgets every graph (and how often each was asked for).
    pub(crate) fn clear(&self) {
        lock(&self.graphs).clear();
        lock(&self.asked).clear();
    }

    /// Whether some graph was captured (not only asked for).
    pub(crate) fn holds_graphs(&self) -> bool {
        lock(&self.graphs).values().any(Option::is_some)
    }
}

/// The graphs a [`Stepper`] replays by itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum StepGraph {
    /// Steps 2 and 3.
    Pair,
    /// Step 2 alone, for the last step of an odd count after the first.
    Even,
}

/// Steps a plan traced on `probes` (standing for `init`) whose results are `outs`, ping-ponging
/// the state between two sets of buffers.
pub(crate) struct Stepper {
    plan: Plan,
    init: Vec<Buffer>,
    probes: Vec<Buffer>,
    ping: Vec<Arc<Storage>>,
    pong: Vec<Arc<Storage>>,
    ping_buffers: Vec<Buffer>,
    pong_buffers: Vec<Buffer>,
    /// Keep every state buffer resident.
    #[allow(dead_code)]
    pins: Vec<Pin>,
    #[allow(dead_code)]
    pong_pins: Vec<Pin>,
    /// Keep the initial state resident.
    #[allow(dead_code)]
    init_pins: Vec<Pin>,
    /// Keep the initial state alive.
    #[allow(dead_code)]
    inits: Vec<Arc<Storage>>,
    /// Steps 2 and 3 as one replayable graph and step 2 alone, as far as the backend has graphs.
    /// Cleared when `pong` is replaced.
    graphs: GraphCache<StepGraph>,
    /// Counts the bytes of the placeholders standing for the probes in `probes`.
    #[allow(dead_code)]
    placeholders: Accounted,
}

impl Stepper {
    /// `Err(plan)` (to be made real with [`Plan::run_once`]) unless the plan can be stepped:
    /// single-buffer states of equal size, results in buffers of their own that the plan writes,
    /// probes that it only reads.
    #[allow(clippy::result_large_err)]
    pub(crate) fn new(
        ctx: &Context,
        plan: Plan,
        init: &[Arc<Storage>],
        probes: &[Arc<Storage>],
        outs: Vec<Arc<Storage>>,
    ) -> std::result::Result<Stepper, Box<Plan>> {
        if plan.launches() == 0 || init.len() != probes.len() || init.len() != outs.len() {
            return Err(Box::new(plan));
        }
        let shaped = init.iter().zip(probes).zip(&outs).all(|((init, probe), out)| {
            [init, probe, out].iter().all(|storage| storage.segments() == 1)
                && init.byte_len() == probe.byte_len()
                && init.byte_len() == out.byte_len()
                && init.layout() == out.layout()
        });
        if !shaped {
            return Err(Box::new(plan));
        }
        let mut pins = Vec::with_capacity(2 * init.len());
        let mut pong_pins = Vec::with_capacity(init.len());
        let mut init_pins = Vec::with_capacity(init.len());
        let buffers = |storages: &[Arc<Storage>], pins: &mut Vec<Pin>| -> Option<Vec<Buffer>> {
            storages
                .iter()
                .map(|storage| {
                    let pin = storage.pin_quiet(0).ok()?;
                    let buffer = pin.buffer().clone();
                    pins.push(pin);
                    Some(buffer)
                })
                .collect()
        };
        let (Some(init_buffers), Some(probe_buffers), Some(ping_buffers)) = (
            buffers(init, &mut init_pins),
            buffers(probes, &mut pins),
            buffers(&outs, &mut pins),
        ) else {
            return Err(Box::new(plan));
        };
        let distinct = ping_buffers.iter().enumerate().all(|(j, out)| {
            plan.writes(out)
                && ping_buffers[..j].iter().all(|other| other != out)
                && init_buffers.iter().chain(&probe_buffers).all(|other| other != out)
        });
        if !distinct || probe_buffers.iter().any(|probe| plan.writes(probe)) {
            return Err(Box::new(plan));
        }
        // Intermediates share buffers only among themselves: a swap of `subst` (state, probes,
        // initial state, ping and pong) never names one, so the sharing cannot be undone or
        // crossed by a replay.
        debug_assert!(
            init_buffers.iter().chain(&probe_buffers).chain(&ping_buffers).all(|b| !plan.transient.contains(b)),
            "a state buffer of a stepper is a transient buffer of its plan"
        );
        let pong: Vec<Arc<Storage>> = outs
            .iter()
            .map(|out| Arc::new(Storage::new(ctx, out.len(), out.layout())))
            .collect();
        let Some(pong_buffers) = buffers(&pong, &mut pong_pins) else {
            return Err(Box::new(plan));
        };
        debug_assert!(
            pong_buffers.iter().all(|b| !plan.transient.contains(b)),
            "a state buffer of a stepper is a transient buffer of its plan"
        );
        // The probes only name the state in the commands: small placeholders take their names,
        // and the probes' memory goes when the arrays holding them do.
        let placeholders: Option<Vec<Buffer>> = probe_buffers.iter().map(|probe| ctx.alloc_buffer(16, probe.usage()).ok()).collect();
        let mut plan = plan;
        let mut counted = Accounted::new(ctx, 0);
        let probe_buffers = match placeholders {
            Some(placeholders) => {
                plan.forget_probes(&probe_buffers, &placeholders);
                pins.retain(|pin| !probe_buffers.contains(pin.buffer()));
                counted.add(placeholders.iter().map(Buffer::size).sum());
                placeholders
            }
            None => probe_buffers,
        };
        Ok(Stepper {
            plan,
            init: init_buffers,
            probes: probe_buffers,
            ping: outs,
            pong,
            ping_buffers,
            pong_buffers,
            pins,
            pong_pins,
            init_pins,
            inits: init.to_vec(),
            graphs: GraphCache::new(ctx),
            placeholders: counted,
        })
    }

    /// Points the first step at a new initial state. `false` (nothing changed) unless `init`
    /// matches the state this stepper was made for and shares no buffer with the buffers the
    /// plan writes or reads as probes.
    pub(crate) fn rebind(&mut self, init: &[Arc<Storage>]) -> bool {
        if init.len() != self.probes.len() {
            return false;
        }
        let shaped = init.iter().zip(&self.ping).all(|(init, ping)| {
            init.segments() == 1
                && init.byte_len() == ping.byte_len()
                && init.layout() == ping.layout()
        });
        if !shaped {
            return false;
        }
        let mut pins = Vec::with_capacity(init.len());
        let mut buffers = Vec::with_capacity(init.len());
        for storage in init {
            let Ok(pin) = storage.pin_quiet(0) else {
                return false;
            };
            buffers.push(pin.buffer().clone());
            pins.push(pin);
        }
        let aliased = buffers.iter().any(|b| {
            self.ping_buffers.contains(b)
                || self.pong_buffers.contains(b)
                || self.probes.contains(b)
        });
        if aliased {
            return false;
        }
        self.init = buffers;
        self.init_pins = pins;
        self.inits = init.to_vec();
        true
    }

    pub(crate) fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Bytes counted for the placeholders that stand for the probes.
    #[cfg(test)]
    pub(crate) fn placeholder_bytes(&self) -> u64 {
        self.placeholders.bytes()
    }

    /// Runs steps `1..=steps`. The first replays on its own (it reads `init`, which `rebind`
    /// changes); the rest cycle through two address-invariant steps (2 and 3), so a backend
    /// that can capture graphs replays them as one. While profiling or recording, and on
    /// backends without graphs, every step is replayed on its own.
    pub(crate) fn run_steps(&self, steps: usize) -> Result<()> {
        if steps == 0 {
            return Ok(());
        }
        self.plan.replay(&self.subst(1), None)?;
        let rest = steps - 1;
        if rest == 0 {
            return Ok(());
        }
        let plain = |from: usize| -> Result<()> {
            for t in from..=steps {
                self.plan.replay(&self.subst(t), None)?;
            }
            Ok(())
        };
        let ctx = &self.plan.ctx;
        if ctx.recording_here() || ctx.is_profiling() {
            return plain(2);
        }
        let launches = self.plan.launches() as u32;
        let mut next = 2;
        if rest >= 2 {
            let pair = self.graphs.get_or_build(StepGraph::Pair, rest / 2, || self.capture_steps(2, 2));
            let Some(pair) = pair else {
                return plain(2);
            };
            for _ in 0..rest / 2 {
                ctx.launch_graph(&*pair, 2 * launches)?;
            }
            next += 2 * (rest / 2);
        }
        if next <= steps {
            match self.graphs.get_or_build(StepGraph::Even, 1, || self.capture_steps(2, 1)) {
                Some(even) => ctx.launch_graph(&*even, launches)?,
                None => plain(next)?,
            }
        }
        Ok(())
    }

    /// Steps `first..first + count` of the plan as one graph, dispatched directly.
    fn capture_steps(&self, first: usize, count: usize) -> Option<Arc<dyn ExecGraph>> {
        let substs: Vec<Vec<(Buffer, Buffer)>> = (first..first + count).map(|t| self.subst(t)).collect();
        let mut builder = GraphBuilder::new();
        for subst in &substs {
            builder.plan(&self.plan, subst, None);
        }
        builder.capture(&self.plan.ctx)
    }

    /// The buffer swaps that make the plan compute step `t` (1-based).
    pub(crate) fn subst(&self, t: usize) -> Vec<(Buffer, Buffer)> {
        let states = self.probes.len();
        let mut swaps = Vec::with_capacity(2 * states);
        for j in 0..states {
            if t == 1 {
                swaps.push((self.probes[j].clone(), self.init[j].clone()));
            } else if t.is_multiple_of(2) {
                swaps.push((self.probes[j].clone(), self.ping_buffers[j].clone()));
                swaps.push((self.ping_buffers[j].clone(), self.pong_buffers[j].clone()));
            } else {
                swaps.push((self.probes[j].clone(), self.pong_buffers[j].clone()));
            }
        }
        swaps
    }

    /// The storages holding the state after step `t`.
    pub(crate) fn state_after(&self, t: usize) -> &[Arc<Storage>] {
        if t % 2 == 1 { &self.ping } else { &self.pong }
    }

    /// Detaches and returns the state after an even number of steps (`pong`) by replacing `pong`
    /// with fresh pooled storages, avoiding any GPU copy kernel; `None` when `t` is odd (`ping`
    /// is baked into the recorded plan).
    pub(crate) fn detach_even(&mut self, ctx: &Context, t: usize) -> Option<Vec<Arc<Storage>>> {
        if !t.is_multiple_of(2) {
            return None;
        }
        let next: Vec<Arc<Storage>> = self
            .ping
            .iter()
            .map(|out| Arc::new(Storage::new(ctx, out.len(), out.layout())))
            .collect();
        let mut pins = Vec::with_capacity(next.len());
        let mut buffers = Vec::with_capacity(next.len());
        for storage in &next {
            let pin = storage.pin_quiet(0).ok()?;
            buffers.push(pin.buffer().clone());
            pins.push(pin);
        }
        if buffers
            .iter()
            .any(|b| self.ping_buffers.contains(b) || self.probes.contains(b))
        {
            return None;
        }
        self.pong_buffers = buffers;
        self.pong_pins = pins;
        self.graphs.clear();
        Some(std::mem::replace(&mut self.pong, next))
    }
}

/// The tracing of one step of a loop: probes standing for the initial state, and the recording
/// of the launches the step makes.
pub(crate) struct Trace {
    ctx: Context,
    pub(crate) init: Vec<Arc<Storage>>,
    pub(crate) probe_storages: Vec<Arc<Storage>>,
    recording: bool,
}

impl Trace {
    /// Evaluates the lazy arrays that exist, the `init` arrays among them, makes the probes that
    /// stand for them and starts recording. Every `init` array must be valid and fit one buffer.
    ///
    /// The lazy arrays are evaluated first because the step may force one of them (it reads a
    /// captured array): a launch evaluating an array that existed before the loop would be
    /// recorded, and replayed at every step, although it has to run once, and it may read buffers
    /// that the replayed step overwrites (`y = &x * 2.0; x += 1.0` in the step) or take over its
    /// operand's buffer.
    pub(crate) fn start<T: GpuType>(init: &[&Array<T>]) -> std::result::Result<(Trace, Vec<Array<T>>), Failure> {
        let (mut trace, probes) = Trace::prepare(init)?;
        trace.begin()?;
        Ok((trace, probes))
    }

    /// [`Trace::start`] without starting the recording: the caller may launch for real first
    /// (see `jit::Attempt`) and then [`Trace::begin`] it.
    pub(crate) fn prepare<T: GpuType>(init: &[&Array<T>]) -> std::result::Result<(Trace, Vec<Array<T>>), Failure> {
        let ctx = init[0].context().clone();
        ctx.jit().evaluate_held();
        let mut storages = Vec::with_capacity(init.len());
        let mut probes = Vec::with_capacity(init.len());
        let mut probe_storages = Vec::with_capacity(init.len());
        for array in init {
            storages.push(Arc::clone(array.storage()?));
            let probe = array.fresh_probe()?;
            probe_storages.push(Arc::clone(probe.storage()?));
            probes.push(probe);
        }
        Ok((Trace { ctx, init: storages, probe_storages, recording: false }, probes))
    }

    pub(crate) fn begin(&mut self) -> std::result::Result<(), Failure> {
        let seed: Vec<_> = self.init.iter().cloned().zip(self.probe_storages.iter().cloned()).collect();
        self.recording = self.ctx.begin_recording(seed.clone());
        if !self.recording {
            for (from, to) in &seed {
                self.ctx.copy_storage(from, to)?;
            }
        }
        Ok(())
    }

    /// Whether the traced code has only built lazy expressions so far: no launch was made, and
    /// nothing aborted the recording.
    pub(crate) fn untouched(&self) -> bool {
        self.recording && self.ctx.recording_here() && self.ctx.recorded() == 0
    }

    /// Stops recording without a plan (nothing was launched).
    pub(crate) fn pause(&mut self) {
        if std::mem::take(&mut self.recording) {
            drop(self.ctx.end_recording().map(|recording| Plan::new(&self.ctx, recording)));
        }
    }

    /// Starts recording again after [`Trace::pause`].
    pub(crate) fn resume(&mut self) -> std::result::Result<(), Failure> {
        self.begin()
    }

    /// Forces `results` (the step applied to the probes), ends the recording and returns what
    /// steps the loop: `None` when everything already ran (an aborted or nested recording, or a
    /// plan that cannot be stepped, which is run once here), with `results` evaluated.
    pub(crate) fn finish<T: GpuType>(
        &mut self,
        results: &[&Array<T>],
        stepable: bool,
    ) -> std::result::Result<Option<Stepper>, Failure> {
        let outs = force(results)?;
        let Some(plan) = self.finish_plan() else { return Ok(None) };
        let stepper = if stepable {
            Stepper::new(&self.ctx, plan, &self.init, &self.probe_storages, outs)
        } else {
            Err(Box::new(plan))
        };
        match stepper {
            Ok(stepper) => Ok(Some(stepper)),
            Err(plan) => {
                plan.run_once()?;
                Ok(None)
            }
        }
    }

    /// Ends the recording and returns its plan to be made real by [`Plan::run_once`] or stepped
    /// by the caller (`None`: aborted, everything already ran).
    pub(crate) fn finish_plan(&mut self) -> Option<Plan> {
        if std::mem::take(&mut self.recording) {
            self.ctx.end_recording().map(|recording| Plan::new(&self.ctx, recording))
        } else {
            None
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.pause();
    }
}

/// Evaluates `arrays` (pending ones of one length in one fused kernel) and returns their storages.
pub(crate) fn force<T: GpuType>(arrays: &[&Array<T>]) -> std::result::Result<Vec<Arc<Storage>>, Failure> {
    let mut groups: Vec<(usize, Vec<Arc<Node>>)> = Vec::new();
    for node in arrays.iter().filter_map(|array| array.pending()) {
        match groups.iter_mut().find(|(len, _)| *len == node.len()) {
            Some((_, nodes)) => nodes.push(Arc::clone(node)),
            None => groups.push((node.len(), vec![Arc::clone(node)])),
        }
    }
    for (_, nodes) in &groups {
        super::evaluate_group(nodes);
    }
    arrays.iter().map(|array| array.storage().map(Arc::clone)).collect()
}
