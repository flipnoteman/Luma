//! Replaying a traced step. A step of a loop is traced once with its state replaced by
//! placeholders (probes) while the context records its launches; the recorded launches are then
//! replayed with the probes' buffers swapped for the real state, step after step.

use std::sync::Arc;

use wgpu::Buffer;

use crate::array::{Array, Failure, Storage};
use crate::context::{Cmd, CmdArg, Context, Recording};
use crate::error::Result;
use crate::gpu::pager::Pin;
use crate::gpu_type::GpuType;

use super::Node;

/// The launches of one traced step, ready to be encoded again.
pub(crate) struct Plan {
    ctx: Context,
    seed: Vec<(Arc<Storage>, Arc<Storage>)>,
    cmds: Vec<Cmd>,
    /// Keep the segments the commands use resident and alive.
    #[allow(dead_code)]
    pins: Vec<Pin>,
    /// Buffers released while recording; given back to the pool when the plan is dropped.
    retired: Vec<Buffer>,
}

impl Plan {
    pub(crate) fn new(ctx: &Context, recording: Recording) -> Plan {
        Plan {
            ctx: ctx.clone(),
            seed: recording.seed,
            cmds: recording.cmds,
            pins: recording.pins,
            retired: recording.retired,
        }
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
}

impl Drop for Plan {
    fn drop(&mut self) {
        for buffer in self.retired.drain(..) {
            self.ctx.recycle(buffer);
        }
    }
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
    /// Keep the initial state alive.
    #[allow(dead_code)]
    inits: Vec<Arc<Storage>>,
}

impl Stepper {
    /// `Err(plan)` (to be made real with [`Plan::run_once`]) unless the plan can be stepped:
    /// single-buffer states of equal size, results in buffers of their own that the plan writes,
    /// probes that it only reads.
    pub(crate) fn new(
        ctx: &Context,
        plan: Plan,
        init: &[Arc<Storage>],
        probes: &[Arc<Storage>],
        outs: Vec<Arc<Storage>>,
    ) -> std::result::Result<Stepper, Plan> {
        if plan.launches() == 0 || init.len() != probes.len() || init.len() != outs.len() {
            return Err(plan);
        }
        let shaped = init.iter().zip(probes).zip(&outs).all(|((init, probe), out)| {
            [init, probe, out].iter().all(|storage| storage.segments() == 1)
                && init.byte_len() == probe.byte_len()
                && init.byte_len() == out.byte_len()
                && init.elem() == out.elem()
        });
        if !shaped {
            return Err(plan);
        }
        let mut pins = Vec::with_capacity(4 * init.len());
        let mut buffers = |storages: &[Arc<Storage>]| -> Option<Vec<Buffer>> {
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
        let (Some(init_buffers), Some(probe_buffers), Some(ping_buffers)) = (buffers(init), buffers(probes), buffers(&outs))
        else {
            return Err(plan);
        };
        let distinct = ping_buffers.iter().enumerate().all(|(j, out)| {
            plan.writes(out)
                && ping_buffers[..j].iter().all(|other| other != out)
                && init_buffers.iter().chain(&probe_buffers).all(|other| other != out)
        });
        if !distinct || probe_buffers.iter().any(|probe| plan.writes(probe)) {
            return Err(plan);
        }
        let pong: Vec<Arc<Storage>> =
            outs.iter().map(|out| Arc::new(Storage::new(ctx, out.len(), out.elem()))).collect();
        let Some(pong_buffers) = buffers(&pong) else { return Err(plan) };
        Ok(Stepper {
            plan,
            init: init_buffers,
            probes: probe_buffers,
            ping: outs,
            pong,
            ping_buffers,
            pong_buffers,
            pins,
            inits: init.to_vec(),
        })
    }

    pub(crate) fn plan(&self) -> &Plan {
        &self.plan
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
            Err(plan)
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
