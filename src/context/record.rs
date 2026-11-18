//! Recording launches: while a recording is active on a thread, `Context::launch` appends the
//! launch to a list instead of encoding it, so that a traced piece of code (one step of a loop)
//! can be replayed many times (see `jit::replay`). Anything that is not a plain launch and needs
//! the GPU state to be real (copies, uploads, readbacks, syncs, indirect launches) first aborts
//! the recording: the captured commands are then encoded in order, as if they had never been held
//! back, and the traced code carries on running normally.
//!
//! A recording belongs to the thread that began it and only that thread's launches are captured
//! by it, so the recordings of the threads that share a context live in thread-local storage:
//! every thread has its own (traces of several threads run side by side, none waits for or
//! disturbs another's), and the launch path of a thread that is not recording looks at nothing
//! but its own thread-local list.
//!
//! Buffers released while recording stay out of the pool, because captured commands still name
//! them, so a step that makes many intermediates would hold one buffer for each. When the
//! recording ends, the intermediates that were allocated and released inside it share buffers
//! whenever the commands that use them do not overlap (see [`Recording::share_transients`]).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use luma_core::{Buffer, BufferUsage};

use crate::array::Storage;
use crate::error::Result;
use crate::gpu::hash::{FxMap, FxSet};
use crate::gpu::pager::{take_dead_buffers, Accounted, Pin};
use crate::kernel::KernelCore;

use super::{Context, RawArg};

/// One captured launch.
#[derive(Debug)]
pub(crate) struct Cmd {
    pub kernel: Arc<KernelCore>,
    pub workgroups: [u32; 3],
    pub size: [u32; 3],
    pub args: Vec<CmdArg>,
}

/// One captured argument of a launch.
#[derive(Debug)]
pub(crate) enum CmdArg {
    Storage { buffer: Buffer, bytes: u64, writable: bool },
    Uniform(Box<[u8]>),
}

impl Cmd {
    /// Encodes the launch with storage buffers replaced per `subst` (simultaneous, by `==`).
    /// The workgroup counts come from `indirect` (buffer, byte offset) when given.
    pub(crate) fn encode(&self, ctx: &Context, subst: &[(Buffer, Buffer)], indirect: Option<(&Buffer, u64)>) -> Result<()> {
        let args: Vec<RawArg<'_>> = self
            .args
            .iter()
            .map(|arg| match arg {
                CmdArg::Storage { buffer, bytes, writable } => {
                    let buffer = subst.iter().find(|(from, _)| from == buffer).map_or(buffer, |(_, to)| to);
                    RawArg::Storage { buffer, bytes: *bytes, writable: *writable }
                }
                CmdArg::Uniform(data) => RawArg::Uniform(data),
            })
            .collect();
        match indirect {
            Some((buffer, offset)) => ctx.launch_indirect(&self.kernel, buffer, offset, self.size, self.workgroups, &args),
            None => ctx.launch(&self.kernel, self.workgroups, self.size, &args),
        }
    }
}

/// What a recording has captured so far, and what keeps it valid.
#[derive(Debug)]
pub(crate) struct Recording {
    /// `(from, to)`: `to` must hold the contents of `from` before any captured command runs.
    pub seed: Vec<(Arc<Storage>, Arc<Storage>)>,
    pub cmds: Vec<Cmd>,
    /// Keeps every segment pinned during the recording resident (and alive).
    pub pins: Vec<Pin>,
    /// Buffers released during the recording; they stay out of the pool while captured commands
    /// still refer to them. Ending the recording merges the transient ones (see `transient`).
    pub retired: Vec<Buffer>,
    /// The buffers (by id) that `alloc_buffer` handed out during the recording. Their contents
    /// were arbitrary when the recording began, so nothing the plan computes can depend on them
    /// before a captured command writes them.
    fresh: FxSet<u64>,
    /// The transient buffers the commands name after the merge: those of `retired` that were
    /// allocated and released inside the recording. They belong to the plan alone (nothing
    /// outside holds them, and no buffer swap of a stepper may involve one).
    pub transient: Vec<Buffer>,
    /// Keeps the bytes of `transient` counted as resident (the segments that owned them are gone)
    /// until the plan, or the recording if it is dropped instead, is.
    pub resident: Option<Accounted>,
    kernels: FxMap<u64, Arc<KernelCore>>,
}

/// The state of this thread's recording on a context; a thread without an entry has none.
#[derive(Debug)]
enum RecState {
    Active(Recording),
    /// The recording was aborted; `end_recording` removes it.
    Aborted,
}

/// The numbers that tell contexts apart in [`RECORDINGS`] (the address of a dropped context can come back).
static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A number no other context of the process has.
pub(super) fn next_context_id() -> u64 {
    NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    /// The recordings of the current thread by context id (at most one per context).
    static RECORDINGS: RefCell<Vec<(u64, RecState)>> = const { RefCell::new(Vec::new()) };
}

impl Context {
    /// Runs `f` on this thread's entry for this context (if it has one) in [`RECORDINGS`].
    fn with_entry<R>(&self, f: impl FnOnce(&mut RecState) -> R) -> Option<R> {
        let id = self.inner.id;
        // A thread that is being torn down has no recordings left to look at.
        RECORDINGS
            .try_with(|all| {
                let mut all = all.borrow_mut();
                if all.is_empty() {
                    return None;
                }
                all.iter_mut().find(|(known, _)| *known == id).map(|(_, state)| f(state))
            })
            .ok()
            .flatten()
    }

    /// Starts capturing this thread's launches. `false` (nothing started) if this thread has a
    /// recording (active or aborted) on the context already. Other threads' recordings do not
    /// matter.
    pub(crate) fn begin_recording(&self, seed: Vec<(Arc<Storage>, Arc<Storage>)>) -> bool {
        let id = self.inner.id;
        RECORDINGS
            .try_with(|all| {
                let mut all = all.borrow_mut();
                if all.iter().any(|(known, _)| *known == id) {
                    return false;
                }
                all.push((
                    id,
                    RecState::Active(Recording {
                        seed,
                        cmds: Vec::new(),
                        pins: Vec::new(),
                        retired: Vec::new(),
                        fresh: FxSet::default(),
                        transient: Vec::new(),
                        resident: None,
                        kernels: FxMap::default(),
                    }),
                ));
                true
            })
            .unwrap_or(false)
    }

    /// Runs `f` on the recording if one is active on the calling thread.
    fn with_recording<R>(&self, f: impl FnOnce(&mut Recording) -> R) -> Option<R> {
        self.with_entry(|state| match state {
            RecState::Active(recording) => Some(f(recording)),
            RecState::Aborted => None,
        })
        .flatten()
    }

    /// Whether this thread has an active recording.
    pub(crate) fn recording_here(&self) -> bool {
        self.with_recording(|_| ()).is_some()
    }

    /// Commands captured so far by this thread's recording (0 if none).
    pub(crate) fn recorded(&self) -> usize {
        self.with_recording(|recording| recording.cmds.len()).unwrap_or(0)
    }

    /// Ends this thread's recording and returns it, or `None` if it was aborted (or there is none).
    pub(crate) fn end_recording(&self) -> Option<Recording> {
        let id = self.inner.id;
        let state = RECORDINGS
            .try_with(|all| {
                let mut all = all.borrow_mut();
                let at = all.iter().position(|(known, _)| *known == id)?;
                Some(all.swap_remove(at).1)
            })
            .ok()
            .flatten();
        match state {
            Some(RecState::Active(mut recording)) => {
                // The entry is gone, so `recycle` gives these straight to the pool.
                for buffer in recording.share_transients() {
                    self.recycle(buffer);
                }
                let kept: u64 = recording.transient.iter().map(Buffer::size).sum();
                if kept > 0 {
                    recording.resident = Some(Accounted::new(self, kept));
                }
                Some(recording)
            }
            _ => None,
        }
    }

    /// Notes that `buffer`, just taken from [`Context::alloc_buffer`], is fresh in this
    /// thread's recording (if it has one).
    pub(super) fn note_fresh(&self, buffer: &Buffer) {
        self.with_recording(|recording| {
            recording.fresh.insert(buffer.id());
        });
    }

    /// Captures a launch if this thread is recording.
    pub(super) fn capture(&self, kernel: &KernelCore, workgroups: [u32; 3], size: [u32; 3], args: &[RawArg<'_>]) -> bool {
        self.with_recording(|recording| {
            let kernel = Arc::clone(
                recording.kernels.entry(kernel.id).or_insert_with(|| Arc::new(kernel.clone())),
            );
            let args = args
                .iter()
                .map(|arg| match *arg {
                    RawArg::Storage { buffer, bytes, writable } => {
                        CmdArg::Storage { buffer: buffer.clone(), bytes, writable }
                    }
                    RawArg::Uniform(data) => CmdArg::Uniform(data.into()),
                })
                .collect();
            recording.cmds.push(Cmd { kernel, workgroups, size, args });
        })
        .is_some()
    }

    /// Holds a buffer back from the pool while a recording refers to it; gives it back if this
    /// thread is not recording.
    pub(super) fn retire(&self, buffer: Buffer) -> Option<Buffer> {
        let mut buffer = Some(buffer);
        self.with_recording(|recording| recording.retire_buffer(&mut buffer));
        buffer
    }

    /// Keeps `pin`'s segment resident until the recording is dropped.
    pub(crate) fn keep_pin(&self, pin: &Pin) {
        self.with_recording(|recording| recording.pins.push(pin.dup()));
    }

    /// Ends the recording of this thread, if any, as if it had never been held back: the seeds
    /// are copied and the captured commands encoded, in order.
    pub(crate) fn abort_recording(&self) -> Result<()> {
        // Taken out (and left aborted) before anything is encoded: the commands must not be captured again.
        let Some(recording) = self.with_entry(|state| match std::mem::replace(state, RecState::Aborted) {
            RecState::Active(recording) => Some(recording),
            RecState::Aborted => None,
        })
        .flatten() else {
            return Ok(());
        };
        let result = (|| {
            for (from, to) in &recording.seed {
                self.copy_storage(from, to)?;
            }
            for cmd in &recording.cmds {
                cmd.encode(self, &[], None)?;
            }
            Ok(())
        })();
        for buffer in recording.retired {
            self.recycle(buffer);
        }
        drop(recording.pins);
        result
    }

    /// Records copying the contents of `from` into `to` (single-segment storages of equal size).
    pub(crate) fn copy_storage(&self, from: &Storage, to: &Storage) -> Result<()> {
        let source = from.pin_quiet(0)?;
        let target = to.pin_quiet(0)?;
        self.copy(source.buffer(), 0, target.buffer(), 0, from.byte_len())
    }
}

/// The first and last command of a live interval (both inclusive).
type Interval = (usize, usize);

/// Assigns each interval `[start, end]` (both ends inclusive) a slot so that intervals sharing
/// a slot never overlap: an interval takes the first slot whose last interval ended strictly
/// before its start, or a new one. Visiting the intervals by start gives the fewest slots (the
/// greatest number of intervals that are live at once). Returns the slot of every interval;
/// slots are numbered in the order they are opened.
pub(super) fn pack_intervals(intervals: &[Interval]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..intervals.len()).collect();
    order.sort_by_key(|&i| (intervals[i].0, i));
    let mut slot_end: Vec<usize> = Vec::new();
    let mut slots = vec![0; intervals.len()];
    for i in order {
        let (start, end) = intervals[i];
        match slot_end.iter().position(|&last| last < start) {
            Some(slot) => {
                slot_end[slot] = end;
                slots[i] = slot;
            }
            None => {
                slots[i] = slot_end.len();
                slot_end.push(end);
            }
        }
    }
    slots
}

impl Recording {
    /// Lets the transient buffers of the recording share memory, and returns the ones the
    /// commands no longer name, for the caller to give back to the pool.
    ///
    /// A buffer is *transient* if it is fresh (allocated during the recording, so it held
    /// nothing that mattered when the recording began) and nothing outside the plan holds it any
    /// more: it was released during the recording (`retired`), or it belongs to a segment that
    /// only this recording's pins still hold (an array of the traced step that was dropped, see
    /// [`take_dead_buffers`]; the pins keep a segment alive, so this is where the intermediates
    /// of a step are). State, probes, seeds, results, buffers that existed before the recording
    /// and buffers still owned at its end keep their identity.
    ///
    /// The live interval of a transient buffer runs from the first captured command that binds
    /// it to the last one, whatever the binding. A replay runs the commands in recorded order on
    /// one queue, so two transient buffers of the same size and usage whose intervals do not
    /// overlap can be one buffer: the second is only written after the last use of the first, as
    /// when the pool hands out a released buffer again. The commands are rewritten to name that
    /// one buffer (the logical `bytes` they bind stay), so graph capture and every replay see
    /// the same aliasing. Intervals that touch at one command stay apart: a command may read one
    /// buffer while it writes another.
    fn share_transients(&mut self) -> Vec<Buffer> {
        let mut transient: Vec<Buffer> = Vec::new();
        let mut kept = Vec::with_capacity(self.retired.len());
        for buffer in std::mem::take(&mut self.retired) {
            match self.fresh.contains(&buffer.id()) {
                true => transient.push(buffer),
                false => kept.push(buffer),
            }
        }
        self.retired = kept;
        transient.extend(take_dead_buffers(&mut self.pins, &self.fresh));
        if transient.is_empty() {
            return Vec::new();
        }

        // Live interval (first and last command that binds it) of each; `None`: never bound.
        let mut intervals: FxMap<u64, Option<Interval>> = transient.iter().map(|buffer| (buffer.id(), None)).collect();
        for (at, cmd) in self.cmds.iter().enumerate() {
            for arg in &cmd.args {
                if let CmdArg::Storage { buffer, .. } = arg
                    && let Some(interval) = intervals.get_mut(&buffer.id())
                {
                    let (first, _) = interval.unwrap_or((at, at));
                    *interval = Some((first, at));
                }
            }
        }

        // Group by what the pool keys on.
        let mut groups: FxMap<(u64, BufferUsage), Vec<(usize, Interval)>> = FxMap::default();
        for (n, buffer) in transient.iter().enumerate() {
            if let Some(Some(interval)) = intervals.get(&buffer.id()) {
                groups.entry((buffer.size(), buffer.usage())).or_default().push((n, *interval));
            }
        }
        // The buffer each one becomes (itself, for the first of its slot); `None`: no command names it.
        let mut becomes: Vec<Option<usize>> = vec![None; transient.len()];
        for members in groups.values() {
            let slots = pack_intervals(&members.iter().map(|&(_, interval)| interval).collect::<Vec<_>>());
            let mut owner: Vec<Option<usize>> = vec![None; slots.iter().max().map_or(0, |last| last + 1)];
            for (&(n, _), &slot) in members.iter().zip(&slots) {
                becomes[n] = Some(*owner[slot].get_or_insert(n));
            }
        }
        let merged: FxMap<u64, Buffer> = becomes
            .iter()
            .enumerate()
            .filter_map(|(n, to)| to.filter(|&to| to != n).map(|to| (transient[n].id(), transient[to].clone())))
            .collect();
        if !merged.is_empty() {
            for cmd in &mut self.cmds {
                for arg in &mut cmd.args {
                    if let CmdArg::Storage { buffer, .. } = arg
                        && let Some(to) = merged.get(&buffer.id())
                    {
                        *buffer = to.clone();
                    }
                }
            }
        }

        // The plan keeps the buffers that are named; the rest is free.
        let mut freed = Vec::new();
        for (n, buffer) in transient.into_iter().enumerate() {
            if becomes[n] == Some(n) {
                self.transient.push(buffer.clone());
                self.retired.push(buffer);
            } else {
                freed.push(buffer);
            }
        }
        freed
    }

    fn retire_buffer(&mut self, buffer: &mut Option<Buffer>) {
        if let Some(buffer) = buffer.take() {
            self.retired.push(buffer);
        }
    }
}
