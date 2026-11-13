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

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use wgpu::Buffer;

use crate::array::Storage;
use crate::error::Result;
use crate::gpu::hash::FxMap;
use crate::gpu::pager::Pin;
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
            Some((buffer, offset)) => ctx.launch_indirect(&self.kernel, buffer, offset, self.size, &args),
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
    /// still refer to them.
    pub retired: Vec<Buffer>,
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
            Some(RecState::Active(recording)) => Some(recording),
            _ => None,
        }
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

impl Recording {
    fn retire_buffer(&mut self, buffer: &mut Option<Buffer>) {
        if let Some(buffer) = buffer.take() {
            self.retired.push(buffer);
        }
    }
}
