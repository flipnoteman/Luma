//! Recording launches: while a recording is active on a thread, `Context::launch` appends the
//! launch to a list instead of encoding it, so that a traced piece of code (one step of a loop)
//! can be replayed many times (see `jit::replay`). Anything that is not a plain launch and needs
//! the GPU state to be real (copies, uploads, readbacks, syncs, indirect launches) first aborts
//! the recording: the captured commands are then encoded in order, as if they had never been held
//! back, and the traced code carries on running normally.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::{self, ThreadId};

use wgpu::Buffer;

use crate::array::Storage;
use crate::error::Result;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;
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
    owner: ThreadId,
    /// `(from, to)`: `to` must hold the contents of `from` before any captured command runs.
    pub seed: Vec<(Arc<Storage>, Arc<Storage>)>,
    pub cmds: Vec<Cmd>,
    /// Keeps every segment pinned during the recording resident (and alive).
    pub pins: Vec<Pin>,
    /// Buffers released during the recording; they stay out of the pool while captured commands
    /// still refer to them.
    pub retired: Vec<Buffer>,
    /// [`Storage::next_id`] when the recording began: older storages are not the recording's.
    first_storage: u64,
    kernels: FxMap<u64, Arc<KernelCore>>,
}

#[derive(Debug, Default)]
pub(super) enum RecState {
    #[default]
    Idle,
    Active(Recording),
    /// The recording was aborted; the owner's `end_recording` returns to `Idle`.
    Aborted,
}

impl Context {
    /// Starts capturing this thread's launches. `false` (nothing started) unless no recording is
    /// active or aborted on the context.
    pub(crate) fn begin_recording(&self, seed: Vec<(Arc<Storage>, Arc<Storage>)>) -> bool {
        let mut state = lock(&self.inner.recording);
        if !matches!(*state, RecState::Idle) {
            return false;
        }
        *state = RecState::Active(Recording {
            owner: thread::current().id(),
            seed,
            cmds: Vec::new(),
            first_storage: Storage::next_id(),
            pins: Vec::new(),
            retired: Vec::new(),
            kernels: FxMap::default(),
        });
        self.inner.recording_active.store(true, Ordering::Release);
        true
    }

    /// Runs `f` on the recording if one is active on the calling thread.
    fn with_recording<R>(&self, f: impl FnOnce(&mut Recording) -> R) -> Option<R> {
        if !self.inner.recording_active.load(Ordering::Relaxed) {
            return None;
        }
        match &mut *lock(&self.inner.recording) {
            RecState::Active(recording) if recording.owner == thread::current().id() => Some(f(recording)),
            _ => None,
        }
    }

    /// Whether this thread has an active recording.
    pub(crate) fn recording_here(&self) -> bool {
        self.with_recording(|_| ()).is_some()
    }

    /// Commands captured so far by this thread's recording (0 if none).
    pub(crate) fn recorded(&self) -> usize {
        self.with_recording(|recording| recording.cmds.len()).unwrap_or(0)
    }

    /// Ends this thread's recording and returns it, or `None` if it was aborted.
    pub(crate) fn end_recording(&self) -> Option<Recording> {
        let mut state = lock(&self.inner.recording);
        match &*state {
            RecState::Active(recording) if recording.owner != thread::current().id() => return None,
            RecState::Idle => return None,
            _ => {}
        }
        self.inner.recording_active.store(false, Ordering::Release);
        match std::mem::take(&mut *state) {
            RecState::Active(recording) => Some(recording),
            _ => None,
        }
    }

    /// Whether a launch may overwrite `storage` in place: always, unless this thread is recording
    /// and the storage is older than the recording. A recorded launch runs again at every replay,
    /// so it may only write buffers the recording itself made (they are rewritten from scratch
    /// each time); an older buffer, such as the operand of an expression created before the
    /// loop, would be overwritten once per replayed step.
    pub(crate) fn may_overwrite(&self, storage: &Storage) -> bool {
        self.with_recording(|recording| storage.id() >= recording.first_storage).unwrap_or(true)
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
        if !self.inner.recording_active.load(Ordering::Relaxed) {
            return Ok(());
        }
        let recording = {
            let mut state = lock(&self.inner.recording);
            match &*state {
                RecState::Active(recording) if recording.owner == thread::current().id() => {}
                _ => return Ok(()),
            }
            self.inner.recording_active.store(false, Ordering::Release);
            match std::mem::replace(&mut *state, RecState::Aborted) {
                RecState::Active(recording) => recording,
                _ => return Ok(()),
            }
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
