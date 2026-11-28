//! Types of the transfer lane of [`Backend`](super::Backend): host staging memory and the tickets
//! of asynchronous copies.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::hostbytes::HostBytes;

/// Which way the bytes of a [`Stage`] travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageDir {
    /// The host writes the stage, the device copies from it (`Backend::transfer_in`).
    ToDevice,
    /// The device copies into the stage, the host reads it (`Backend::transfer_out`).
    FromDevice,
}

/// Host memory a backend implements a [`Stage`] with.
pub trait NativeStage: Any + fmt::Debug + Send + Sync + 'static {
    /// The staged bytes.
    fn bytes(&self) -> &[u8];
    /// The staged bytes, for writing.
    fn bytes_mut(&mut self) -> &mut [u8];
}

/// The default representation: plain heap memory.
#[derive(Debug)]
struct HeapStage(HostBytes);

impl NativeStage for HeapStage {
    fn bytes(&self) -> &[u8] {
        self.0.as_slice()
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_mut_slice()
    }
}

/// Backend-owned host memory the device copies from or to without another host copy (pinned
/// memory on CUDA). Allocated with `Backend::stage_alloc`; its bytes are zero when allocated.
///
/// The host owns the bytes except between a transfer's start and the completion of its
/// [`Ticket`]: do not touch them then.
#[derive(Debug)]
pub struct Stage {
    dir: StageDir,
    native: Box<dyn NativeStage>,
}

impl Stage {
    /// A stage on plain heap memory, the default of `Backend::stage_alloc`.
    pub fn heap(bytes: u64, dir: StageDir) -> Stage {
        Stage { dir, native: Box::new(HeapStage(HostBytes::zeroed(bytes as usize))) }
    }

    /// A stage on a backend's own memory.
    pub fn new(dir: StageDir, native: impl NativeStage) -> Stage {
        Stage { dir, native: Box::new(native) }
    }

    /// The direction the stage was allocated for.
    pub fn dir(&self) -> StageDir {
        self.dir
    }

    /// The size in bytes.
    pub fn len(&self) -> u64 {
        self.native.bytes().len() as u64
    }

    /// Whether the stage holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.native.bytes().is_empty()
    }

    /// The bytes.
    pub fn as_slice(&self) -> &[u8] {
        self.native.bytes()
    }

    /// The bytes, for writing.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.native.bytes_mut()
    }

    /// Whether the stage is plain heap memory (the default of `Backend::stage_alloc`).
    pub fn is_heap(&self) -> bool {
        let native: &dyn Any = &*self.native;
        native.is::<HeapStage>()
    }

    /// The backend's own representation, `None` for another backend's (or the heap default).
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        let native: &dyn Any = &*self.native;
        native.downcast_ref()
    }

    /// [`Stage::downcast_ref`] with write access.
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let native: &mut dyn Any = &mut *self.native;
        native.downcast_mut()
    }
}

/// What a ticket of the default implementations stands for.
#[derive(Debug)]
enum Marker {
    /// Everything the ticket stands for has completed.
    Done,
    /// The work submitted so far (the default ticket of `transfer_in` and `barrier`).
    Flushed,
}

/// The completion of an asynchronous transfer or of the compute work up to a barrier. Cheap to
/// clone; opaque to everyone but the backend that made it.
#[derive(Clone)]
pub struct Ticket(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for Ticket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Ticket")
    }
}

impl Ticket {
    /// A ticket of a backend's own kind.
    pub fn new(native: impl Any + Send + Sync) -> Ticket {
        Ticket(Arc::new(native))
    }

    /// A ticket whose work has completed.
    pub fn done() -> Ticket {
        Ticket::new(Marker::Done)
    }

    /// A ticket for everything submitted so far, completed once the device is idle.
    pub fn flushed() -> Ticket {
        Ticket::new(Marker::Flushed)
    }

    /// The backend's own representation.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }

    /// Whether this is [`Ticket::done`].
    pub fn is_done_marker(&self) -> bool {
        matches!(self.0.downcast_ref::<Marker>(), Some(Marker::Done))
    }

    /// Whether this is [`Ticket::flushed`].
    pub fn is_flushed_marker(&self) -> bool {
        matches!(self.0.downcast_ref::<Marker>(), Some(Marker::Flushed))
    }
}
