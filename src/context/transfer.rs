//! Uploads, readbacks and the waits for them.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use bytemuck::Pod;
use luma_core::hal::{advise_huge_pages, HostBytes};
use luma_core::{BackendError, Buffer};

use crate::array::Storage;
use crate::error::{LumaError, Result};
use luma_core::hal::size_class;

use super::Context;

/// Fewest time a wait polls before it blocks in the driver, whatever the recent waits were. A
/// thread blocked in the driver is woken late: 20-40 us on a desktop RTX 3090, 65 us on a laptop
/// RTX 3080 (blocking at 500 us rather than polling to the end of a 733 us kernel returned the
/// laptop's `eval` 130 us later). Burning a core for this long costs less than that on every
/// job and is what this crate always did.
pub(super) const SYNC_SPIN_MIN: Duration = Duration::from_micros(500);

/// Most a wait polls before it blocks in the driver, however long the recent waits were. It
/// bounds the CPU time a wait burns on a job that runs much longer than predicted.
const SYNC_SPIN_MAX: Duration = Duration::from_millis(4);

/// Wait durations the spin budget is derived from, see [`WaitHistory`].
pub(super) const WAIT_SAMPLES: usize = 4;

/// The durations of the last [`WAIT_SAMPLES`] waits of a context ([`Context::sync`] and the
/// readbacks), which decide how long the next one polls for the GPU before it blocks.
///
/// A blocked thread is woken late, so a wait that is polled to its end returns sooner than one
/// that blocks, but polling burns a core. The budget is one and a half times the longest of the
/// last [`WAIT_SAMPLES`] waits, at least [`SYNC_SPIN_MIN`]; when that exceeds [`SYNC_SPIN_MAX`]
/// it is [`SYNC_SPIN_MIN`] instead, because polling would not catch a job that long anyway and
/// the wake-up delay is a small part of it. So a loop of evaluations that take up to
/// 2.7 ms ([`SYNC_SPIN_MAX`] / 1.5) never blocks, a loop of longer ones polls for
/// [`SYNC_SPIN_MIN`] per wait and then blocks, a wait that turns out much longer than predicted
/// costs at most [`SYNC_SPIN_MAX`] of polling (normally at most 1.5 times the prediction), and a
/// context starts with [`SYNC_SPIN_MIN`]. The newest sample replaces the oldest, so one outlier
/// (a long job among short ones) is forgotten after [`WAIT_SAMPLES`] waits.
///
/// Racy by design: waits from several threads overwrite each other's samples, which only makes
/// the prediction a little worse.
///
/// Measured (median of 300 `eval`s of Black-Scholes, wait phase after the submission, against
/// the fixed 500 us poll; RTX 3090 / laptop RTX 3080; GPU kernel in brackets):
/// - 1<<22 options: 95 us (78) / 205 us (186), unchanged: the kernel ends inside 500 us either way.
/// - 1<<24: 319 us (302) / 910-965 us (733) before, 318 us / 789 us after. The laptop's kernel
///   outlasts 500 us, so the old poll blocked and the wake-up cost 135 us.
/// - 1<<26: 1235 us (1198) before, 1214 us after / 3220 us (2919), unchanged: a 3 ms wait is
///   beyond [`SYNC_SPIN_MAX`] / 1.5.
/// - Loops of Black-Scholes kernels of 1<<24 per `eval`: on the laptop 2 kernels take 1.83 ms
///   before and 1.62 ms after, at 95% CPU instead of 38%; on the desktop 5 kernels take 1.62 ms
///   before and 1.59 ms after, at 100% instead of 40%. Chains of 4-15 ms (desktop
///   20 and 40 kernels, laptop 5, 10 and 20) and of 100 ms use the same CPU time as before
///   (8-21% of the wall time, 15% for 100 ms on the laptop; 11-15% and 43% on the desktop,
///   where recording the 320 kernels of the long chain is most of it).
/// - Yielding the thread in the poll loop gives the same wait (789 us / 318 us), so it buys
///   nothing and is not done.
///
/// What is left after the kernel, 17 us on the desktop and 20 us on the laptop (56 us on the
/// laptop once the kernel moves more than about 170 MB or runs longer than about 450 us), is
/// not on the host: a poll takes 0.14 us, the one that sees the end under 2 us.
#[derive(Debug)]
pub(super) struct WaitHistory {
    /// Microseconds, saturating.
    samples: [AtomicU32; WAIT_SAMPLES],
    next: AtomicU32,
}

impl WaitHistory {
    pub(super) fn new() -> Self {
        WaitHistory { samples: std::array::from_fn(|_| AtomicU32::new(0)), next: AtomicU32::new(0) }
    }

    /// How long the next wait polls before it blocks.
    pub(super) fn budget(&self) -> Duration {
        let longest = self.samples.iter().map(|s| s.load(Ordering::Relaxed)).max().unwrap_or(0);
        let wanted = Duration::from_micros(3 * u64::from(longest) / 2);
        if wanted > SYNC_SPIN_MAX { SYNC_SPIN_MIN } else { wanted.max(SYNC_SPIN_MIN) }
    }

    /// Notes how long a wait took, polling and blocking together.
    pub(super) fn record(&self, wait: Duration) {
        let at = self.next.fetch_add(1, Ordering::Relaxed) as usize % WAIT_SAMPLES;
        let micros = u32::try_from(wait.as_micros()).unwrap_or(u32::MAX);
        self.samples[at].store(micros, Ordering::Relaxed);
    }
}

impl Context {
    /// Records a copy of `data` into the start of `dst`, ordered after everything recorded so
    /// far. `data.len()` must be a multiple of 4.
    ///
    /// The bytes go through a mappable staging buffer of the backend. Those are reused: once the
    /// batch that copies out of one is submitted it is mapped again, and the next upload of its
    /// size class writes straight into it.
    pub(crate) fn upload(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        self.abort_recording_for("upload")?;
        let bytes = data.len() as u64;
        let max = self.max_buffer_bytes();
        size_class(bytes, max).ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        self.relieving(bytes, || match self.hal().record_upload(dst, data, max) {
            Ok(()) => Ok(Some(())),
            Err(BackendError::OutOfMemory { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        })
    }

    /// Host memory the backend uploads from without another copy (pinned memory), `bytes` bytes
    /// the caller writes completely; `None` where the backend has none. A stage of the cache
    /// [`Context::stage_release`] fills is reused when the backend's own (not a heap stage) has the size.
    pub(crate) fn upload_stage(&self, bytes: u64) -> Option<luma_core::hal::Stage> {
        if !self.hal().pinned_staging() {
            return None;
        }
        {
            let mut cache = self.inner.stages.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = luma_core::hal::StageDir::ToDevice;
            if let Some(at) = cache.iter().position(|stage| stage.dir() == dir && stage.len() == bytes && !stage.is_heap()) {
                return Some(cache.swap_remove(at));
            }
        }
        self.hal().stage_alloc_uninit(bytes, luma_core::hal::StageDir::ToDevice).ok()
    }

    /// [`Context::upload`] of the first `bytes` bytes of a stage from [`Context::upload_stage`],
    /// asynchronous on the backend's transfer lane: the ticket completes when the copy has run
    /// and the stage may be written again (a stage the backend kept is empty afterwards).
    pub(crate) fn upload_staged(&self, dst: &Buffer, stage: &mut luma_core::hal::Stage, bytes: u64) -> Result<luma_core::Ticket> {
        self.abort_recording_for("upload")?;
        let max = self.max_buffer_bytes();
        size_class(bytes, max).ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        self.relieving(bytes, || match self.hal().record_upload_staged(dst, stage, bytes, max) {
            Ok(ticket) => Ok(Some(ticket)),
            Err(BackendError::OutOfMemory { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        })
    }

    /// Host memory for [`Context::readback_staged`], `bytes` bytes (pinned where the backend has
    /// any); `None` where the backend has no staging memory of its own.
    pub(crate) fn readback_stage(&self, bytes: u64) -> Option<luma_core::hal::Stage> {
        if !self.hal().pinned_staging() {
            return None;
        }
        self.hal().stage_alloc_uninit(bytes, luma_core::hal::StageDir::FromDevice).ok()
    }

    /// Records a copy of the first `bytes` bytes of `src` into a stage from [`Context::readback_stage`],
    /// ordered after everything recorded so far and asynchronous on the backend's transfer lane:
    /// the ticket completes when the bytes are in the stage. The caller keeps `src` unwritten
    /// until then.
    pub(crate) fn readback_staged(&self, src: &Buffer, stage: &mut luma_core::hal::Stage, bytes: u64) -> Result<luma_core::Ticket> {
        self.abort_recording_for("readback")?;
        self.notify_released();
        Ok(self.hal().record_readback_staged(src, stage, bytes)?)
    }

    /// Whether the copy `ticket` stands for has completed (a failed query counts as done: the
    /// waiting paths report the error).
    pub(crate) fn ticket_done(&self, ticket: &luma_core::Ticket) -> bool {
        self.hal().ticket_done(ticket).unwrap_or(true)
    }

    /// Whether `from_slice` writes straight into mappable storage buffers (unified memory)
    /// instead of copying out of a staging buffer.
    pub(crate) fn zero_copy(&self) -> bool {
        self.inner.zero_copy
    }

    /// A storage buffer holding `data` at its start, written by the host without a GPU copy.
    /// Only with [`Context::zero_copy`]; give it back with [`Context::recycle`]. `data.len()`
    /// must be a multiple of 4.
    pub(crate) fn alloc_filled(&self, data: &[u8]) -> Result<Buffer> {
        let bytes = data.len() as u64;
        let max = self.max_buffer_bytes();
        let class = size_class(bytes, max).ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        #[cfg(test)]
        self.inner.created_bytes.fetch_add(class, std::sync::atomic::Ordering::Relaxed);
        self.relieving(class, || match self.hal().alloc_filled(data, max) {
            Ok(buffer) => Ok(Some(buffer)),
            Err(BackendError::OutOfMemory { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        })
    }

    /// Host pointer to the first `bytes` of `buffer` when the host may touch it now, see [`Backend::host_ptr`].
    pub(crate) fn host_ptr(&self, buffer: &Buffer, bytes: u64) -> Option<*mut u8> {
        self.hal().host_ptr(buffer, bytes)
    }

    /// An uninitialized mappable storage buffer of at least `bytes` bytes, like [`Context::alloc_filled`]
    /// without the data. Only with [`Context::zero_copy`].
    pub(crate) fn alloc_host_buffer(&self, bytes: u64) -> Result<Buffer> {
        let max = self.max_buffer_bytes();
        let class = size_class(bytes, max).ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        #[cfg(test)]
        self.inner.created_bytes.fetch_add(class, std::sync::atomic::Ordering::Relaxed);
        let buffer = self.relieving(class, || match self.hal().alloc_host_buffer(bytes, max) {
            Ok(buffer) => Ok(Some(buffer)),
            Err(BackendError::OutOfMemory { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        })?;
        if buffer.size() == bytes
            && let Some(ptr) = self.host_ptr(&buffer, bytes)
        {
            // SAFETY: the buffer is fresh and idle, so no one else touches its `bytes` bytes.
            advise_huge_pages(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<std::mem::MaybeUninit<u8>>(), bytes as usize) });
        }
        Ok(buffer)
    }

    /// Copies the first `bytes` bytes of `buffer` into a new vector, see [`Context::read_into`].
    pub(crate) fn read<T: Pod>(&self, buffer: &Buffer, bytes: u64) -> Result<Vec<T>> {
        self.read_vec(bytes, |dst| self.read_into(buffer, dst))
    }

    /// Copies the data of `storage` into a new vector, segment by segment.
    pub(crate) fn read_storage<T: Pod>(&self, storage: &Storage) -> Result<Vec<T>> {
        self.read_vec(storage.byte_len(), |dst| self.read_storage_into(storage, dst))
    }

    /// Copies the data of `storage` into `dst` (exactly `storage.byte_len()` bytes), after
    /// everything recorded so far; on success every byte of `dst` is written.
    pub(crate) fn read_storage_into(&self, storage: &Storage, dst: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let mut rest = dst;
        for j in 0..storage.segments() {
            let (piece, tail) = rest.split_at_mut(storage.segment_byte_len(j) as usize);
            storage.segment(j).read_into(piece)?;
            rest = tail;
        }
        Ok(())
    }

    /// A vector of `bytes / size_of::<T>()` elements, in memory advised to use huge pages, that
    /// `fill` writes completely.
    fn read_vec<T: Pod>(
        &self,
        bytes: u64,
        fill: impl FnOnce(&mut [std::mem::MaybeUninit<u8>]) -> Result<()>,
    ) -> Result<Vec<T>> {
        let len = usize::try_from(bytes)
            .map_err(|_| LumaError::InvalidShape(format!("{bytes} bytes do not fit in host memory")))?
            / std::mem::size_of::<T>();
        let mut out: Vec<T> = Vec::with_capacity(len);
        let spare = out.spare_capacity_mut();
        // SAFETY: a `MaybeUninit<T>` slice viewed as the same bytes; `T: Pod` has no padding or
        // invalid bit patterns, so any initialized bytes make valid elements.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(), len * std::mem::size_of::<T>())
        };
        advise_huge_pages(dst);
        fill(dst)?;
        // SAFETY: `fill` wrote all `len * size_of::<T>()` bytes.
        unsafe { out.set_len(len) };
        Ok(out)
    }

    /// Copies the first `dst.len()` bytes of `buffer` into `dst`, after everything recorded so
    /// far; on success every byte of `dst` is written. The backend reads large buffers in pieces,
    /// each its own submission, so copying one piece out of its staging buffer overlaps the
    /// transfer of the next.
    pub(crate) fn read_into(&self, buffer: &Buffer, dst: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        self.abort_recording_for("readback")?;
        self.notify_released();
        let waited = self.hal().read_into(buffer, dst, false, self.inner.waits.budget())?;
        self.inner.waits.record(waited);
        Ok(())
    }

    /// Reads the first `bytes` bytes of `buffer` like [`Context::read`], but through one staging
    /// buffer of a bounded size and without noting the waits in [`WaitHistory`].
    /// For readbacks made while the device is full.
    pub(crate) fn read_spill(&self, buffer: &Buffer, bytes: u64) -> Result<HostBytes> {
        self.abort_recording_for("readback")?;
        let len = usize::try_from(bytes).map_err(|_| LumaError::InvalidShape(format!("{bytes} bytes do not fit in host memory")))?;
        HostBytes::filled_by::<LumaError>(len, |dst| {
            self.notify_released();
            self.hal().read_into(buffer, dst, true, self.inner.waits.budget())?;
            Ok(())
        })
    }
}

/// The transfer lanes of the backend (W1-P2b), as a schedule's executor uses them: ring buffers, staging memory, copies that run
/// beside the compute lane, and the tickets that order them.
/// Bytes of finished transfer memory a context keeps.
const STAGE_CACHE_BYTES: u64 = 1 << 30;

impl Context {
    /// A device buffer of at least `bytes` bytes that a transfer may copy into or out of (not pooled: it is freed when dropped, after
    /// every ticket that touches it has completed).
    pub(crate) fn lane_buffer(&self, bytes: u64) -> Result<Buffer> {
        let max = self.max_buffer_bytes();
        let class = size_class(bytes, max).ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        self.relieving(class, || Ok(self.hal().create_buffer_unordered(class, crate::context::STORAGE_USAGE)))
    }

    /// Host memory for transfers in direction `dir`.
    /// A stage given back by [`Context::stage_release`] is reused when it has the right direction and size.
    pub(crate) fn stage_alloc(&self, bytes: u64, dir: luma_core::hal::StageDir) -> Result<luma_core::hal::Stage> {
        {
            let mut cache = self.inner.stages.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(at) = cache.iter().position(|stage| stage.dir() == dir && stage.len() == bytes) {
                return Ok(cache.swap_remove(at));
            }
        }
        Ok(self.hal().stage_alloc(bytes, dir)?)
    }

    /// Keeps the host memory of a transfer whose ticket has completed for the next [`Context::stage_alloc`]: at most
    /// [`STAGE_CACHE_BYTES`] bytes are kept, the rest is freed.
    pub(crate) fn stage_release(&self, stage: luma_core::hal::Stage) {
        let mut cache = self.inner.stages.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let held: u64 = cache.iter().map(luma_core::hal::Stage::len).sum();
        if held + stage.len() <= STAGE_CACHE_BYTES {
            cache.push(stage);
        }
    }

    /// [`luma_core::hal::Backend::transfer_in`].
    pub(crate) fn transfer_in(&self, stage: &luma_core::hal::Stage, dst: &Buffer, bytes: u64, after: &[luma_core::hal::Ticket]) -> Result<luma_core::hal::Ticket> {
        self.abort_recording_for("transfer")?;
        Ok(self.hal().transfer_in(stage, dst, 0, bytes, after)?)
    }

    /// [`luma_core::hal::Backend::transfer_out`].
    pub(crate) fn transfer_out(&self, src: &Buffer, stage: &mut luma_core::hal::Stage, bytes: u64, after: &[luma_core::hal::Ticket]) -> Result<luma_core::hal::Ticket> {
        self.abort_recording_for("transfer")?;
        Ok(self.hal().transfer_out(src, 0, stage, bytes, after)?)
    }

    /// [`luma_core::hal::Backend::barrier`].
    pub(crate) fn lane_barrier(&self, after: &[luma_core::hal::Ticket]) -> Result<luma_core::hal::Ticket> {
        Ok(self.hal().barrier(after)?)
    }

    /// Waits for `ticket`.
    pub(crate) fn ticket_wait(&self, ticket: &luma_core::hal::Ticket) -> Result<()> {
        Ok(self.hal().ticket_wait(ticket, SYNC_SPIN_MIN)?)
    }
}
