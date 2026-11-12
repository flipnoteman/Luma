//! Uploads, readbacks and the waits for them.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytemuck::Pod;

use wgpu::{Buffer, BufferUsages};

use crate::error::{LumaError, Result};
use crate::array::Storage;
use crate::gpu::pool::{self};

use super::{lock, Context, MAPPABLE_STORAGE_USAGE};

/// Usage of readback buffers.
pub(super) const STAGING_USAGE: BufferUsages = BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

/// Usage of upload staging buffers.
const UPLOAD_USAGE: BufferUsages = BufferUsages::MAP_WRITE.union(BufferUsages::COPY_SRC);

/// Readbacks larger than twice this go in pieces of this size, see [`Context::read`].
const READ_CHUNK: u64 = 32 << 20;

/// Mapped upload staging buffers ready to be written.
#[derive(Debug)]
pub(super) struct FreeUploads {
    buffers: Vec<Buffer>,
    bytes: u64,
    /// Bytes of buffers in the unsubmitted batch or waiting for a `map_async` callback.
    pub(super) in_flight: u64,
    /// Most bytes of idle buffers kept for reuse; scales with the memory budget.
    cap: u64,
}

impl FreeUploads {
    pub(super) fn new(cap: u64) -> Self {
        FreeUploads { buffers: Vec::new(), bytes: 0, in_flight: 0, cap }
    }

    fn take(&mut self, size: u64) -> Option<Buffer> {
        let at = self.buffers.iter().position(|buffer| buffer.size() == size)?;
        self.bytes -= size;
        Some(self.buffers.swap_remove(at))
    }

    /// Keeps `buffer` unless that would exceed the cap; returns it otherwise.
    pub(super) fn put(&mut self, buffer: Buffer) -> Option<Buffer> {
        if self.bytes + buffer.size() <= self.cap {
            self.bytes += buffer.size();
            self.buffers.push(buffer);
            None
        } else {
            Some(buffer)
        }
    }

    pub(super) fn drain(&mut self) -> Vec<Buffer> {
        self.bytes = 0;
        std::mem::take(&mut self.buffers)
    }
}

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

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 8 << 20;

/// `dst.copy_from_slice(src)` split over several threads for large copies: one core cannot
/// saturate unified memory, and the page faults of a fresh destination spread out too. The
/// threads store past the cache (see [`copy_streaming`]).
pub(crate) fn copy_parallel(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.write_copy_of_slice(src);
        return;
    }
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (s, d) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            scope.spawn(move || copy_streaming(s, d));
        }
    });
}

/// `dst.copy_from_slice(src)` with non-temporal stores: the destination lines are not read into
/// the cache before being overwritten, which a large readback (bound by host memory bandwidth
/// while the GPU's transfer writes the next chunk) would pay for with a third of its traffic.
#[cfg(target_arch = "x86_64")]
fn copy_streaming(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    const LINE: usize = 64;
    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let head = dst.as_ptr().align_offset(LINE).min(dst.len());
    let (dst_head, dst_body) = dst.split_at_mut(head);
    let (src_head, src_body) = src.split_at(head);
    dst_head.write_copy_of_slice(src_head);
    let (src_lines, src_tail) = src_body.as_chunks::<LINE>();
    let (dst_lines, dst_tail) = dst_body.as_chunks_mut::<LINE>();
    for (s, d) in src_lines.iter().zip(dst_lines) {
        for part in (0..LINE).step_by(16) {
            // SAFETY: `part + 16 <= LINE`, so both pointers stay inside their 64-byte chunks;
            // the load is unaligned, and `d` is 64-byte aligned (`head`), as the store requires.
            // SSE2 is part of the x86_64 baseline.
            unsafe {
                let value = _mm_loadu_si128(s.as_ptr().add(part).cast::<__m128i>());
                _mm_stream_si128(d.as_mut_ptr().add(part).cast::<__m128i>(), value);
            }
        }
    }
    // SAFETY: SSE is part of the x86_64 baseline. Orders the streaming stores before whatever
    // reads the destination next, on any thread.
    unsafe { _mm_sfence() };
    dst_tail.write_copy_of_slice(src_tail);
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_streaming(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) {
    dst.write_copy_of_slice(src);
}

/// The outcome of a `map_async` request, see [`map_for_read`].
type MapReceiver = mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>;

/// Requests a read mapping of `slice`; [`Context::wait_map`] waits for it.
fn map_for_read(slice: wgpu::BufferSlice<'_>) -> MapReceiver {
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver
}

/// Backs a large readback destination with transparent huge pages: faulting in, and later
/// freeing, its 4 KiB pages one by one costs about as much as the copy itself.
#[cfg(target_os = "linux")]
fn advise_huge_pages(dst: &mut [std::mem::MaybeUninit<u8>]) {
    const PAGE: usize = 4096;
    if dst.len() < PARALLEL_COPY_MIN {
        return;
    }
    let start = (dst.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let end = (dst.as_mut_ptr() as usize + dst.len()) & !(PAGE - 1);
    // SAFETY: the page-aligned range lies within `dst`; `madvise` only changes how the kernel
    // backs it, never its contents, and failure is harmless.
    unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_: &mut [std::mem::MaybeUninit<u8>]) {}

/// [`copy_parallel`] into write-only mapped memory.
fn write_parallel(src: &[u8], mut dst: wgpu::WriteOnly<'_, [u8]>) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    if src.len() < PARALLEL_COPY_MIN || threads == 1 {
        dst.copy_from_slice(src);
        return;
    }
    /// A disjoint part of the mapping, handed to one thread.
    struct Part(std::ptr::NonNull<[u8]>);
    // SAFETY: parts are disjoint and outlive the scope the threads run in.
    unsafe impl Send for Part {}

    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    let base = dst.as_raw_ptr().cast::<u8>();
    let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (i, s) in src.chunks(chunk).enumerate() {
            // SAFETY: `i * chunk + s.len() <= dst.len()`, so the part lies inside the mapping.
            let part = Part(std::ptr::NonNull::slice_from_raw_parts(unsafe { base.add(i * chunk) }, s.len()));
            scope.spawn(move || {
                let part = part;
                // SAFETY: the part is mapped for writing and no other thread touches it.
                unsafe { wgpu::WriteOnly::new(part.0) }.copy_from_slice(s);
            });
        }
    });
}

impl Context {
    /// Records a copy of `data` into the start of `dst`, ordered after everything recorded so
    /// far. `data.len()` must be a multiple of 4.
    ///
    /// The bytes go through a mappable staging buffer. Those are reused: once the batch that
    /// copies out of one is submitted it is mapped again, and the next upload of its size class
    /// writes straight into it.
    pub(crate) fn upload(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        let bytes = data.len() as u64;
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        let staging = self.take_upload_buffer(&self.inner.uploads, class, UPLOAD_USAGE)?;
        {
            let mut view = staging.get_mapped_range_mut(..bytes).map_err(LumaError::MapRange)?;
            write_parallel(data, view.slice(..));
        }
        staging.unmap();

        let mut guard = self.batch()?;
        let encoder = self.encoder(&mut guard);
        encoder.copy_buffer_to_buffer(&staging, 0, dst, 0, bytes);
        lock(&self.inner.uploads).in_flight += staging.size();
        guard.uploads.push(staging);
        Ok(())
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
        let max = self.inner.max_buffer_bytes;
        let class = pool::size_class(bytes, max)
            .ok_or(LumaError::BufferTooLarge { requested: bytes, max })?;
        let buffer = self.take_upload_buffer(&self.inner.mappable, class, MAPPABLE_STORAGE_USAGE)?;
        {
            let mut view = buffer.get_mapped_range_mut(..bytes).map_err(LumaError::MapRange)?;
            write_parallel(data, view.slice(..));
        }
        buffer.unmap();
        Ok(buffer)
    }

    /// Takes a mapped `MAP_WRITE` buffer of `class` bytes from `pool`, polling completed
    /// submissions (and waiting if more bytes than the idle cap are in flight) before
    /// allocating a fresh one.
    fn take_upload_buffer(
        &self,
        pool: &Mutex<FreeUploads>,
        class: u64,
        usage: BufferUsages,
    ) -> Result<Buffer> {
        if let Some(buffer) = self.take_idle(pool, class) {
            return Ok(buffer);
        }
        let (in_flight, cap) = {
            let uploads = lock(pool);
            (uploads.in_flight, uploads.cap)
        };
        if in_flight > cap {
            self.flush_locked(&mut *self.batch()?);
            self.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| LumaError::Poll(Arc::new(e)))?;
            if let Some(buffer) = lock(pool).take(class) {
                return Ok(buffer);
            }
        }
        self.new_buffer(class, usage, true)
    }

    /// An idle mapped buffer of `class` bytes from `pool`. Polls completed submissions first,
    /// which hands back the buffers whose `map_async` callback is due. Never submits or waits.
    fn take_idle(&self, pool: &Mutex<FreeUploads>, class: u64) -> Option<Buffer> {
        if let Some(buffer) = lock(pool).take(class) {
            return Some(buffer);
        }
        let _ = self.device().poll(wgpu::PollType::Poll);
        lock(pool).take(class)
    }

    /// A staging buffer holding `data` at its start, unmapped and ready to be copied out of, for
    /// the uniform slots of [`Arena::upload`]. It counts as in flight: give it to
    /// [`Batch::uploads`], which maps it again once the batch that copies out of it is
    /// submitted. `None` if no staging buffer can be had (the device is out of memory).
    ///
    /// This runs while the batch is being submitted, with the batch locked, so unlike
    /// [`Context::take_upload_buffer`] it must not flush or wait: it reuses an idle buffer or
    /// creates a mapped one.
    pub(super) fn stage(&self, data: &[u8]) -> Option<Buffer> {
        let bytes = data.len() as u64;
        let staging = self.mapped_staging(bytes)?;
        {
            // A plain copy: the slots of one chunk are kilobytes, and `write_parallel` asks the
            // OS for the thread count first, which costs more than copying them.
            let mut view = staging.get_mapped_range_mut(..bytes).ok()?;
            view.slice(..).copy_from_slice(data);
        }
        staging.unmap();
        lock(&self.inner.uploads).in_flight += staging.size();
        Some(staging)
    }

    /// An upload staging buffer for `bytes` bytes, still mapped: an idle one, or a new one.
    pub(super) fn mapped_staging(&self, bytes: u64) -> Option<Buffer> {
        let class = pool::size_class(bytes, self.inner.max_buffer_bytes)?;
        match self.take_idle(&self.inner.uploads, class) {
            Some(buffer) => Some(buffer),
            None => self.create_buffer(class, UPLOAD_USAGE, true),
        }
    }

    /// Writes `data` at the start of the mapped staging buffer `staging` and unmaps it, so that
    /// the commands recorded to copy out of it can be submitted. Only [`Arena::upload`] calls
    /// this, for the buffer of a [`Prefix`], which is mapped for as long as it is not filled.
    pub(super) fn fill(&self, staging: &Buffer, data: &[u8]) {
        if !data.is_empty() {
            match staging.get_mapped_range_mut(..data.len() as u64) {
                Ok(mut view) => view.slice(..).copy_from_slice(data),
                Err(error) => {
                    lock(&self.inner.gpu_error).get_or_insert_with(|| format!("uniform staging buffer: {error}"));
                }
            }
        }
        staging.unmap();
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
        let elem = storage.elem();
        let mut rest = dst;
        for j in 0..storage.segments() {
            let (piece, tail) = rest.split_at_mut(storage.range(j).len() * elem);
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
    /// far; on success every byte of `dst` is written. Large reads go in [`READ_CHUNK`] pieces,
    /// each its own submission, so copying one piece out of its staging buffer overlaps the
    /// transfer of the next.
    pub(crate) fn read_into(&self, buffer: &Buffer, dst: &mut [std::mem::MaybeUninit<u8>]) -> Result<()> {
        let bytes = dst.len() as u64;
        let chunk = if bytes > 2 * READ_CHUNK { READ_CHUNK } else { bytes.max(1) };
        let mut staged = Vec::new();
        for offset in (0..bytes).step_by(chunk as usize) {
            let size = chunk.min(bytes - offset);
            match self.alloc_buffer(size, STAGING_USAGE) {
                Ok(staging) => staged.push((offset, size, staging)),
                Err(err) => {
                    for (_, _, s) in staged {
                        self.recycle(s);
                    }
                    return Err(err);
                }
            }
        }
        let mut maps = Vec::with_capacity(staged.len());
        match self.batch() {
            Ok(mut guard) => {
                for (offset, size, staging) in &staged {
                    self.encoder(&mut guard).copy_buffer_to_buffer(buffer, *offset, staging, 0, *size);
                    let submission = self.flush_locked(&mut guard);
                    maps.push((submission, map_for_read(staging.slice(..*size))));
                }
            }
            Err(err) => {
                for (_, _, s) in staged {
                    self.recycle(s);
                }
                return Err(err);
            }
        }

        let mut longest = Duration::ZERO;
        let mut staged_iter = staged.into_iter().zip(maps);
        while let Some(((offset, size, staging), (submission, mapped))) = staged_iter.next() {
            let step = self.wait_map(submission, &mapped).and_then(|waited| {
                longest = longest.max(waited);
                let view = staging.slice(..size).get_mapped_range().map_err(LumaError::MapRange)?;
                copy_parallel(&view, &mut dst[offset as usize..(offset + size) as usize]);
                Ok(())
            });
            staging.unmap();
            if let Err(err) = step {
                for ((_, _, rest), _) in staged_iter {
                    rest.unmap();
                }
                return Err(err);
            }
            self.recycle(staging);
        }
        self.inner.waits.record(longest);
        Ok(())
    }

    /// Reads the first `bytes` bytes of `buffer` like [`Context::read`], but through one staging
    /// buffer of at most [`READ_CHUNK`] bytes and without noting the waits in [`WaitHistory`].
    /// For readbacks made while the device is full.
    pub(crate) fn read_spill(&self, buffer: &Buffer, bytes: u64) -> Result<Vec<u8>> {
        self.read_vec(bytes, |dst| {
            let chunk = READ_CHUNK.min(bytes).max(1);
            let staging = self.alloc_buffer(chunk, STAGING_USAGE)?;
            for offset in (0..bytes).step_by(chunk as usize) {
                let size = chunk.min(bytes - offset);
                let submission = {
                    let mut guard = self.batch()?;
                    self.encoder(&mut guard).copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
                    self.flush_locked(&mut guard)
                };
                let mapped = map_for_read(staging.slice(..size));
                let step = self.wait_map(submission, &mapped).and_then(|_| {
                    let view = staging.slice(..size).get_mapped_range().map_err(LumaError::MapRange)?;
                    copy_parallel(&view, &mut dst[offset as usize..(offset + size) as usize]);
                    Ok(())
                });
                staging.unmap();
                step?;
            }
            self.recycle(staging);
            Ok(())
        })
    }

    /// Waits for the GPU and returns the first `bytes` bytes of `staging`, a `MAP_READ` buffer
    /// whose contents were copied in by a submitted batch.
    pub(super) fn map_read<T: Pod>(&self, staging: &Buffer, bytes: u64) -> Result<Vec<T>> {
        let slice = staging.slice(..bytes);
        let waited = self.wait_map(None, &map_for_read(slice))?;
        self.inner.waits.record(waited);
        let result = slice.get_mapped_range().map(|view| bytemuck::pod_collect_to_vec(&view));
        staging.unmap();
        result.map_err(LumaError::MapRange)
    }

    /// Waits until `submission` (the latest one if `None`) has run and the mapping `mapped`
    /// reports on is done, and returns how long that took. Polls for it first, like
    /// [`Context::sync`].
    fn wait_map(&self, submission: Option<wgpu::SubmissionIndex>, mapped: &MapReceiver) -> Result<Duration> {
        let start = Instant::now();
        let mapped = match self.spin_for_map(mapped) {
            Some(received) => received,
            None => {
                self.device()
                    .poll(wgpu::PollType::Wait { submission_index: submission, timeout: None })
                    .map_err(|e| LumaError::Poll(Arc::new(e)))?;
                // The callback runs on whichever thread polls the device first, so it may still
                // be in flight on another thread after our poll returns; wait for it.
                mapped.recv()
            }
        };
        let waited = start.elapsed();
        let mapped = mapped.map_err(|_| LumaError::Map(wgpu::BufferAsyncError))?.map_err(LumaError::Map);
        // An error the GPU raised while running the batch explains a failed map better.
        self.take_gpu_error()?;
        mapped.map(|()| waited)
    }

    /// Polls without blocking for up to the spin budget ([`WaitHistory`]) until the mapping
    /// `mapped` reports on is done. `None` if it is not (or a poll failed, which the blocking
    /// wait reports).
    fn spin_for_map(
        &self,
        mapped: &MapReceiver,
    ) -> Option<std::result::Result<std::result::Result<(), wgpu::BufferAsyncError>, mpsc::RecvError>> {
        let deadline = Instant::now() + self.inner.waits.budget();
        loop {
            self.device().poll(wgpu::PollType::Poll).ok()?;
            match mapped.try_recv() {
                Ok(result) => return Some(Ok(result)),
                Err(mpsc::TryRecvError::Disconnected) => return Some(Err(mpsc::RecvError)),
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::hint::spin_loop();
        }
    }
}
