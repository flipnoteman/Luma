//! Segments: the device buffers an array is split into, and which of them are resident.
//!
//! An array larger than one device buffer (or than the device's memory) cannot live in a single
//! buffer, so its storage is a list of [`Segment`]s, each owning one buffer for a contiguous
//! element range. A segment gets its buffer when it is first [pinned](Segment::pin); a [`Pin`]
//! keeps the buffer in place while a command using it is recorded. When the data on the device
//! would pass the context's memory budget, the least recently pinned segments move to host
//! memory and come back at their next pin.

use std::cell::Cell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use crate::context::{copy_parallel, Context, STORAGE_USAGE};
use crate::error::Result;
use crate::gpu::pool;

/// Locks `mutex`, ignoring poisoning: the guarded lists stay consistent even if a holder
/// panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Smallest list length worth pruning.
const PRUNE_MIN: usize = 32;

/// Where a segment's data is.
#[derive(Debug)]
enum State {
    /// No buffer yet; the contents are unspecified, as those of a pooled buffer are.
    Empty,
    /// In a device buffer (possibly larger than the data it holds).
    Device(wgpu::Buffer),
    /// Moved to host memory to make room; copied back at the next pin.
    Host(Vec<u8>),
}

/// One device buffer holding a contiguous element range of an array.
///
/// Lock order: a segment's `state` may be held while taking the pager's `resident` list, never
/// the reverse.
#[derive(Debug)]
pub(crate) struct Segment {
    ctx: Context,
    /// Bytes of array data it holds (the buffer may be larger).
    bytes: u64,
    state: Mutex<State>,
    /// Number of live [`Pin`]s; a pinned segment must keep its buffer.
    pins: AtomicU32,
    /// Value of the pager's clock at the last pin.
    stamp: AtomicU64,
}

/// Keeps a segment's buffer on the device while a command using it is being recorded.
#[derive(Debug)]
pub(crate) struct Pin {
    segment: Arc<Segment>,
    buffer: wgpu::Buffer,
}

impl Pin {
    /// The segment's device buffer.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.segment.pins.fetch_sub(1, Ordering::Release);
    }
}

/// The segments that hold a device buffer, newest registrations last.
#[derive(Debug, Default)]
struct Resident {
    list: Vec<Weak<Segment>>,
    /// Length at which dead entries are dropped next.
    prune_at: usize,
}

/// Which segments of a context's arrays hold a device buffer, and how much memory they may use.
#[derive(Debug)]
pub(crate) struct Pager {
    /// Segments in [`State::Device`]; entries of dropped segments are pruned as it grows.
    resident: Mutex<Resident>,
    /// Sum of the sizes of their buffers.
    resident_bytes: AtomicU64,
    /// Counts pins; stamps segments in the order they were last pinned.
    clock: AtomicU64,
    /// Bytes of array data kept on the device; `None`: segments only move when the device
    /// reports that it is full.
    budget: Option<u64>,
    /// Bytes of array data moved to host memory so far.
    spilled_bytes: AtomicU64,
    /// Whether the spill warning has been printed.
    warned: AtomicBool,
    /// Held by the thread that is evicting, so one thread evicts at a time.
    evicting: Mutex<()>,
}

impl Pager {
    /// A pager with nothing resident that keeps `budget` bytes of array data on the device.
    pub(crate) fn new(budget: Option<u64>) -> Self {
        Pager {
            resident: Mutex::new(Resident::default()),
            resident_bytes: AtomicU64::new(0),
            clock: AtomicU64::new(0),
            budget,
            spilled_bytes: AtomicU64::new(0),
            warned: AtomicBool::new(false),
            evicting: Mutex::new(()),
        }
    }

    /// Records that `segment` now holds a device buffer of `size` bytes.
    fn register(&self, segment: &Arc<Segment>, size: u64) {
        self.resident_bytes.fetch_add(size, Ordering::Relaxed);
        let mut resident = lock(&self.resident);
        resident.list.push(Arc::downgrade(segment));
        if resident.list.len() > resident.prune_at.max(PRUNE_MIN) {
            resident.list.retain(|weak| weak.strong_count() > 0);
            resident.prune_at = resident.list.len().saturating_mul(2);
        }
    }
}

impl Segment {
    /// A segment of `bytes` bytes without a buffer; it gets one at its first pin.
    pub(crate) fn empty(ctx: &Context, bytes: u64) -> Arc<Segment> {
        Arc::new(Segment {
            ctx: ctx.clone(),
            bytes,
            state: Mutex::new(State::Empty),
            pins: AtomicU32::new(0),
            stamp: AtomicU64::new(0),
        })
    }

    /// A segment of `bytes` bytes that already lives in `buffer` (from [`Context::alloc_buffer`]
    /// or [`Context::alloc_filled`]); the segment recycles it when dropped.
    pub(crate) fn filled(ctx: &Context, bytes: u64, buffer: wgpu::Buffer) -> Arc<Segment> {
        let size = buffer.size();
        let segment = Arc::new(Segment {
            ctx: ctx.clone(),
            bytes,
            state: Mutex::new(State::Device(buffer)),
            pins: AtomicU32::new(0),
            stamp: AtomicU64::new(0),
        });
        ctx.pager().register(&segment, size);
        segment
    }

    /// The segment's buffer, allocating it (and bringing its data back from host memory) on
    /// first use. It stays on the device until the returned [`Pin`] is dropped.
    pub(crate) fn pin(self: &Arc<Self>) -> Result<Pin> {
        let pager = self.ctx.pager();
        loop {
            if let State::Device(buffer) = &*lock(&self.state) {
                self.pins.fetch_add(1, Ordering::Acquire);
                self.stamp.store(pager.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
                return Ok(Pin { segment: Arc::clone(self), buffer: buffer.clone() });
            }
            if let Some(budget) = pager.budget {
                // The budget is soft: with nothing left to evict the allocation goes ahead.
                let class = pool::size_class(self.bytes, self.ctx.max_buffer_bytes()).unwrap_or(self.bytes);
                while pager.resident_bytes.load(Ordering::Relaxed) + class > budget && self.ctx.evict_one()? { std::hint::spin_loop() }
            }
            let buffer = self.ctx.alloc_buffer(self.bytes, STORAGE_USAGE)?;
            let mut state = lock(&self.state);
            match &*state {
                State::Device(_) => {
                    // Another thread pinned it meanwhile.
                    drop(state);
                    self.ctx.recycle(buffer);
                }
                State::Host(data) => {
                    if let Err(e) = self.ctx.upload(&buffer, data) {
                        drop(state);
                        self.ctx.recycle(buffer);
                        return Err(e);
                    }
                    pager.register(self, buffer.size());
                    *state = State::Device(buffer);
                }
                State::Empty => {
                    pager.register(self, buffer.size());
                    *state = State::Device(buffer);
                }
            }
        }
    }

    /// Copies the segment's data into `dst` (`dst.len()` bytes), after everything recorded so
    /// far. A segment in host memory is copied from there and stays there.
    pub(crate) fn read_into(self: &Arc<Self>, dst: &mut [MaybeUninit<u8>]) -> Result<()> {
        if let State::Host(data) = &*lock(&self.state) {
            copy_parallel(&data[..dst.len()], dst);
            return Ok(());
        }
        let pin = self.pin()?;
        self.ctx.read_into(pin.buffer(), dst)
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let State::Device(buffer) = std::mem::replace(state, State::Empty) {
            self.ctx.pager().resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
            self.ctx.recycle(buffer);
        }
    }
}

thread_local! {
    /// Whether the current thread is inside [`Context::evict_one`]: its readback allocates
    /// staging buffers, which must not start another eviction.
    static EVICTING: Cell<bool> = const { Cell::new(false) };
}

/// Clears [`EVICTING`] on every way out of an eviction.
struct EvictingGuard;

impl Drop for EvictingGuard {
    fn drop(&mut self) {
        EVICTING.with(|flag| flag.set(false));
    }
}

/// Printed once per context when array data first moves to host memory.
const SPILL_WARNING: &str = "luma: device memory is full; array data is being moved to host memory and back as needed. Results are unaffected; throughput drops to the transfer rate.";

impl Context {
    /// Moves the least recently pinned unpinned segment to host memory. `Ok(false)`: nothing to
    /// move. `Ok(true)`: a segment moved, or the choice raced with a pin; the caller retries.
    pub(crate) fn evict_one(&self) -> Result<bool> {
        if EVICTING.with(Cell::get) {
            return Ok(false);
        }
        EVICTING.with(|flag| flag.set(true));
        let _guard = EvictingGuard;
        let pager = self.pager();
        let _one_at_a_time = lock(&pager.evicting);

        // Choose the victim; `resident` is released before the victim's `state` is locked.
        let victim = {
            let mut resident = lock(&pager.resident);
            resident.list.retain(|weak| weak.strong_count() > 0);
            resident
                .list
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|segment| segment.pins.load(Ordering::Acquire) == 0)
                .min_by_key(|segment| segment.stamp.load(Ordering::Relaxed))
        };
        let Some(victim) = victim else { return Ok(false) };

        // Held to the end: a concurrent pin of the victim waits for the move to finish.
        let mut state = lock(&victim.state);
        let State::Device(buffer) = &*state else { return Ok(true) };
        if victim.pins.load(Ordering::Acquire) != 0 {
            return Ok(true);
        }
        let buffer = buffer.clone();
        // Flushes and waits, so every recorded use of the buffer has run.
        let data = self.read::<u8>(&buffer, victim.bytes)?;
        *state = State::Host(data);
        pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        pager.spilled_bytes.fetch_add(victim.bytes, Ordering::Relaxed);
        lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&victim)));
        self.recycle(buffer);
        drop(state);

        if !pager.warned.swap(true, Ordering::Relaxed) {
            if log::log_enabled!(log::Level::Warn) {
                log::warn!("{SPILL_WARNING}");
            } else {
                eprintln!("{SPILL_WARNING}");
            }
        }
        Ok(true)
    }

    /// Bytes of array data moved from the device to host memory so far.
    pub fn spilled_bytes(&self) -> u64 {
        self.pager().spilled_bytes.load(Ordering::Relaxed)
    }
}
