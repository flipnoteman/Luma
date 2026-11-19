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
use std::sync::{Arc, Mutex, PoisonError, Weak};

use luma_core::hal::copy_parallel;
use luma_core::Buffer;

use crate::array::Storage;
use crate::context::{Context, STORAGE_USAGE};
use crate::error::Result;
use crate::gpu::hash::{FxMap, FxSet};
use crate::gpu::pool;
use crate::gpu::lock;

/// Smallest list length worth pruning.
const PRUNE_MIN: usize = 32;

/// Where a segment's data is.
#[derive(Debug)]
enum State {
    /// No buffer yet; the contents are unspecified, as those of a pooled buffer are.
    Empty,
    /// In a device buffer (possibly larger than the data it holds).
    Device(Buffer),
    /// Moved to host memory to make room; copied back at the next pin.
    Host(Vec<u8>),
    /// A range of the buffer of another storage (see [`Segment::windowed`]); gets a buffer of its
    /// own, a copy of the range, at the first pin.
    Window(Window),
}

/// Where a windowed segment's data is: `bytes` bytes of the one buffer of `pack`, from `offset`.
#[derive(Debug)]
struct Window {
    pack: Arc<Storage>,
    offset: u64,
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
    buffer: Buffer,
}

impl Pin {
    /// Another pin of the same segment.
    pub(crate) fn dup(&self) -> Pin {
        self.segment.pins.fetch_add(1, Ordering::Acquire);
        Pin { segment: Arc::clone(&self.segment), buffer: self.buffer.clone() }
    }

    /// The segment's device buffer.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.segment.pins.fetch_sub(1, Ordering::Release);
    }
}

/// Bytes of device buffers that no segment owns but a recording, a plan or a loop holds, counted
/// in the pager's `resident_bytes` (so the memory budget sees them) until this is dropped.
#[derive(Debug)]
pub(crate) struct Accounted {
    ctx: Context,
    bytes: u64,
}

impl Accounted {
    /// Counts `bytes` as resident.
    pub(crate) fn new(ctx: &Context, bytes: u64) -> Accounted {
        ctx.pager().resident_bytes.fetch_add(bytes, Ordering::Relaxed);
        Accounted { ctx: ctx.clone(), bytes }
    }

    /// Counts `bytes` more.
    pub(crate) fn add(&mut self, bytes: u64) {
        self.ctx.pager().resident_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.bytes += bytes;
    }

    /// Stops counting `bytes` of what was counted so far.
    pub(crate) fn sub(&mut self, bytes: u64) {
        debug_assert!(bytes <= self.bytes, "uncounting more than was counted");
        self.ctx.pager().resident_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.bytes -= bytes;
    }

    /// Stops counting everything counted so far.
    pub(crate) fn clear(&mut self) {
        self.ctx.pager().resident_bytes.fetch_sub(self.bytes, Ordering::Relaxed);
        self.bytes = 0;
    }

    /// Bytes counted so far.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Accounted {
    fn drop(&mut self) {
        self.ctx.pager().resident_bytes.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// Ends the pins in `pins` of every segment that nothing but those pins holds and whose buffer
/// is in `fresh` (by id), and returns those buffers: they no longer belong to any segment (the
/// segments are dropped without giving them to the pool) and the caller owns them. Their bytes
/// leave `resident_bytes`; whoever keeps a buffer counts it again (see [`Accounted`]).
///
/// A segment is held by the storage that owns it, by windows of it through that storage, and by
/// pins. When every strong reference is one of `pins`, no array reads or writes the segment any
/// more, and nothing can reach it again (a `Weak` of the pager's list cannot pick a victim that
/// is pinned). The pins are taken out last, so the segment stays pinned throughout.
pub(crate) fn take_dead_buffers(pins: &mut Vec<Pin>, fresh: &FxSet<u64>) -> Vec<Buffer> {
    let mut held: FxMap<*const Segment, usize> = FxMap::default();
    for pin in pins.iter() {
        *held.entry(Arc::as_ptr(&pin.segment)).or_insert(0) += 1;
    }
    let mut taken = Vec::new();
    let mut dead: FxSet<*const Segment> = FxSet::default();
    for pin in pins.iter() {
        let segment = &pin.segment;
        let key = Arc::as_ptr(segment);
        if dead.contains(&key) || Arc::strong_count(segment) != held[&key] {
            continue;
        }
        let mut state = lock(&segment.state);
        let State::Device(buffer) = &*state else { continue };
        if !fresh.contains(&buffer.id()) {
            continue;
        }
        let State::Device(buffer) = std::mem::replace(&mut *state, State::Empty) else { continue };
        let pager = segment.ctx.pager();
        pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), key));
        drop(state);
        dead.insert(key);
        taken.push(buffer);
    }
    pins.retain(|pin| !dead.contains(&Arc::as_ptr(&pin.segment)));
    taken
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
    /// Bytes counted against the memory budget.
    #[cfg(test)]
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Counts `bytes` of device buffers the context holds until it is dropped (nothing gives
    /// them back, so there is nothing to uncount).
    pub(crate) fn count_for_life(&self, bytes: u64) {
        self.resident_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Whether resident array data has reached three quarters of the memory budget.
    pub(crate) fn under_pressure(&self) -> bool {
        self.budget.is_some_and(|budget| self.resident_bytes.load(Ordering::Relaxed) >= budget / 4 * 3)
    }

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
        segment.stamp.store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
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
    pub(crate) fn filled(ctx: &Context, bytes: u64, buffer: Buffer) -> Arc<Segment> {
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

    /// A segment of `bytes` bytes that shows `bytes` bytes of `pack` (a storage of one segment)
    /// from byte `offset`, without a buffer of its own: kernels that know windows read the
    /// range through the pack (see `jit::pack`), anything else that pins the segment gets a
    /// copy of it first (see [`Segment::pin`]), so the data stays valid whatever writes to it.
    pub(crate) fn windowed(ctx: &Context, bytes: u64, pack: Arc<Storage>, offset: u64) -> Arc<Segment> {
        Arc::new(Segment {
            ctx: ctx.clone(),
            bytes,
            state: Mutex::new(State::Window(Window { pack, offset })),
            pins: AtomicU32::new(0),
            stamp: AtomicU64::new(0),
        })
    }

    /// Whether the segment is a window or a device buffer no command being recorded holds.
    pub(crate) fn can_adopt(&self) -> bool {
        match &*lock(&self.state) {
            State::Device(_) => self.pins.load(Ordering::Acquire) == 0,
            State::Window(_) => true,
            State::Empty | State::Host(_) => false,
        }
    }

    /// Makes the segment a window of `pack` from byte `offset` (of another one, if it was
    /// one): the buffer it had is given back to the pool (commands recorded so far that use it
    /// still run first), and the segment shows the range of `pack` that holds the same data now.
    /// `false` if the segment has no device buffer, or one that a recording holds.
    pub(crate) fn adopt(self: &Arc<Self>, pack: Arc<Storage>, offset: u64) -> bool {
        let mut state = lock(&self.state);
        match &*state {
            State::Window(_) => {
                *state = State::Window(Window { pack, offset });
                return true;
            }
            State::Device(_) if self.pins.load(Ordering::Acquire) == 0 => {}
            _ => return false,
        }
        let State::Device(buffer) = std::mem::replace(&mut *state, State::Window(Window { pack, offset })) else { return false };
        let pager = self.ctx.pager();
        pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(self)));
        drop(state);
        self.ctx.recycle(buffer);
        true
    }

    /// The pack and the byte offset in it if the segment still only shows a range of it.
    pub(crate) fn window(&self) -> Option<(Arc<Storage>, u64)> {
        match &*lock(&self.state) {
            State::Window(window) => Some((Arc::clone(&window.pack), window.offset)),
            _ => None,
        }
    }

    /// The segment's buffer, allocating it (and bringing its data back from host memory) on
    /// first use; a window gets a buffer holding a copy of its range. It stays on the device
    /// until the returned [`Pin`] is dropped.
    pub(crate) fn pin(self: &Arc<Self>) -> Result<Pin> {
        let pin = self.pin_unrecorded()?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    fn pin_unrecorded(self: &Arc<Self>) -> Result<Pin> {
        let pager = self.ctx.pager();
        loop {
            if let State::Device(buffer) = &*lock(&self.state) {
                self.pins.fetch_add(1, Ordering::Acquire);
                self.stamp.store(pager.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
                return Ok(Pin { segment: Arc::clone(self), buffer: buffer.clone() });
            }
            if matches!(&*lock(&self.state), State::Host(_)) {
                self.ctx.abort_recording()?;
            }
            let window = match &*lock(&self.state) {
                State::Window(window) => Some((Arc::clone(&window.pack), window.offset)),
                _ => None,
            };
            let buffer = match self.ctx.take_recorded(self.bytes, STORAGE_USAGE) {
                Some(buffer) => buffer,
                None => {
                    self.ctx.make_room(self.bytes)?;
                    self.ctx.alloc_buffer(self.bytes, STORAGE_USAGE)?
                }
            };
            if let Some((pack, offset)) = window {
                // The copy runs after the launch that wrote the pack and before any later one.
                let source = pack.pin(0)?;
                if let Err(e) = self.ctx.copy(source.buffer(), offset, &buffer, 0, self.bytes) {
                    self.ctx.recycle(buffer);
                    return Err(e);
                }
            }
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
                State::Empty | State::Window(_) => {
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
    /// Evicts segments until `bytes` more fit in the memory budget. The budget is soft: with
    /// nothing left to evict the allocation goes ahead.
    pub(crate) fn make_room(&self, bytes: u64) -> Result<()> {
        let pager = self.pager();
        let Some(budget) = pager.budget else { return Ok(()) };
        let class = pool::size_class(bytes, self.max_buffer_bytes()).unwrap_or(bytes);
        while pager.resident_bytes.load(Ordering::Relaxed) + class > budget && self.evict_one()? { std::hint::spin_loop() }
        Ok(())
    }

    /// Moves the least recently pinned unpinned segment to host memory. `Ok(false)`: nothing to
    /// move. `Ok(true)`: a segment moved, or the choice raced with a pin; the caller retries.
    pub(crate) fn evict_one(&self) -> Result<bool> {
        if EVICTING.with(Cell::get) {
            return Ok(false);
        }
        // Before any lock: aborting a recording can pin (and so lock) the victim's segment.
        self.abort_recording()?;
        let before = self.pager().resident_bytes.load(Ordering::Relaxed);
        self.jit().drop_speculated();
        if self.pager().resident_bytes.load(Ordering::Relaxed) < before {
            return Ok(true);
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
        let data = self.read_spill(&buffer, victim.bytes)?;
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

    /// What the context holds in device memory now (see [`MemoryStats`]).
    pub fn memory_stats(&self) -> MemoryStats {
        let pager = self.pager();
        MemoryStats {
            resident_bytes: pager.resident_bytes.load(Ordering::Relaxed),
            pooled_bytes: self.pooled_bytes(),
            spilled_bytes: pager.spilled_bytes.load(Ordering::Relaxed),
            budget_bytes: pager.budget,
        }
    }
}

/// What a context holds in device memory, from [`Context::memory_stats`]. Sizes are those of the
/// device buffers, which may be larger than the data they hold.
///
/// The two byte counts cover the device buffers luma makes: `resident_bytes` those in use,
/// `pooled_bytes` those idle. What the driver or the backend keeps besides them is in neither
/// and shows in `nvidia-smi` or the like on top of them: the device context itself (a few
/// hundred MiB on CUDA), compiled kernel code, the backend's own uniform and staging chunks,
/// what a captured graph keeps internally (the device copy of its uniforms is counted), and,
/// on CUDA, the driver's pool reserve of memory that was freed but not yet returned to the
/// device ([`Context::trim`] returns it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes of device buffers in use: array data; what recordings, recorded plans and loops
    /// keep (the intermediates of a plan, state placeholders, a loop's work buffers, a captured
    /// graph's uniforms); the scratch an operation is using; the context's own counters. This
    /// is what the memory budget counts.
    pub resident_bytes: u64,
    /// Bytes of idle buffers kept for reuse; [`Context::trim`] releases them.
    pub pooled_bytes: u64,
    /// Bytes of array data moved from the device to host memory so far.
    pub spilled_bytes: u64,
    /// Bytes of array data the context keeps on the device (`None`: no budget; data only moves
    /// to host memory when the device reports that it is full).
    pub budget_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_starts_at_three_quarters_of_the_budget() {
        let pager = Pager::new(Some(1000));
        pager.resident_bytes.store(749, Ordering::Relaxed);
        assert!(!pager.under_pressure());
        pager.resident_bytes.store(750, Ordering::Relaxed);
        assert!(pager.under_pressure());
        assert!(!Pager::new(None).under_pressure());
    }
}
