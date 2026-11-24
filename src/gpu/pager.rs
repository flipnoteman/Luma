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

use luma_core::hal::{copy_parallel, copy_parallel_host, HostBytes};
use luma_core::Buffer;

use crate::array::Storage;
use crate::context::{Context, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu::hash::{FxMap, FxSet};
use crate::jit::lineage::{Lineage, Recompute};
use luma_core::hal::size_class;
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
    Host(HostBytes),
    /// A range of the buffer of another storage (see [`Segment::windowed`]); gets a buffer of its
    /// own, a copy of the range, at the first pin.
    Window(Window),
    /// Not in memory: the contents are what the segment's lineage computes (see
    /// [`crate::jit::lineage`]), run again at the next pin. The lineage holds the inputs.
    Lineage,
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
    /// A valid copy of the device bytes, held while the segment is on the device: if it is
    /// `Some`, nothing wrote the device buffer since the copy was made. Taken by every write
    /// pin ([`Segment::pin`]); lock order: after `state`.
    host_copy: Mutex<Option<HostBytes>>,
    /// Whether `host_copy` may be `Some` (a cheap check for the write pins).
    has_copy: AtomicBool,
    /// How the data was computed, if by a plain store kernel (see [`crate::jit::lineage`]);
    /// lock order: after `state`.
    lineage: Mutex<Option<Box<Lineage>>>,
    /// Whether `lineage` may be `Some` (a cheap check for the write pins).
    has_lineage: AtomicBool,
    /// The segments whose lineage reads this one. A write pin of this segment settles them
    /// first (see [`Segment::pin`]); lock order: never held while another segment's `state` is
    /// taken.
    dependents: Mutex<Dependents>,
    /// Whether `dependents` may hold a segment (a cheap check for the write pins).
    has_dependents: AtomicBool,
    /// Number of live [`Pin`]s; a pinned segment must keep its buffer.
    pins: AtomicU32,
    /// Value of the pager's clock at the last pin.
    stamp: AtomicU64,
    /// The texture of this segment's data made last (see [`crate::texture`]).
    texture: Mutex<Option<crate::texture::Cached>>,
}

/// The segments that read one segment through their lineage; entries of dropped segments are
/// pruned as the list grows.
#[derive(Debug, Default)]
struct Dependents {
    list: Vec<Weak<Segment>>,
    /// Length at which dead entries are dropped next.
    prune_at: usize,
}

/// Keeps a segment's buffer on the device while a command using it is being recorded.
#[derive(Debug)]
pub(crate) struct Pin {
    segment: Arc<Segment>,
    buffer: Buffer,
}

impl Pin {
    /// The segment this pin keeps resident.
    pub(crate) fn segment(&self) -> &Arc<Segment> {
        &self.segment
    }

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
        segment.invalidate_host_copy();
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
    /// Bytes of host memory the context may use for host copies of device data.
    host_budget: u64,
    /// Bytes of array data in [`State::Host`] now (the only copy of it).
    host_state_bytes: AtomicU64,
    /// Bytes of kept host copies of segments on the device.
    host_copy_bytes: AtomicU64,
    /// Evictions that moved a valid host copy instead of reading the buffer back.
    clean_evictions: AtomicU64,
    /// Evictions that read the buffer back.
    readback_evictions: AtomicU64,
    /// Evictions that moved nothing: the segment is recomputed at its next pin (see `jit::lineage`).
    recompute_evictions: AtomicU64,
    /// Bytes of inputs that segments evicted by recompute keep alive.
    lineage_pinned_bytes: AtomicU64,
    /// Evict by lineage whenever it is allowed, whatever the cost model says (the fuzzer's
    /// arrays are too small for recomputing to beat a transfer).
    #[cfg(test)]
    force_lineage: AtomicBool,
    /// Whether the over-budget warning has been printed.
    host_warned: AtomicBool,
    /// Windows given a buffer of their own, a copy of their range (see [`Segment::windowed`]).
    window_copies: AtomicU64,
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

    /// Windows copied out of their pack so far.
    #[cfg(test)]
    pub(crate) fn window_copies(&self) -> u64 {
        self.window_copies.load(Ordering::Relaxed)
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

    /// Whether the context keeps array data under a memory budget: only there is a segment's
    /// lineage recorded (see `jit::lineage`).
    pub(crate) fn evicts_under_budget(&self) -> bool {
        self.budget.is_some()
    }

    /// Makes eviction by lineage ignore its cost (tests).
    #[cfg(test)]
    pub(crate) fn force_lineage(&self) {
        self.force_lineage.store(true, Ordering::Relaxed);
    }

    /// Whether eviction by lineage ignores its cost.
    pub(crate) fn lineage_forced(&self) -> bool {
        #[cfg(test)]
        return self.force_lineage.load(Ordering::Relaxed);
        #[cfg(not(test))]
        false
    }

    /// A pager with nothing resident that keeps `budget` bytes of array data on the device.
    pub(crate) fn new(budget: Option<u64>, host_budget: u64) -> Self {
        Pager {
            resident: Mutex::new(Resident::default()),
            resident_bytes: AtomicU64::new(0),
            clock: AtomicU64::new(0),
            budget,
            spilled_bytes: AtomicU64::new(0),
            host_budget,
            host_state_bytes: AtomicU64::new(0),
            host_copy_bytes: AtomicU64::new(0),
            clean_evictions: AtomicU64::new(0),
            readback_evictions: AtomicU64::new(0),
            recompute_evictions: AtomicU64::new(0),
            lineage_pinned_bytes: AtomicU64::new(0),
            #[cfg(test)]
            force_lineage: AtomicBool::new(false),
            host_warned: AtomicBool::new(false),
            window_copies: AtomicU64::new(0),
            warned: AtomicBool::new(false),
            evicting: Mutex::new(()),
        }
    }

    /// Whether host copies are kept at all: only where segments move off the device.
    fn keeps_copies(&self) -> bool {
        self.budget.is_some() && self.host_budget > 0
    }

    /// Bytes of host memory held for device data: the segments in host memory and the copies.
    fn host_bytes(&self) -> u64 {
        self.host_state_bytes.load(Ordering::Relaxed) + self.host_copy_bytes.load(Ordering::Relaxed)
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
    /// The texture of this segment's data, see [`crate::texture`].
    pub(crate) fn texture_slot(&self) -> &Mutex<Option<crate::texture::Cached>> {
        &self.texture
    }

    /// A segment of `bytes` bytes in `state`, with nothing else to its name yet.
    fn with_state(ctx: &Context, bytes: u64, state: State) -> Arc<Segment> {
        Arc::new(Segment {
            ctx: ctx.clone(),
            bytes,
            state: Mutex::new(state),
            host_copy: Mutex::new(None),
            has_copy: AtomicBool::new(false),
            lineage: Mutex::new(None),
            has_lineage: AtomicBool::new(false),
            dependents: Mutex::new(Dependents::default()),
            has_dependents: AtomicBool::new(false),
            pins: AtomicU32::new(0),
            stamp: AtomicU64::new(0),
            texture: Mutex::new(None),
        })
    }

    /// A segment of `bytes` bytes without a buffer; it gets one at its first pin.
    pub(crate) fn empty(ctx: &Context, bytes: u64) -> Arc<Segment> {
        Self::with_state(ctx, bytes, State::Empty)
    }

    /// A segment of `bytes` bytes that already lives in `buffer` (from [`Context::alloc_buffer`]
    /// or [`Context::alloc_filled`]); the segment recycles it when dropped.
    pub(crate) fn filled(ctx: &Context, bytes: u64, buffer: Buffer) -> Arc<Segment> {
        let size = buffer.size();
        let segment = Self::with_state(ctx, bytes, State::Device(buffer));
        ctx.pager().register(&segment, size);
        segment
    }

    /// A segment of `bytes` bytes that shows `bytes` bytes of `pack` (a storage of one segment)
    /// from byte `offset`, without a buffer of its own: kernels that know windows read the
    /// range through the pack (see `jit::pack`), anything else that pins the segment gets a
    /// copy of it first (see [`Segment::pin`]), so the data stays valid whatever writes to it.
    pub(crate) fn windowed(ctx: &Context, bytes: u64, pack: Arc<Storage>, offset: u64) -> Arc<Segment> {
        Self::with_state(ctx, bytes, State::Window(Window { pack, offset }))
    }

    /// Whether the segment is a window or a device buffer no command being recorded holds.
    pub(crate) fn can_adopt(&self) -> bool {
        match &*lock(&self.state) {
            State::Device(_) => self.pins.load(Ordering::Acquire) == 0,
            State::Window(_) => true,
            State::Empty | State::Host(_) | State::Lineage => false,
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
        self.invalidate_host_copy();
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
    ///
    /// The pin may be used to write: the host copy of the data, if the segment keeps one, is
    /// dropped. [`Segment::pin_read`] keeps it.
    pub(crate) fn pin(self: &Arc<Self>) -> Result<Pin> {
        let pin = self.pin_unrecorded(true)?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    /// [`Segment::pin`] for a command that only reads the buffer: a host copy stays valid.
    pub(crate) fn pin_read(self: &Arc<Self>) -> Result<Pin> {
        let pin = self.pin_unrecorded(false)?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    /// [`Segment::pin`] without keeping the pin for the recording (see `keep_pin`).
    pub(crate) fn pin_quiet(self: &Arc<Self>) -> Result<Pin> {
        self.pin_unrecorded(true)
    }

    /// Drops the kept host copy: the device buffer is about to be, or was, written.
    pub(crate) fn invalidate_host_copy(&self) {
        if self.has_copy.load(Ordering::Acquire) {
            let mut copy = lock(&self.host_copy);
            if let Some(data) = copy.take() {
                self.ctx.pager().host_copy_bytes.fetch_sub(data.len() as u64, Ordering::Relaxed);
            }
            self.has_copy.store(false, Ordering::Release);
        }
    }

    fn pin_unrecorded(self: &Arc<Self>, write: bool) -> Result<Pin> {
        if !write {
            return self.pin_device(false);
        }
        // A write: what is evicted by lineage over this data is recomputed first, while it holds
        // the old values, and no lineage over it survives (see `jit::lineage`).
        if self.has_dependents.load(Ordering::Acquire) {
            self.release_dependents()?;
        }
        let pin = self.pin_device(true)?;
        if self.has_lineage.load(Ordering::Acquire) {
            // The data is about to change: the pin keeps the segment resident, so it is not evicted by lineage meanwhile.
            self.clear_lineage();
        }
        Ok(pin)
    }

    /// Before a write to this segment: recomputes the dependents that are evicted by lineage,
    /// then drops the lineage of **every** dependent, resident ones included (their weak inputs
    /// would still upgrade and describe the old values), and forgets them.
    fn release_dependents(self: &Arc<Self>) -> Result<()> {
        let dependents = std::mem::take(&mut lock(&self.dependents).list);
        for (k, weak) in dependents.iter().enumerate() {
            let Some(dependent) = weak.upgrade() else { continue };
            if let Err(error) = dependent.give_up_lineage() {
                // The write does not happen: the dependents not dealt with stay registered.
                lock(&self.dependents).list.extend(dependents[k..].iter().cloned());
                return Err(error);
            }
        }
        let dependents = lock(&self.dependents);
        if dependents.list.is_empty() {
            self.has_dependents.store(false, Ordering::Release);
        }
        Ok(())
    }

    /// Recomputes the segment if it is evicted by lineage, then drops its lineage.
    fn give_up_lineage(self: &Arc<Self>) -> Result<()> {
        loop {
            // Pinned, the recomputed segment cannot be evicted by lineage again before its lineage is gone.
            let _pin = if matches!(&*lock(&self.state), State::Lineage) { Some(self.pin_device(false)?) } else { None };
            if self.drop_lineage() {
                return Ok(());
            }
        }
    }

    /// Drops the lineage, unless the segment is evicted by lineage (its data exists as that
    /// only); whether it was dropped.
    fn drop_lineage(&self) -> bool {
        let state = lock(&self.state);
        if matches!(&*state, State::Lineage) {
            return false;
        }
        self.clear_lineage();
        true
    }

    /// Drops the lineage. The segment must not be evicted by lineage: the caller holds its
    /// `state` or a pin on it.
    fn clear_lineage(&self) {
        if self.has_lineage.load(Ordering::Acquire) {
            let old = {
                let mut lineage = lock(&self.lineage);
                self.has_lineage.store(false, Ordering::Release);
                lineage.take()
            };
            drop(old);
        }
    }

    /// Records that the data of this segment is what `lineage` computes from `inputs`: the
    /// segment is a dependent of each of their segments.
    pub(crate) fn set_lineage(self: &Arc<Self>, lineage: Lineage, inputs: &[Arc<Storage>]) {
        for input in inputs {
            input.only_segment().add_dependent(self);
        }
        *lock(&self.lineage) = Some(Box::new(lineage));
        self.has_lineage.store(true, Ordering::Release);
    }

    fn add_dependent(&self, dependent: &Arc<Segment>) {
        let mut dependents = lock(&self.dependents);
        dependents.list.push(Arc::downgrade(dependent));
        if dependents.list.len() > dependents.prune_at.max(PRUNE_MIN) {
            dependents.list.retain(|weak| weak.strong_count() > 0);
            dependents.prune_at = dependents.list.len().saturating_mul(2);
        }
        self.has_dependents.store(true, Ordering::Release);
    }

    /// Whether the segment holds a device buffer.
    pub(crate) fn on_device(&self) -> bool {
        matches!(&*lock(&self.state), State::Device(_))
    }

    /// The inputs, strongly, if evicting this segment by lineage is allowed and cheaper than
    /// moving it (a segment with a valid host copy, `clean`, only needs an upload later; see [`crate::jit::lineage::Snapshot::evictable_inputs`]). Besides that, no
    /// segment that reads this one may be evicted by lineage itself, and (checked there) neither
    /// may an input: with both, a recomputation never reaches through more than one level.
    ///
    /// Takes the state locks of other segments, one at a time: the caller holds none.
    fn lineage_candidate(&self, clean: bool) -> Option<Box<[Arc<Storage>]>> {
        if !self.has_lineage.load(Ordering::Acquire) {
            return None;
        }
        let snapshot = lock(&self.lineage).as_ref()?.snapshot();
        let dependents: Vec<Arc<Segment>> = lock(&self.dependents).list.iter().filter_map(Weak::upgrade).collect();
        if dependents.iter().any(|dependent| matches!(&*lock(&dependent.state), State::Lineage)) {
            return None;
        }
        snapshot.evictable_inputs(&self.ctx, self.bytes, clean)
    }

    /// Makes `inputs` the strong inputs of the lineage (the caller evicts the segment by
    /// lineage, holding its `state`); the bytes they keep alive. `None` if the lineage is gone.
    fn hold_lineage_inputs(&self, inputs: Box<[Arc<Storage>]>) -> Option<u64> {
        Some(lock(&self.lineage).as_mut()?.hold(inputs))
    }

    /// Lets go of the strong inputs of the lineage: the segment is resident again.
    fn release_lineage_inputs(&self) {
        let freed = lock(&self.lineage).as_mut().map(|lineage| lineage.release());
        if let Some((bytes, inputs)) = freed {
            self.ctx.pager().lineage_pinned_bytes.fetch_sub(bytes, Ordering::Relaxed);
            drop(inputs);
        }
    }

    /// What recomputes the segment; `None` if it is no longer evicted by lineage.
    fn recompute_plan(&self) -> Result<Option<Recompute>> {
        let plan = lock(&self.lineage).as_ref().and_then(|lineage| lineage.recompute());
        if plan.is_none() && matches!(&*lock(&self.state), State::Lineage) {
            return Err(LumaError::Gpu("a segment evicted by recompute lost its lineage".into()));
        }
        Ok(plan)
    }

    /// The device buffer of the segment, in place for a resident one; the segment is allocated,
    /// brought back from host memory, copied out of its pack or recomputed otherwise.
    fn pin_device(self: &Arc<Self>, write: bool) -> Result<Pin> {
        let pager = self.ctx.pager();
        loop {
            if let State::Device(buffer) = &*lock(&self.state) {
                if write {
                    self.invalidate_host_copy();
                }
                self.pins.fetch_add(1, Ordering::Acquire);
                self.stamp.store(pager.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
                return Ok(Pin { segment: Arc::clone(self), buffer: buffer.clone() });
            }
            let evicted_by_lineage = matches!(&*lock(&self.state), State::Lineage);
            if evicted_by_lineage || matches!(&*lock(&self.state), State::Host(_)) {
                self.ctx.abort_recording()?;
            }
            let window = match &*lock(&self.state) {
                State::Window(window) => Some((Arc::clone(&window.pack), window.offset)),
                _ => None,
            };
            // The inputs of a recomputation are pinned first, so that making room does not evict them.
            let recompute = if evicted_by_lineage {
                match self.recompute_plan()? {
                    Some(plan) => Some((plan.pin_inputs()?, plan)),
                    None => continue,
                }
            } else {
                None
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
                let source = pack.pin_read(0)?;
                pager.window_copies.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = self.ctx.copy(source.buffer(), offset, &buffer, 0, self.bytes) {
                    self.ctx.recycle(buffer);
                    return Err(e);
                }
            }
            if let Some((pinned, plan)) = &recompute
                && let Err(e) = plan.launch(&self.ctx, &buffer, self.bytes, pinned)
            {
                self.ctx.recycle(buffer);
                return Err(e);
            }
            let mut state = lock(&self.state);
            if matches!(&*state, State::Device(_)) {
                // Another thread pinned it meanwhile.
                drop(state);
                self.ctx.recycle(buffer);
                continue;
            }
            if matches!(&*state, State::Host(_)) {
                let State::Host(data) = std::mem::replace(&mut *state, State::Empty) else { unreachable!() };
                if let Err(e) = self.ctx.upload(&buffer, data.as_slice()) {
                    *state = State::Host(data);
                    drop(state);
                    self.ctx.recycle(buffer);
                    return Err(e);
                }
                pager.register(self, buffer.size());
                *state = State::Device(buffer);
                let len = data.len() as u64;
                pager.host_state_bytes.fetch_sub(len, Ordering::Relaxed);
                // The device holds what the host does: keep the copy while the budget allows, unless this pin may change the data.
                if !write && pager.keeps_copies() && pager.host_bytes() + len <= pager.host_budget {
                    let mut copy = lock(&self.host_copy);
                    *copy = Some(data);
                    pager.host_copy_bytes.fetch_add(len, Ordering::Relaxed);
                    self.has_copy.store(true, Ordering::Release);
                }
            } else {
                let was_evicted_by_lineage = matches!(&*state, State::Lineage);
                pager.register(self, buffer.size());
                *state = State::Device(buffer);
                drop(state);
                if was_evicted_by_lineage {
                    self.release_lineage_inputs();
                }
            }
        }
    }

    /// Copies the segment's data into `dst` (`dst.len()` bytes), after everything recorded so
    /// far. A segment in host memory is copied from there and stays there.
    pub(crate) fn read_into(self: &Arc<Self>, dst: &mut [MaybeUninit<u8>]) -> Result<()> {
        // On the CPU backend the segment is in the same memory as `dst`: a copy between host buffers, bound by the host's memory.
        let copy = if self.ctx.is_cpu() { copy_parallel_host } else { copy_parallel };
        if self.with_host_bytes(|data| copy(&data[..dst.len()], dst)).is_some() {
            return Ok(());
        }
        let pin = self.pin_read()?;
        self.ctx.read_into(pin.buffer(), dst)
    }

    /// Calls `f` with the segment's data when the host can read it right now: held in host
    /// memory, or in an idle directly accessible device buffer. `None` otherwise.
    pub(crate) fn with_host_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        if self.ctx.recording_here() {
            return None;
        }
        match &*lock(&self.state) {
            State::Host(data) => Some(f(&data.as_slice()[..self.bytes as usize])),
            State::Device(buffer) if self.pins.load(Ordering::Acquire) == 0 => self.ctx.host_ptr(buffer, self.bytes).map(|ptr| {
                // SAFETY: `host_ptr` vouches for `self.bytes` readable bytes no GPU command writes, and the state lock keeps the buffer from being evicted.
                f(unsafe { std::slice::from_raw_parts(ptr, self.bytes as usize) })
            }),
            State::Device(_) | State::Empty | State::Window(_) | State::Lineage => None,
        }
    }

    /// [`Segment::with_host_bytes`] with write access; a device buffer must not be pinned by a recording.
    pub(crate) fn with_host_bytes_mut<R>(self: &Arc<Self>, f: impl FnOnce(&mut [u8]) -> R) -> Option<R> {
        if self.ctx.recording_here() {
            return None;
        }
        // A write: what is evicted by lineage over this data is recomputed first (the pin path reports a failure of that).
        if self.has_dependents.load(Ordering::Acquire) && self.release_dependents().is_err() {
            return None;
        }
        match &mut *lock(&self.state) {
            State::Host(data) => {
                self.clear_lineage();
                Some(f(&mut data.as_mut_slice()[..self.bytes as usize]))
            }
            State::Device(buffer) if self.pins.load(Ordering::Acquire) == 0 => self.ctx.host_ptr(buffer, self.bytes).map(|ptr| {
                self.invalidate_host_copy();
                self.clear_lineage();
                // SAFETY: as in `with_host_bytes`; the state lock and the absent pins make this the only access.
                f(unsafe { std::slice::from_raw_parts_mut(ptr, self.bytes as usize) })
            }),
            State::Device(_) | State::Empty | State::Window(_) | State::Lineage => None,
        }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        // The texture reads the buffer: it goes before the buffer does.
        self.texture.get_mut().unwrap_or_else(PoisonError::into_inner).take();
        self.invalidate_host_copy();
        if let Some(lineage) = self.lineage.get_mut().unwrap_or_else(PoisonError::into_inner).take() {
            self.ctx.pager().lineage_pinned_bytes.fetch_sub(lineage.held_bytes(), Ordering::Relaxed);
        }
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        match std::mem::replace(state, State::Empty) {
            State::Device(buffer) => {
                self.ctx.pager().resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
                self.ctx.recycle(buffer);
            }
            State::Host(data) => {
                self.ctx.pager().host_state_bytes.fetch_sub(data.len() as u64, Ordering::Relaxed);
            }
            State::Empty | State::Window(_) | State::Lineage => {}
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
        let class = size_class(bytes, self.max_buffer_bytes()).unwrap_or(bytes);
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
        // Whether recomputing the victim beats reading it back; looked at before its state is
        // locked, because it takes the state locks of the inputs.
        let candidate = victim.lineage_candidate(victim.has_copy.load(Ordering::Acquire));

        // Held to the end: a concurrent pin of the victim waits for the move to finish.
        let mut state = lock(&victim.state);
        let State::Device(buffer) = &*state else { return Ok(true) };
        if victim.pins.load(Ordering::Acquire) != 0 {
            return Ok(true);
        }
        let buffer = buffer.clone();
        let kept = if victim.has_copy.load(Ordering::Acquire) { lock(&victim.host_copy).take() } else { None };
        if let Some(inputs) = candidate
            && let Some(bytes) = victim.hold_lineage_inputs(inputs)
        {
            // A kept host copy is dropped: the segment is cheaper to recompute than to upload again.
            if let Some(copy) = &kept {
                victim.has_copy.store(false, Ordering::Release);
                pager.host_copy_bytes.fetch_sub(copy.len() as u64, Ordering::Relaxed);
            }
            // Nothing moves: the next pin runs the kernel again into a new buffer.
            pager.lineage_pinned_bytes.fetch_add(bytes, Ordering::Relaxed);
            pager.recompute_evictions.fetch_add(1, Ordering::Relaxed);
            *state = State::Lineage;
            lock(&victim.texture).take();
            pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
            lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&victim)));
            self.recycle(buffer);
            drop(state);
            return Ok(true);
        }
        let data = match kept {
            Some(data) => {
                // Nothing wrote the buffer since the copy was made: it is the data, no readback.
                victim.has_copy.store(false, Ordering::Release);
                pager.host_copy_bytes.fetch_sub(data.len() as u64, Ordering::Relaxed);
                #[cfg(debug_assertions)]
                if pager.clean_evictions.load(Ordering::Relaxed) % 16 == 0 {
                    let real = self.read_spill(&buffer, victim.bytes)?;
                    assert!(real.as_slice() == data.as_slice(), "a kept host copy differs from the device buffer: a write did not invalidate it");
                }
                pager.clean_evictions.fetch_add(1, Ordering::Relaxed);
                data
            }
            None => {
                // Flushes and waits, so every recorded use of the buffer has run.
                let data = self.read_spill(&buffer, victim.bytes)?;
                pager.readback_evictions.fetch_add(1, Ordering::Relaxed);
                data
            }
        };
        pager.host_state_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
        *state = State::Host(data);
        // The texture holds the buffer the data just left.
        lock(&victim.texture).take();
        pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        pager.spilled_bytes.fetch_add(victim.bytes, Ordering::Relaxed);
        lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&victim)));
        self.recycle(buffer);
        drop(state);
        self.enforce_host_budget();

        if !pager.warned.swap(true, Ordering::Relaxed) {
            if log::log_enabled!(log::Level::Warn) {
                log::warn!("{SPILL_WARNING}");
            } else {
                eprintln!("{SPILL_WARNING}");
            }
        }
        Ok(true)
    }

    /// Drops kept host copies, least recently pinned first, until the host holds no more than
    /// its budget; the segments in host memory are the only copy of their data and stay.
    fn enforce_host_budget(&self) {
        let pager = self.pager();
        if pager.host_bytes() <= pager.host_budget {
            return;
        }
        let mut holders: Vec<Arc<Segment>> = {
            let resident = lock(&pager.resident);
            resident.list.iter().filter_map(Weak::upgrade).filter(|segment| segment.has_copy.load(Ordering::Acquire)).collect()
        };
        holders.sort_by_key(|segment| segment.stamp.load(Ordering::Relaxed));
        for segment in holders {
            if pager.host_bytes() <= pager.host_budget {
                break;
            }
            segment.invalidate_host_copy();
        }
        if pager.host_state_bytes.load(Ordering::Relaxed) > pager.host_budget && !pager.host_warned.swap(true, Ordering::Relaxed) {
            let message = "luma: array data moved off the device exceeds the host budget (ContextOptions::host_budget / LUMA_HOST_BUDGET); it stays in host memory.";
            if log::log_enabled!(log::Level::Warn) {
                log::warn!("{message}");
            } else {
                eprintln!("{message}");
            }
        }
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

    /// How the context pages data between the device and host memory (see [`PagingStats`]).
    pub fn paging_stats(&self) -> PagingStats {
        let pager = self.pager();
        PagingStats {
            device_free_bytes: self.hal().memory_now().map(|now| now.free),
            host_copy_bytes: pager.host_copy_bytes.load(Ordering::Relaxed),
            clean_evictions: pager.clean_evictions.load(Ordering::Relaxed),
            readback_evictions: pager.readback_evictions.load(Ordering::Relaxed),
            recompute_evictions: pager.recompute_evictions.load(Ordering::Relaxed),
            lineage_pinned_bytes: pager.lineage_pinned_bytes.load(Ordering::Relaxed),
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

/// How the context pages array data between the device and host memory, from
/// [`Context::paging_stats`]; what [`MemoryStats`] does not say. (A separate type because
/// `MemoryStats` can be built by a struct literal and so cannot grow fields.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PagingStats {
    /// Bytes of device memory free right now, as the device reports them (everything on it, not
    /// only this context's); `None` where the backend cannot tell (wgpu, CPU).
    pub device_free_bytes: Option<u64>,
    /// Bytes of host copies kept of segments that are on the device (valid until the segment is
    /// written; evicting such a segment moves nothing).
    pub host_copy_bytes: u64,
    /// Evictions that needed no transfer because a valid host copy was kept.
    pub clean_evictions: u64,
    /// Evictions that read the buffer back.
    pub readback_evictions: u64,
    /// Evictions that moved nothing and read nothing back: the segment is a cheap expression of
    /// inputs that stay on the device, and its next use runs that expression again (see
    /// `jit::lineage`).
    pub recompute_evictions: u64,
    /// Bytes of arrays that segments evicted by recompute keep alive as their inputs (counted
    /// once per holder, so an input shared by several of them is counted several times). Not
    /// part of the memory budget; a value that stays high is a leak to look at.
    pub lineage_pinned_bytes: u64,
}

/// The default host budget: half of the available host memory, capped by the cgroup limit; 0
/// (no kept copies) where neither is known.
pub(crate) fn default_host_budget() -> u64 {
    let available = available_host_memory().unwrap_or(0);
    let limit = cgroup_limit().unwrap_or(u64::MAX);
    (available / 2).min(limit / 2)
}

#[cfg(target_os = "linux")]
fn available_host_memory() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(target_os = "macos")]
fn available_host_memory() -> Option<u64> {
    let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn available_host_memory() -> Option<u64> {
    None
}

/// The memory limit of the process's cgroup, if there is one.
#[cfg(target_os = "linux")]
fn cgroup_limit() -> Option<u64> {
    ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
        .iter()
        .find_map(|path| std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok())
}

#[cfg(not(target_os = "linux"))]
fn cgroup_limit() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_starts_at_three_quarters_of_the_budget() {
        let pager = Pager::new(Some(1000), 0);
        pager.resident_bytes.store(749, Ordering::Relaxed);
        assert!(!pager.under_pressure());
        pager.resident_bytes.store(750, Ordering::Relaxed);
        assert!(pager.under_pressure());
        assert!(!Pager::new(None, 0).under_pressure());
    }
}
