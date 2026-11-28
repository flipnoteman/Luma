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
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use luma_core::hal::{copy_parallel, copy_parallel_host, HostBytes};
use luma_core::Buffer;

use crate::array::Storage;
use crate::file::FileStore;
use crate::context::{Context, STORAGE_USAGE};
use crate::error::{LumaError, Result};
use crate::gpu::hash::{FxMap, FxSet};
use crate::jit::lineage::{Lineage, Recompute};
use crate::jit::split::Schedule;
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
    /// The bytes exist only at the segment's home, a range of a file (see [`Home`]); read at the
    /// next pin.
    Away,
}

/// The byte range of a file that holds a segment's data: where the data comes from when the
/// segment is first used, and where it goes when it is evicted dirty or flushed.
#[derive(Debug)]
pub(crate) struct Home {
    pub(crate) store: Arc<FileStore>,
    /// Byte offset of the segment's data in the file.
    pub(crate) offset: u64,
    /// Bytes of the file the segment covers: at most the segment's bytes; the rest (the tail of
    /// a packed segment that does not fill its last word) is zero and never written.
    pub(crate) file_bytes: u64,
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
    /// How many of them were taken by a may-write pin while a lineage read this segment. A pin that is held (a stepper's, a
    /// recording's, a captured graph's) can launch writes long after it was taken, which no
    /// later pin call announces: no lineage is used to evict over an input while one is held.
    write_pins: AtomicU32,
    /// Counts the write announcements ([`Segment::note_write`]): a lineage is not used to evict
    /// once an input's count differs from the one it recorded.
    epoch: AtomicU64,
    /// Value of the pager's clock at the last pin.
    stamp: AtomicU64,
    /// Where the data lives in a file, if the segment is file-backed.
    home: Option<Home>,
    /// Whether the home holds the current bytes (the segment is "clean"): true while `Away`,
    /// false from every may-write pin until the next write-back.
    home_valid: AtomicBool,
    /// Set when a may-write pin was taken of a segment whose file is read-only: the segment is
    /// anonymous from then on (copy on write), so that the file is never written.
    detached: AtomicBool,
    /// The texture of this segment's data made last (see [`crate::texture`]).
    texture: Mutex<Option<crate::texture::Cached>>,
    /// The schedule that holds a step writing this segment (see [`crate::jit::split`]): a cell next to `state`, not a variant of
    /// it, because an in-place step writes segments that are on the device or in host memory and must keep their state.
    planned: Mutex<Option<Weak<Schedule>>>,
    /// Whether `planned` is `Some` or `failed` is set: the one relaxed load the pin paths pay ([`Segment::settle`]).
    planned_flag: AtomicBool,
    /// Why the step that was to write this segment failed; every later use of the data fails with it.
    failed: OnceLock<Arc<LumaError>>,
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
    /// Whether this pin may be written through (see [`Segment::write_pins`]).
    write: bool,
}

impl Pin {
    /// The segment this pin keeps resident.
    pub(crate) fn segment(&self) -> &Arc<Segment> {
        &self.segment
    }

    /// Another pin of the same segment.
    pub(crate) fn dup(&self) -> Pin {
        self.segment.pins.fetch_add(1, Ordering::Acquire);
        if self.write {
            self.segment.write_pins.fetch_add(1, Ordering::Acquire);
        }
        Pin { segment: Arc::clone(&self.segment), buffer: self.buffer.clone(), write: self.write }
    }

    /// The segment's device buffer.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.segment.pins.fetch_sub(1, Ordering::Release);
        if self.write {
            self.segment.write_pins.fetch_sub(1, Ordering::Release);
        }
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

/// Orders eviction victims: smaller first, `None` keeps the segment.
type Rank<'a> = dyn Fn(&Segment) -> Option<(u8, u64)> + 'a;

/// Device memory a schedule's stage reserved (see [`Pager::reserve`]): counted as resident until dropped.
#[derive(Debug)]
pub(crate) struct Reservation {
    _counted: Accounted,
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
        // A file-backed segment is never dismantled here: its data may be dirty, and dropping the segment writes it back.
        if dead.contains(&key) || Arc::strong_count(segment) != held[&key] || segment.home.is_some() {
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
    /// Bytes of segments brought back from host memory to the device.
    uploaded_bytes: AtomicU64,
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
    /// Bytes of host memory the CPU backend may use for file-backed data (a CPU context has no
    /// device budget; anonymous arrays are not bounded).
    home_budget: u64,
    /// Sum of the sizes of the buffers of file-backed segments on the device.
    resident_homed_bytes: AtomicU64,
    /// Bytes read from files into segments.
    io_read_bytes: AtomicU64,
    /// Bytes written back to files.
    io_written_bytes: AtomicU64,
    /// Evictions of file-backed segments (and detachments from read-only files): the buffer is
    /// dropped, the file is the copy.
    home_drops: AtomicU64,
    /// Write-backs of dirty file-backed segments.
    home_writebacks: AtomicU64,
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
    /// Bytes brought back from host memory to the device, and bytes moved to host memory, so far (what a schedule's stats difference).
    pub(crate) fn traffic(&self) -> (u64, u64) {
        (self.uploaded_bytes.load(Ordering::Relaxed), self.spilled_bytes.load(Ordering::Relaxed))
    }

    /// Makes `bytes` of device memory available (evicting by `rank`, see [`Pager::eviction_rank`]) and counts them until the
    /// returned guard drops. `keep`: the segments the stage binds, which are never victims.
    pub(crate) fn reserve(
        &self,
        ctx: &Context,
        bytes: u64,
        keep: &FxSet<*const Segment>,
        next_use: &dyn Fn(&Segment) -> Option<u64>,
    ) -> Result<Reservation> {
        if let Some(budget) = self.budget {
            let rank = |segment: &Segment| Pager::eviction_rank(segment, keep, next_use);
            while self.resident_bytes.load(Ordering::Relaxed) + bytes > budget && ctx.evict_ranked(false, Some(&rank))? {
                std::hint::spin_loop();
            }
        }
        Ok(Reservation { _counted: Accounted::new(ctx, bytes) })
    }

    /// The order in which `reserve` gives up segments (Belady: the schedule knows the future): clean ones that are never used again,
    /// clean ones by farthest next use, then dirty ones the same way. A segment of `keep` is never a victim. Clean costs nothing to
    /// evict (a kept host copy, a valid file); dirty goes through the readback of `evict_one`.
    fn eviction_rank(segment: &Segment, keep: &FxSet<*const Segment>, next_use: &dyn Fn(&Segment) -> Option<u64>) -> Option<(u8, u64)> {
        if keep.contains(&std::ptr::from_ref(segment)) {
            return None;
        }
        let clean = segment.has_copy.load(Ordering::Acquire) || (segment.home().is_some() && segment.home_valid.load(Ordering::Acquire));
        let class = if clean { 0 } else { 2 };
        Some(match next_use(segment) {
            None => (class, 0),
            Some(next) => (class + 1, u64::MAX - next),
        })
    }

    /// The memory budget, if the context keeps one.
    pub(crate) fn budget(&self) -> Option<u64> {
        self.budget
    }

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
            uploaded_bytes: AtomicU64::new(0),
            host_budget,
            host_state_bytes: AtomicU64::new(0),
            host_copy_bytes: AtomicU64::new(0),
            clean_evictions: AtomicU64::new(0),
            readback_evictions: AtomicU64::new(0),
            recompute_evictions: AtomicU64::new(0),
            lineage_pinned_bytes: AtomicU64::new(0),
            #[cfg(test)]
            force_lineage: AtomicBool::new(false),
            home_budget: host_budget,
            resident_homed_bytes: AtomicU64::new(0),
            io_read_bytes: AtomicU64::new(0),
            io_written_bytes: AtomicU64::new(0),
            home_drops: AtomicU64::new(0),
            home_writebacks: AtomicU64::new(0),
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
        if segment.home.is_some() {
            self.resident_homed_bytes.fetch_add(size, Ordering::Relaxed);
        }
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
        // A texture of a planned segment would show data a pending step is about to replace; a failure is reported by the pin.
        let _ = self.settle();
        &self.texture
    }

    /// Commits the schedule that holds a step writing this segment (see [`crate::jit::split`]), if there is one: the data is what
    /// the step computes from then on. One relaxed load when nothing is planned. Fails if that step failed (now or earlier).
    /// Called before any lock of the segment is taken.
    #[inline]
    pub(crate) fn settle(&self) -> Result<()> {
        if !self.planned_flag.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.settle_planned()
    }

    #[cold]
    fn settle_planned(&self) -> Result<()> {
        let schedule = lock(&self.planned).as_ref().and_then(Weak::upgrade);
        if let Some(schedule) = schedule {
            schedule.settle()?;
        } else if self.failed.get().is_none() {
            // The schedule is gone without committing (it cannot be, but a stale cell must not stay).
            self.clear_planned();
        }
        match self.failed.get() {
            Some(error) => Err(LumaError::clone(error)),
            None => Ok(()),
        }
    }

    /// The error of the step that was to write this segment, if it failed.
    pub(crate) fn failure(&self) -> Option<Arc<LumaError>> {
        self.failed.get().cloned()
    }

    /// Whether a step that writes this segment is waiting in a schedule (or failed).
    pub(crate) fn is_planned(&self) -> bool {
        self.planned_flag.load(Ordering::Relaxed)
    }

    /// Records that a step of `schedule` writes this segment. A segment another schedule plans is settled first (that schedule commits).
    pub(crate) fn plan_for(&self, schedule: &Arc<Schedule>) -> Result<()> {
        loop {
            let other = {
                let mut cell = lock(&self.planned);
                match cell.as_ref().map(|weak| (weak.as_ptr(), weak.upgrade())) {
                    Some((at, _)) if std::ptr::eq(at, Arc::as_ptr(schedule)) => return Ok(()),
                    Some((_, Some(other))) if !other.is_finished() => other,
                    _ => {
                        *cell = Some(Arc::downgrade(schedule));
                        self.planned_flag.store(true, Ordering::Relaxed);
                        return Ok(());
                    }
                }
            };
            other.settle()?;
        }
    }

    /// Settles what a step that only reads this segment must not run ahead of: a write another schedule plans.
    pub(crate) fn settle_other(&self, schedule: &Arc<Schedule>) -> Result<()> {
        if !self.planned_flag.load(Ordering::Relaxed) {
            return Ok(());
        }
        let other = lock(&self.planned).as_ref().and_then(Weak::upgrade);
        match other {
            Some(other) if !Arc::ptr_eq(&other, schedule) => other.settle(),
            _ => Ok(()),
        }
    }

    /// Ends the plan of a committed step: the data is what the step wrote.
    pub(crate) fn clear_planned(&self) {
        *lock(&self.planned) = None;
        self.planned_flag.store(self.failed.get().is_some(), Ordering::Relaxed);
    }

    /// Marks the data as lost: the step that was to write it failed with `error`.
    pub(crate) fn poison(&self, error: &Arc<LumaError>) {
        let _ = self.failed.set(Arc::clone(error));
        *lock(&self.planned) = None;
        self.planned_flag.store(true, Ordering::Relaxed);
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
            write_pins: AtomicU32::new(0),
            epoch: AtomicU64::new(0),
            stamp: AtomicU64::new(0),
            home: None,
            home_valid: AtomicBool::new(false),
            detached: AtomicBool::new(false),
            texture: Mutex::new(None),
            planned: Mutex::new(None),
            planned_flag: AtomicBool::new(false),
            failed: OnceLock::new(),
        })
    }

    /// A segment of `bytes` bytes whose data is at `home`, not in memory yet: the file is read
    /// when the segment is first pinned.
    pub(crate) fn homed(ctx: &Context, bytes: u64, home: Home) -> Arc<Segment> {
        let mut segment = Segment {
            ctx: ctx.clone(),
            bytes,
            state: Mutex::new(State::Away),
            host_copy: Mutex::new(None),
            has_copy: AtomicBool::new(false),
            lineage: Mutex::new(None),
            has_lineage: AtomicBool::new(false),
            dependents: Mutex::new(Dependents::default()),
            has_dependents: AtomicBool::new(false),
            pins: AtomicU32::new(0),
            write_pins: AtomicU32::new(0),
            epoch: AtomicU64::new(0),
            stamp: AtomicU64::new(0),
            home: None,
            home_valid: AtomicBool::new(true),
            detached: AtomicBool::new(false),
            texture: Mutex::new(None),
            planned: Mutex::new(None),
            planned_flag: AtomicBool::new(false),
            failed: OnceLock::new(),
        };
        segment.home = Some(home);
        Arc::new(segment)
    }

    /// The home of the segment while it is file-backed: `None` for an anonymous segment and for
    /// one detached from a read-only file.
    pub(crate) fn home(&self) -> Option<&Home> {
        self.home.as_ref().filter(|_| !self.detached.load(Ordering::Acquire))
    }

    /// Stops counting a device buffer of `size` bytes of this segment as resident.
    fn uncount(&self, size: u64) {
        let pager = self.ctx.pager();
        pager.resident_bytes.fetch_sub(size, Ordering::Relaxed);
        if self.home.is_some() {
            pager.resident_homed_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    /// Notes a write to a homed segment: the home no longer holds the data, and a read-only
    /// file's segment lets go of it.
    fn home_written(&self) {
        let Some(home) = &self.home else { return };
        self.home_valid.store(false, Ordering::Release);
        if !home.store.writable() && !self.detached.swap(true, Ordering::AcqRel) {
            self.ctx.pager().home_drops.fetch_add(1, Ordering::Relaxed);
        }
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
        if self.home.is_some() || self.is_planned() {
            return false;
        }
        match &*lock(&self.state) {
            State::Device(_) => self.pins.load(Ordering::Acquire) == 0,
            State::Window(_) => true,
            State::Empty | State::Host(_) | State::Lineage | State::Away => false,
        }
    }

    /// Makes the segment a window of `pack` from byte `offset` (of another one, if it was
    /// one): the buffer it had is given back to the pool (commands recorded so far that use it
    /// still run first), and the segment shows the range of `pack` that holds the same data now.
    /// `false` if the segment has no device buffer, or one that a recording holds.
    pub(crate) fn adopt(self: &Arc<Self>, pack: Arc<Storage>, offset: u64) -> bool {
        if self.is_planned() {
            return false;
        }
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
        let pin = self.pin_unrecorded(true, false)?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    /// [`Segment::pin`] for a command that overwrites every byte of the data: a file-backed
    /// segment that is not in memory is given a buffer without reading its file.
    pub(crate) fn pin_overwrite(self: &Arc<Self>) -> Result<Pin> {
        let pin = self.pin_unrecorded(true, true)?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    /// [`Segment::pin`] for a command that only reads the buffer: a host copy stays valid.
    #[inline]
    pub(crate) fn pin_read(self: &Arc<Self>) -> Result<Pin> {
        let pin = self.pin_unrecorded(false, false)?;
        self.ctx.keep_pin(&pin);
        Ok(pin)
    }

    /// [`Segment::pin`] without keeping the pin for the recording (see `keep_pin`).
    pub(crate) fn pin_quiet(self: &Arc<Self>) -> Result<Pin> {
        self.pin_unrecorded(true, false)
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

    fn pin_unrecorded(self: &Arc<Self>, write: bool, overwrite: bool) -> Result<Pin> {
        self.settle()?;
        if !write {
            return self.pin_device(false, false);
        }
        self.note_write()?;
        let pin = self.pin_device(true, overwrite)?;
        if self.home.is_some() {
            self.home_written();
        }
        if self.has_lineage.load(Ordering::Acquire) {
            // The data is about to change: the pin keeps the segment resident, so it is not evicted by lineage meanwhile.
            self.clear_lineage();
        }
        Ok(pin)
    }

    /// The one announcement of a write to this segment's data, made before every may-write pin
    /// and host write (`Segment::pin`, `pin_quiet`, `with_host_bytes_mut`; every writer of a
    /// buffer gets it through one of them): counts the write, and settles the dependents (see
    /// [`Segment::release_dependents`]) while the data still is the old.
    fn note_write(self: &Arc<Self>) -> Result<()> {
        // Nothing reads this segment through a lineage: nothing to count or settle.
        if self.has_dependents.load(Ordering::Acquire) {
            self.epoch.fetch_add(1, Ordering::Release);
            self.release_dependents()?;
        }
        Ok(())
    }

    /// Whether any pin is held: it may be written through without another announcement, so no
    /// lineage is recorded over this segment then.
    pub(crate) fn is_pinned(&self) -> bool {
        self.pins.load(Ordering::Acquire) != 0
    }

    /// The count of write announcements so far.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Whether a may-write pin is held: whoever holds it may write without announcing it.
    pub(crate) fn write_held(&self) -> bool {
        self.write_pins.load(Ordering::Acquire) != 0
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
            let _pin = if matches!(&*lock(&self.state), State::Lineage) { Some(self.pin_device(false, false)?) } else { None };
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
    /// a readback and an upload (see [`crate::jit::lineage::Snapshot::evictable_inputs`]). Besides that, no
    /// segment that reads this one may be evicted by lineage itself, and (checked there) neither
    /// may an input: with both, a recomputation never reaches through more than one level.
    ///
    /// Takes the state locks of other segments, one at a time: the caller holds none.
    fn lineage_candidate(&self) -> Option<Box<[Arc<Storage>]>> {
        if !self.has_lineage.load(Ordering::Acquire) {
            return None;
        }
        let snapshot = lock(&self.lineage).as_ref()?.snapshot();
        let dependents: Vec<Arc<Segment>> = lock(&self.dependents).list.iter().filter_map(Weak::upgrade).collect();
        if dependents.iter().any(|dependent| matches!(&*lock(&dependent.state), State::Lineage)) {
            return None;
        }
        snapshot.evictable_inputs(&self.ctx, self.bytes)
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
    fn pin_device(self: &Arc<Self>, write: bool, overwrite: bool) -> Result<Pin> {
        let pager = self.ctx.pager();
        loop {
            if let State::Device(buffer) = &*lock(&self.state) {
                if write {
                    self.invalidate_host_copy();
                }
                self.pins.fetch_add(1, Ordering::Acquire);
                // Counted only while lineage reads this segment (one load otherwise); a pin held from before that is refused at record time (see `Segment::is_pinned`).
                let counted = write && self.has_dependents.load(Ordering::Acquire);
                if counted {
                    self.write_pins.fetch_add(1, Ordering::Acquire);
                }
                self.stamp.store(pager.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
                return Ok(Pin { segment: Arc::clone(self), buffer: buffer.clone(), write: counted });
            }
            let (evicted_by_lineage, away, host) = match &*lock(&self.state) {
                State::Lineage => (true, false, false),
                State::Away => (false, true, false),
                State::Host(_) => (false, false, true),
                _ => (false, false, false),
            };
            if evicted_by_lineage {
                self.ctx.abort_recording_for("recompute of an evicted segment")?;
            } else if away {
                self.ctx.abort_recording_for("file segment upload")?;
                // Aborting runs recorded commands, which may have pinned (and written) this very segment.
                if !matches!(&*lock(&self.state), State::Away) {
                    continue;
                }
            } else if host {
                self.ctx.abort_recording_for("host segment upload")?;
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
            if away && self.ctx.is_cpu() && pager.budget.is_none() {
                self.ctx.make_room_homed(self.bytes)?;
            }
            let buffer = match self.ctx.take_recorded(self.bytes, STORAGE_USAGE) {
                Some(buffer) => buffer,
                None => {
                    self.ctx.make_room(self.bytes)?;
                    self.ctx.alloc_buffer(self.bytes, STORAGE_USAGE)?
                }
            };
            if away && self.home().is_none() {
                self.ctx.recycle(buffer);
                return Err(LumaError::Gpu("a segment without a home lost its data".into()));
            }
            if away && let Some(home) = self.home() {
                // Nothing reads the file when the caller overwrites every byte of a writable file (the bytes of a read-only one are kept for the copy a write detaches).
                if !(overwrite && home.store.writable()) && let Err(e) = self.fill_from_home(home, &buffer) {
                    self.ctx.recycle(buffer);
                    return Err(e);
                }
            }
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
                pager.uploaded_bytes.fetch_add(len, Ordering::Relaxed);
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

    /// Fills `buffer` (idle, at least `self.bytes` bytes) with the data at `home`: read straight
    /// into it where the host can write the buffer (the CPU backend), through a host buffer
    /// and an upload elsewhere. The bytes past the file's are zero.
    fn fill_from_home(&self, home: &Home, buffer: &Buffer) -> Result<()> {
        let file_bytes = home.file_bytes as usize;
        let total = self.bytes as usize;
        if self.ctx.is_cpu()
            && !self.ctx.recording_here()
            && let Some(ptr) = self.ctx.host_ptr(buffer, self.bytes)
        {
            // SAFETY: `host_ptr` vouches for `self.bytes` bytes no command touches, and the buffer is not shared with anything yet.
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<MaybeUninit<u8>>(), total) };
            let (data, tail) = dst.split_at_mut(file_bytes);
            home.store.read(home.offset, data)?;
            tail.fill(MaybeUninit::new(0));
        } else {
            let mut data = HostBytes::zeroed(total);
            // SAFETY: initialized bytes viewed as `MaybeUninit<u8>`; the read only writes initialized bytes.
            let dst = unsafe { std::slice::from_raw_parts_mut(data.as_mut_slice().as_mut_ptr().cast::<MaybeUninit<u8>>(), file_bytes) };
            home.store.read(home.offset, dst)?;
            self.ctx.upload(buffer, data.as_slice())?;
        }
        self.ctx.pager().io_read_bytes.fetch_add(home.file_bytes, Ordering::Relaxed);
        Ok(())
    }

    /// Writes the data in `buffer` to the segment's home. The caller has waited for every
    /// command that writes the buffer (see [`Context::read_spill`]).
    fn write_home(&self, home: &Home, data: &HostBytes) -> Result<()> {
        home.store.write(home.offset, &data.as_slice()[..home.file_bytes as usize])?;
        let pager = self.ctx.pager();
        pager.io_written_bytes.fetch_add(home.file_bytes, Ordering::Relaxed);
        pager.home_writebacks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Writes the data of a segment that is on the device and dirty to its (writable) home and
    /// marks it clean; the buffer stays. Does nothing for a clean segment or one that is not on
    /// the device.
    pub(crate) fn write_back(&self) -> Result<()> {
        self.settle()?;
        let Some(home) = self.home() else { return Ok(()) };
        if self.home_valid.load(Ordering::Acquire) {
            return Ok(());
        }
        // Before the state lock: aborting a recording can pin this segment.
        self.ctx.abort_recording_for("write back of a file segment")?;
        let state = lock(&self.state);
        let State::Device(buffer) = &*state else { return Ok(()) };
        if self.home_valid.load(Ordering::Acquire) {
            return Ok(());
        }
        let data = self.ctx.read_spill(buffer, self.bytes)?;
        self.write_home(home, &data)?;
        self.home_valid.store(true, Ordering::Release);
        Ok(())
    }

    /// Copies the segment's data into `dst` (`dst.len()` bytes), after everything recorded so
    /// far. A segment in host memory is copied from there and stays there.
    pub(crate) fn read_into(self: &Arc<Self>, dst: &mut [MaybeUninit<u8>]) -> Result<()> {
        self.settle()?;
        if matches!(&*lock(&self.state), State::Away)
            && let Some(home) = self.home()
        {
            // Read from the file: no buffer, no device round trip.
            let (data, tail) = dst.split_at_mut(dst.len().min(home.file_bytes as usize));
            home.store.read(home.offset, data)?;
            tail.fill(MaybeUninit::new(0));
            self.ctx.pager().io_read_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
            return Ok(());
        }
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
        // A planned segment holds stale bytes: the caller falls into the pin path, which commits.
        if self.is_planned() || self.ctx.recording_here() {
            return None;
        }
        match &*lock(&self.state) {
            State::Host(data) => Some(f(&data.as_slice()[..self.bytes as usize])),
            State::Device(buffer) if self.pins.load(Ordering::Acquire) == 0 => self.ctx.host_ptr(buffer, self.bytes).map(|ptr| {
                // SAFETY: `host_ptr` vouches for `self.bytes` readable bytes no GPU command writes, and the state lock keeps the buffer from being evicted.
                f(unsafe { std::slice::from_raw_parts(ptr, self.bytes as usize) })
            }),
            State::Device(_) | State::Empty | State::Window(_) | State::Lineage | State::Away => None,
        }
    }

    /// [`Segment::with_host_bytes`] with write access; a device buffer must not be pinned by a recording.
    pub(crate) fn with_host_bytes_mut<R>(self: &Arc<Self>, f: impl FnOnce(&mut [u8]) -> R) -> Option<R> {
        if self.is_planned() || self.ctx.recording_here() {
            return None;
        }
        // A write: what is evicted by lineage over this data is recomputed first (the pin path reports a failure of that).
        if self.note_write().is_err() {
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
                self.home_written();
                // SAFETY: as in `with_host_bytes`; the state lock and the absent pins make this the only access.
                f(unsafe { std::slice::from_raw_parts_mut(ptr, self.bytes as usize) })
            }),
            State::Device(_) | State::Empty | State::Window(_) | State::Lineage | State::Away => None,
        }
    }
}

impl Segment {
    /// What dropping a file-backed segment that is on the device does besides recycling the buffer:
    /// the last dirty copy of the data goes to the file first (a failure is logged and kept for `flush`).
    #[cold]
    #[inline(never)]
    fn drop_homed(&self, buffer: &Buffer) {
        if let Some(home) = self.home()
            && !self.home_valid.load(Ordering::Acquire)
        {
            let written = self.ctx.read_spill(buffer, self.bytes).and_then(|data| self.write_home(home, &data));
            if let Err(error) = written {
                home.store.failed_with(&error);
                log::error!("luma: could not write back a dropped file segment: {error}");
            }
        }
        self.ctx.pager().resident_homed_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
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
                if self.home.is_some() {
                    self.drop_homed(&buffer);
                }
                self.ctx.pager().resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
                self.ctx.recycle(buffer);
            }
            State::Host(data) => {
                self.ctx.pager().host_state_bytes.fetch_sub(data.len() as u64, Ordering::Relaxed);
            }
            State::Empty | State::Window(_) | State::Lineage | State::Away => {}
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

    /// On the CPU backend, evicts file-backed segments (least recently pinned first) until
    /// `bytes` more of them fit in the host budget for file data. Anonymous arrays are not
    /// bounded there.
    pub(crate) fn make_room_homed(&self, bytes: u64) -> Result<()> {
        let pager = self.pager();
        let class = size_class(bytes, self.max_buffer_bytes()).unwrap_or(bytes);
        while pager.resident_homed_bytes.load(Ordering::Relaxed) + class > pager.home_budget && self.evict_one_homed()? {
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Moves the least recently pinned unpinned segment to host memory. `Ok(false)`: nothing to
    /// move. `Ok(true)`: a segment moved, or the choice raced with a pin; the caller retries.
    pub(crate) fn evict_one(&self) -> Result<bool> {
        self.evict_one_in(false)
    }

    /// [`Context::evict_one`] for file-backed segments only.
    fn evict_one_homed(&self) -> Result<bool> {
        self.evict_one_in(true)
    }

    fn evict_one_in(&self, homed_only: bool) -> Result<bool> {
        self.evict_ranked(homed_only, None)
    }

    /// [`Context::evict_one`] choosing the victim by `rank` (smallest first; `None` keeps a segment) instead of by recency.
    fn evict_ranked(&self, homed_only: bool, rank: Option<&Rank<'_>>) -> Result<bool> {
        if EVICTING.with(Cell::get) {
            return Ok(false);
        }
        // Before any lock: aborting a recording can pin (and so lock) the victim's segment.
        self.abort_recording_for("eviction under memory pressure")?;
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
            let candidates = resident
                .list
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|segment| segment.pins.load(Ordering::Acquire) == 0 && (!homed_only || segment.home().is_some()));
            match rank {
                None => candidates.min_by_key(|segment| segment.stamp.load(Ordering::Relaxed)),
                Some(rank) => candidates.filter_map(|segment| rank(&segment).map(|rank| (rank, segment))).min_by_key(|(rank, _)| *rank).map(|(_, segment)| segment),
            }
        };
        let Some(victim) = victim else { return Ok(false) };
        // Whether recomputing the victim beats reading it back; looked at before its state is
        // locked, because it takes the state locks of the inputs.
        let candidate = if victim.has_copy.load(Ordering::Acquire) { None } else { victim.lineage_candidate() };

        // Held to the end: a concurrent pin of the victim waits for the move to finish.
        let mut state = lock(&victim.state);
        let State::Device(buffer) = &*state else { return Ok(true) };
        if victim.pins.load(Ordering::Acquire) != 0 {
            return Ok(true);
        }
        let buffer = buffer.clone();
        if let Some(home) = victim.home() {
            // The file is the copy: a clean segment is dropped, a dirty one is written back first.
            if victim.home_valid.load(Ordering::Acquire) {
                #[cfg(debug_assertions)]
                if pager.clean_evictions.load(Ordering::Relaxed).is_multiple_of(16) {
                    let real = self.read_spill(&buffer, victim.bytes)?;
                    let mut file = vec![std::mem::MaybeUninit::<u8>::uninit(); home.file_bytes as usize];
                    home.store.read(home.offset, &mut file)?;
                    // SAFETY: `read` initialized every byte.
                    let file: &[u8] = unsafe { std::slice::from_raw_parts(file.as_ptr().cast::<u8>(), file.len()) };
                    assert!(file == &real.as_slice()[..file.len()], "a clean file segment differs from its file: a write did not mark it dirty");
                }
                pager.clean_evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                // Flushes and waits, so every recorded use of the buffer has run.
                let data = self.read_spill(&buffer, victim.bytes)?;
                victim.write_home(home, &data)?;
                victim.home_valid.store(true, Ordering::Release);
            }
            if home.store.drop_cache() {
                home.store.advise_drop(home.offset, home.file_bytes);
            }
            pager.home_drops.fetch_add(1, Ordering::Relaxed);
            *state = State::Away;
            lock(&victim.texture).take();
            victim.uncount(buffer.size());
            lock(&pager.resident).list.retain(|weak| !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&victim)));
            self.recycle(buffer);
            drop(state);
            return Ok(true);
        }
        let kept = if victim.has_copy.load(Ordering::Acquire) { lock(&victim.host_copy).take() } else { None };
        if kept.is_none()
            && let Some(inputs) = candidate
            && let Some(bytes) = victim.hold_lineage_inputs(inputs)
        {
            // Nothing moves: the next pin runs the kernel again into a new buffer.
            pager.lineage_pinned_bytes.fetch_add(bytes, Ordering::Relaxed);
            pager.recompute_evictions.fetch_add(1, Ordering::Relaxed);
            *state = State::Lineage;
            lock(&victim.texture).take();
            victim.uncount(buffer.size());
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
                if pager.clean_evictions.load(Ordering::Relaxed).is_multiple_of(16) {
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
        victim.uncount(buffer.size());
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
            io_read_bytes: pager.io_read_bytes.load(Ordering::Relaxed),
            uploaded_bytes: pager.uploaded_bytes.load(Ordering::Relaxed),
            io_written_bytes: pager.io_written_bytes.load(Ordering::Relaxed),
            home_drops: pager.home_drops.load(Ordering::Relaxed),
            home_writebacks: pager.home_writebacks.load(Ordering::Relaxed),
            file_resident_bytes: pager.resident_homed_bytes.load(Ordering::Relaxed),
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
    /// Bytes read from files into segments of file-backed arrays (see [`Array::open`](crate::Array::open)).
    pub io_read_bytes: u64,
    /// Bytes of segments brought back from host memory to the device (arrays that were moved off it to make room and used again).
    pub uploaded_bytes: u64,
    /// Bytes written back to files, at eviction, `flush` and drop.
    pub io_written_bytes: u64,
    /// Buffers of file-backed segments dropped because the file holds their data: evictions of
    /// clean and of written-back segments, and segments of a read-only file that a write made
    /// anonymous (copy on write).
    pub home_drops: u64,
    /// Write-backs of dirty file-backed segments to their files.
    pub home_writebacks: u64,
    /// Bytes of device buffers that file-backed segments hold now.
    pub file_resident_bytes: u64,
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

    /// Three one-segment arrays on the device under a budget that holds three; `reserve` of two segments' worth evicts one.
    fn reserve_victim(next: [Option<u64>; 3], keep: &[usize]) -> usize {
        let options = crate::ContextOptions::from_env().memory_budget(4096).chunk_bytes(1024);
        let ctx = Context::with_options(options).expect("a device is available");
        let arrays: Vec<crate::Array<f32>> = (0..3).map(|k| crate::Array::from_slice_on(&ctx, &[k as f32; 256], 256).unwrap()).collect();
        let segments: Vec<Arc<Segment>> = arrays.iter().map(|a| Arc::clone(a.recorded_storage().unwrap().only_segment())).collect();
        assert!(segments.iter().all(|s| s.on_device()));
        let keep: FxSet<*const Segment> = keep.iter().map(|&k| Arc::as_ptr(&segments[k])).collect();
        let next_use = |segment: &Segment| segments.iter().position(|s| std::ptr::eq(Arc::as_ptr(s), segment)).and_then(|k| next[k]);
        let reservation = ctx.pager().reserve(&ctx, 2048, &keep, &next_use).unwrap();
        let evicted: Vec<usize> = (0..3).filter(|&k| !segments[k].on_device()).collect();
        drop(reservation);
        assert_eq!(evicted.len(), 1, "exactly one segment makes room: {evicted:?}");
        evicted[0]
    }

    #[test]
    fn reserve_evicts_the_farthest_next_use_first() {
        // All used again: the one used last goes.
        assert_eq!(reserve_victim([Some(5), Some(9), Some(2)], &[]), 1);
        // One never used again goes before any that is.
        assert_eq!(reserve_victim([Some(5), Some(9), None], &[]), 2);
        // The segments the stage binds are never victims.
        assert_eq!(reserve_victim([Some(5), Some(9), None], &[1, 2]), 0);
    }
}
