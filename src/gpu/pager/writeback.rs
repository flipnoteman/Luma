//! Write-back of dirty file segments in the background.
//!
//! A dirty segment leaves the device in three places: [`Segment::write_back`] (flush), the
//! eviction of a dirty victim and the drop of a segment. Each starts a [`WriteJob`] instead of
//! reading the buffer back and writing the file one after the other: the readback `R` is a copy on
//! the transfer lane into a stage of the pager's write ring, which starts when the compute work
//! recorded so far has run; once it completed, the file write `W` goes to the I/O pool. So the
//! readback of segment i overlaps the compute of i+1 and the file write of i-1.
//!
//! Every file keeps its jobs (`FileStore::pending`); whatever reads the file, and flush, close and
//! the drop of an array, wait for the jobs that overlap what they touch. The pager reaps finished
//! jobs lazily (on `make_room`, at pins and at flushes), so no thread is needed besides the pool.
//! A job that failed puts its data back (see [`WriteJob::failed`]): the segment is dirty again
//! and the error comes out at the next call that waits.

use luma_io::IoTicket;

use super::*;

/// How far a job is advanced.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Goal {
    /// The readback completed (so the device buffer is free) and the file write started.
    Read,
    /// The file write completed.
    Done,
}

/// What the segment does with the device buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// The segment keeps its buffer ([`Segment::write_back`]); it is clean once the write landed.
    Flush,
    /// The buffer was the segment's and goes back to the pool once it was read: an evicted
    /// segment (now `Away`) or a dropped one.
    Evict,
}

enum Phase {
    Reading { stage: Stage, ticket: Ticket, buffer: Option<Buffer> },
    Writing { stage: Stage, io: IoTicket },
    Done(Option<LumaError>),
}

/// A dirty segment on its way to its file.
pub(crate) struct WriteJob {
    ctx: Context,
    store: Arc<FileStore>,
    offset: u64,
    file_bytes: u64,
    /// Bytes of the segment's data.
    bytes: u64,
    /// Bytes the readback copies (the file's, rounded up to a word).
    copied: u64,
    /// The segment, unless it was dropped; it gets its data back if the write fails.
    segment: Option<Arc<Segment>>,
    kind: Kind,
    /// Tell the page cache to forget the bytes once written.
    advise: bool,
    /// Whether the device buffer is still the job's, counted as resident.
    holds_buffer: AtomicBool,
    finished: AtomicBool,
    inner: Mutex<Phase>,
}

impl std::fmt::Debug for WriteJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteJob").field("offset", &self.offset).field("kind", &self.kind).field("finished", &self.finished.load(Ordering::Relaxed)).finish_non_exhaustive()
    }
}

impl WriteJob {
    /// Whether the job overlaps the file range `[offset, offset + len)`.
    pub(crate) fn overlaps(&self, offset: u64, len: u64) -> bool {
        self.offset < offset.saturating_add(len) && offset < self.offset + self.file_bytes
    }

    /// Whether the job ended, in success or not.
    pub(crate) fn is_done(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Moves the job on towards `goal`, waiting for it when `block`. `Ok(true)`: the goal is
    /// reached; `Ok(false)`: not yet (only without `block`); `Err`: the job failed, its data is
    /// back at the segment.
    pub(crate) fn advance(&self, block: bool, goal: Goal) -> Result<bool> {
        let mut phase = lock(&self.inner);
        loop {
            match std::mem::replace(&mut *phase, Phase::Done(None)) {
                Phase::Done(None) => return Ok(true),
                Phase::Done(Some(error)) => {
                    *phase = Phase::Done(Some(error.clone()));
                    return Err(error);
                }
                Phase::Reading { stage, ticket, buffer } => {
                    if !block && !self.ctx.ticket_done(&ticket) {
                        *phase = Phase::Reading { stage, ticket, buffer };
                        return Ok(false);
                    }
                    let waited = self.ctx.ticket_wait(&ticket).and_then(|()| {
                        // A readback whose mapping failed shows no bytes.
                        if stage.as_slice().len() < self.copied as usize { Err(LumaError::Gpu("the readback of a file segment failed".into())) } else { Ok(()) }
                    });
                    if let Err(error) = waited {
                        let error = self.read_failed(buffer, stage, error);
                        *phase = Phase::Done(Some(error.clone()));
                        return Err(error);
                    }
                    self.release_buffer(buffer);
                    self.settled();
                    // SAFETY: the stage holds `file_bytes` read bytes; it stays in the phase (at a fixed address) until the ticket completes, and nothing writes it meanwhile.
                    let io = unsafe { self.store.start_write(self.offset, stage.as_slice().as_ptr(), self.file_bytes as usize) };
                    *phase = Phase::Writing { stage, io };
                    if goal == Goal::Read {
                        return Ok(true);
                    }
                }
                Phase::Writing { stage, io } => {
                    if goal == Goal::Read {
                        *phase = Phase::Writing { stage, io };
                        return Ok(true);
                    }
                    let result = if block { Some(self.store.wait_io(&io)) } else { self.store.poll_io(&io) };
                    let Some(result) = result else {
                        *phase = Phase::Writing { stage, io };
                        return Ok(false);
                    };
                    match result {
                        Ok(_) => {
                            self.written(stage);
                            self.finish();
                            return Ok(true);
                        }
                        Err(e) => {
                            let error = self.store.write_error(self.offset, e);
                            self.failed(Some(&stage), &error);
                            self.ctx.pager().give_back_stage(stage);
                            *phase = Phase::Done(Some(error.clone()));
                            self.finish();
                            return Err(error);
                        }
                    }
                }
            }
        }
    }

    /// The segment's buffer may be written again.
    fn settled(&self) {
        if let Some(home) = self.segment.as_ref().and_then(|segment| segment.home.as_ref()) {
            home.flushing.store(false, Ordering::Release);
        }
    }

    /// The readback is done with the buffer: it leaves the resident count and goes to the pool.
    fn release_buffer(&self, buffer: Option<Buffer>) {
        let Some(buffer) = buffer else { return };
        let pager = self.ctx.pager();
        pager.resident_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        pager.resident_homed_bytes.fetch_sub(buffer.size(), Ordering::Relaxed);
        self.holds_buffer.store(false, Ordering::Release);
        self.ctx.recycle(buffer);
    }

    /// The readback failed: the device buffer still holds the data, read it by the slow way.
    fn read_failed(&self, buffer: Option<Buffer>, stage: Stage, error: LumaError) -> LumaError {
        let data = buffer.as_ref().and_then(|buffer| self.ctx.read_spill(buffer, self.bytes).ok());
        self.release_buffer(buffer);
        self.settled();
        self.failed_with_data(data, &error);
        self.ctx.pager().give_back_stage(stage);
        self.finish();
        error
    }

    /// The file write landed.
    fn written(&self, stage: Stage) {
        let pager = self.ctx.pager();
        pager.io_written_bytes.fetch_add(self.file_bytes, Ordering::Relaxed);
        pager.home_writebacks.fetch_add(1, Ordering::Relaxed);
        if self.advise && self.store.drop_cache() {
            self.store.advise_drop(self.offset, self.file_bytes);
        }
        pager.give_back_stage(stage);
    }

    /// The write failed: the stage holds the data, keep it.
    fn failed(&self, stage: Option<&Stage>, error: &LumaError) {
        let data = stage.map(|stage| {
            let mut data = HostBytes::zeroed(self.bytes as usize);
            let len = (self.copied as usize).min(stage.as_slice().len());
            data.as_mut_slice()[..len].copy_from_slice(&stage.as_slice()[..len]);
            data
        });
        self.failed_with_data(data, error);
    }

    /// Records the error, makes the segment dirty again and puts `data` back as its host copy
    /// (when the segment was evicted and nothing wrote it since).
    fn failed_with_data(&self, data: Option<HostBytes>, error: &LumaError) {
        self.store.failed_with(error);
        let Some(segment) = &self.segment else {
            log::error!("luma: could not write back a dropped file segment: {error}");
            return;
        };
        segment.home_valid.store(false, Ordering::Release);
        if self.kind != Kind::Evict {
            return;
        }
        let Some(data) = data else { return };
        // A thread that holds the segment's state is evicting or pinning it: it lets go within moments, unless it is this thread, which then gives up (the segment is not `Away` in that case, or about to be written whole).
        for _ in 0..64 {
            match segment.state.try_lock() {
                Ok(mut state) => {
                    if matches!(&*state, State::Away) {
                        self.ctx.pager().host_state_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
                        *state = State::Host(data);
                    }
                    return;
                }
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    let mut state = poisoned.into_inner();
                    if matches!(&*state, State::Away) {
                        self.ctx.pager().host_state_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
                        *state = State::Host(data);
                    }
                    return;
                }
                Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
            }
        }
        log::error!("luma: the data of a failed write-back could not be put back: {error}");
    }

    fn finish(&self) {
        if !self.finished.swap(true, Ordering::AcqRel) {
            self.ctx.pager().writes_pending.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for WriteJob {
    fn drop(&mut self) {
        // A write is never dropped: the engine reads the stage, the transfer lane writes it.
        let phase = std::mem::replace(self.inner.get_mut().unwrap_or_else(PoisonError::into_inner), Phase::Done(None));
        match phase {
            Phase::Reading { ticket, buffer, .. } => {
                let _ = self.ctx.ticket_wait(&ticket);
                self.release_buffer(buffer);
            }
            Phase::Writing { io, .. } => {
                let _ = self.store.wait_io(&io);
            }
            Phase::Done(_) => {}
        }
        self.finish();
    }
}

/// The write stages (`StageDir::FromDevice`) not in use by a job.
#[derive(Debug, Default)]
pub(super) struct OutRing {
    idle: Vec<Stage>,
}

impl Pager {
    /// Bytes of write stages that may exist at once: what the read ring may hold.
    fn write_budget(&self, segment: u64) -> u64 {
        self.readahead_budget(segment) + 2 * segment
    }

    /// A finished job gives its stage back to the ring.
    fn give_back_stage(&self, stage: Stage) {
        lock(&self.wring).idle.push(stage);
    }
}

impl Context {
    /// Whether dirty file segments leave the device in the background: a device backend with host
    /// staging memory, with read-ahead on (`LUMA_READAHEAD=0` keeps the synchronous path).
    pub(super) fn async_writes(&self) -> bool {
        !self.is_cpu() && self.pager().readahead_depth.load(Ordering::Relaxed) > 0 && self.hal().pinned_staging()
    }

    /// A write stage of at least `bytes` bytes: an idle one, a new one while the ring is within
    /// its budget, else the oldest write's after it completed.
    fn write_stage(&self, bytes: u64) -> Result<Stage> {
        let pager = self.pager();
        loop {
            let waiting = pager.writes_pending.load(Ordering::Relaxed) > 0;
            {
                let mut ring = lock(&pager.wring);
                let fit = ring.idle.iter().enumerate().filter(|(_, stage)| stage.len() >= bytes).min_by_key(|(_, stage)| stage.len()).map(|(at, _)| at);
                if let Some(at) = fit {
                    return Ok(ring.idle.swap_remove(at));
                }
                let held = pager.wstage_bytes.load(Ordering::Relaxed);
                if held + bytes <= pager.write_budget(bytes) || !waiting {
                    // Stages too small for this segment (a short tail's) are of no use any more.
                    let too_small: u64 = std::mem::take(&mut ring.idle).iter().map(Stage::len).sum();
                    let now = pager.wstage_bytes.fetch_sub(too_small, Ordering::Relaxed) - too_small + bytes;
                    pager.wstage_bytes.fetch_add(bytes, Ordering::Relaxed);
                    pager.wstage_peak.fetch_max(now, Ordering::Relaxed);
                    drop(ring);
                    return match self.readback_stage(bytes) {
                        Some(stage) => Ok(stage),
                        None => {
                            pager.wstage_bytes.fetch_sub(bytes, Ordering::Relaxed);
                            Err(LumaError::Gpu("no host staging memory for a write-back".into()))
                        }
                    };
                }
            }
            if !self.wait_oldest_write() {
                // Nothing in flight: the stages are all idle ones that are too small.
                lock(&pager.wring).idle.clear();
                pager.wstage_bytes.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Starts writing the data of `buffer` (the first `bytes` bytes, of a segment whose file range
    /// is `home`) to its file in the background. The buffer must stay unwritten until the readback
    /// ran: an `Evict` job takes it over (a handle of it, whose bytes stay counted as resident
    /// until then).
    pub(super) fn start_write(&self, segment: Option<Arc<Segment>>, home: &Home, bytes: u64, buffer: &Buffer, kind: Kind, advise: bool) -> Result<Arc<WriteJob>> {
        if home.file_bytes == 0 {
            return Err(LumaError::Gpu("an empty write-back".into()));
        }
        // Writes of this range that are still going on land first: the file ends with the newest bytes.
        home.store.drain_overlap(home.offset, home.file_bytes)?;
        let copied = home.file_bytes.next_multiple_of(4).min(bytes);
        let mut stage = self.write_stage(bytes)?;
        let ticket = match self.readback_staged(buffer, &mut stage, copied) {
            Ok(ticket) => ticket,
            Err(error) => {
                self.pager().give_back_stage(stage);
                return Err(error);
            }
        };
        let pager = self.pager();
        let owned = (kind == Kind::Evict).then(|| buffer.clone());
        let job = Arc::new(WriteJob {
            ctx: self.clone(),
            store: Arc::clone(&home.store),
            offset: home.offset,
            file_bytes: home.file_bytes,
            bytes,
            copied,
            segment,
            kind,
            advise,
            holds_buffer: AtomicBool::new(owned.is_some()),
            finished: AtomicBool::new(false),
            inner: Mutex::new(Phase::Reading { stage, ticket, buffer: owned }),
        });
        pager.writes_pending.fetch_add(1, Ordering::Relaxed);
        lock(&pager.writes).push_back(Arc::downgrade(&job));
        home.store.push_pending(Arc::clone(&job));
        Ok(job)
    }

    /// Advances the jobs that can move without waiting, and forgets the finished ones.
    pub(crate) fn reap_writes(&self) {
        let pager = self.pager();
        if pager.writes_pending.load(Ordering::Relaxed) == 0 {
            return;
        }
        let jobs: Vec<Arc<WriteJob>> = lock(&pager.writes).iter().filter_map(Weak::upgrade).filter(|job| !job.is_done()).collect();
        for job in jobs {
            // A failure is recorded at the file and surfaces at its next waiting call.
            let _ = job.advance(false, Goal::Done);
        }
        lock(&pager.writes).retain(|weak| weak.upgrade().is_some_and(|job| !job.is_done()));
    }

    /// Waits for the oldest job that is not done to complete; `false` if there is none.
    fn wait_oldest_write(&self) -> bool {
        let pager = self.pager();
        let job = {
            let mut writes = lock(&pager.writes);
            loop {
                match writes.front().map(Weak::upgrade) {
                    None => return false,
                    Some(Some(job)) if !job.is_done() => break job,
                    Some(_) => {
                        writes.pop_front();
                    }
                }
            }
        };
        let _ = job.advance(true, Goal::Done);
        true
    }

    /// Waits for the oldest readback that still holds a device buffer, which frees it; `false` if
    /// there is none (or this thread is evicting, which must not wait).
    pub(crate) fn wait_one_readback(&self) -> bool {
        if EVICTING.with(Cell::get) {
            return false;
        }
        let pager = self.pager();
        if pager.writes_pending.load(Ordering::Relaxed) == 0 {
            return false;
        }
        let job = lock(&pager.writes).iter().filter_map(Weak::upgrade).find(|job| job.holds_buffer.load(Ordering::Acquire) && !job.is_done());
        let Some(job) = job else { return false };
        let _ = job.advance(true, Goal::Read);
        true
    }
}

impl Segment {
    /// Whether a write-back that [`Segment::write_back`] started may still be reading the buffer.
    pub(super) fn flushing(&self) -> bool {
        self.home.as_ref().is_some_and(|home| home.flushing.load(Ordering::Acquire))
    }

    /// Waits for the readback of a write-back that [`Segment::write_back`] started: the buffer
    /// may be written again after it.
    pub(super) fn settle_flush(&self) {
        let Some(home) = &self.home else { return };
        let job = lock(&home.flush_job).upgrade();
        if let Some(job) = job {
            let _ = job.advance(true, Goal::Read);
        }
        home.flushing.store(false, Ordering::Release);
    }
}
