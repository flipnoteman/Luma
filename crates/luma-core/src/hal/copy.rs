//! Host memory copies shared by every backend's upload and readback paths.

use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use super::topology;

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 2 << 20;

/// Smallest share of a parallel copy, so that a copy just above [`PARALLEL_COPY_MIN`] is not
/// cut into pieces that cost more to hand out than to copy.
const PART_MIN: usize = 512 << 10;

/// Most threads (the caller included) a copy between host buffers is spread over.
const MAX_THREADS: usize = 16;

/// Most threads of a copy that stages a transfer to or from a device, and the number of
/// threads of the pool where the number of physical cores is not known.
const STAGING_THREADS: usize = 8;

/// The unit threads take from a copy between host buffers (see [`copy_parallel_host`]).
const CHUNK: usize = 256 << 10;

/// How long an idle worker polls for the next share before it parks.
const WORKER_SPIN: Duration = Duration::from_micros(200);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The shares of one copy still to be copied by workers, or the chunks of a copy between host buffers
/// (see [`copy_parallel_host`]) that threads take from `next` until it is all taken.
struct Job {
    remaining: AtomicUsize,
    waiter: Thread,
    src: *const u8,
    dst: *mut MaybeUninit<u8>,
    len: usize,
    /// Offset of the next chunk to take.
    next: AtomicUsize,
}

impl Job {
    /// Copies chunks of [`CHUNK`] bytes until none is left.
    ///
    /// # Safety
    /// `src` and `dst` must be valid for `len` bytes (and distinct) for the whole call.
    unsafe fn copy_chunks(&self) {
        loop {
            let at = self.next.fetch_add(CHUNK, Ordering::Relaxed);
            if at >= self.len {
                return;
            }
            let len = CHUNK.min(self.len - at);
            // SAFETY: the chunk lies within the buffers; chunks are handed out once, so no two threads
            // copy the same bytes.
            unsafe { copy_streaming(std::slice::from_raw_parts(self.src.add(at), len), std::slice::from_raw_parts_mut(self.dst.add(at), len)) };
        }
    }
}

/// One share of a parallel copy, waiting for a worker; a share of no bytes is a ticket to take chunks.
struct Part {
    src: *const u8,
    dst: *mut MaybeUninit<u8>,
    len: usize,
    job: *const Job,
}

// SAFETY: the pointers stay valid until the share has been counted in `Job::remaining`: the
// thread that created the share does not return (and so does not free the buffers or the job)
// before that.
unsafe impl Send for Part {}

impl Part {
    /// Copies the share (or takes chunks), then tells its caller.
    fn run(self) {
        // SAFETY: `src` and `dst` are `len` bytes of two distinct buffers (see `Send` above); the
        // shares of a call do not overlap.
        unsafe {
            let job = &*self.job;
            if self.len == 0 {
                job.copy_chunks();
            } else {
                copy_streaming(std::slice::from_raw_parts(self.src, self.len), std::slice::from_raw_parts_mut(self.dst, self.len));
            }
            // The waiter is cloned before the count drops: the job lives on the caller's stack,
            // which is free to go once the count reaches zero.
            let waiter = job.waiter.clone();
            if job.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                waiter.unpark();
            }
        }
    }
}

/// Threads that stay ready between copies: starting one costs as much as copying a megabyte
/// or two, and waking a parked one 50 us on a busy machine, while a copy that large is meant to
/// take a few hundred microseconds. A worker polls for [`WORKER_SPIN`] after a share before it
/// parks, which covers the gap between the pieces of a transfer.
struct Pool {
    queue: Mutex<VecDeque<Part>>,
    ready: Condvar,
    /// Shares in `queue`, for the workers that poll.
    queued: AtomicUsize,
    /// Workers parked on `ready`; only counted and changed under the `queue` lock.
    sleeping: AtomicUsize,
    /// Threads a copy is spread over, the caller included.
    threads: usize,
}

impl Pool {
    /// The next share: polls for [`WORKER_SPIN`], then parks until there is one.
    fn next(&self) -> Part {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if self.queued.load(Ordering::Acquire) > 0
                && let Some(part) = self.pop()
            {
                return part;
            }
            spins += 1;
            if spins.is_multiple_of(64) && start.elapsed() >= WORKER_SPIN {
                break;
            }
            std::hint::spin_loop();
        }
        let mut queue = lock(&self.queue);
        loop {
            if let Some(part) = queue.pop_front() {
                self.queued.fetch_sub(1, Ordering::Release);
                return part;
            }
            self.sleeping.fetch_add(1, Ordering::Relaxed);
            queue = self.ready.wait(queue).unwrap_or_else(PoisonError::into_inner);
            self.sleeping.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn pop(&self) -> Option<Part> {
        let part = lock(&self.queue).pop_front();
        if part.is_some() {
            self.queued.fetch_sub(1, Ordering::Release);
        }
        part
    }

    /// Hands `parts` to the workers.
    fn push(&self, parts: &mut VecDeque<Part>) {
        let count = parts.len();
        let sleeping = {
            let mut queue = lock(&self.queue);
            queue.append(parts);
            self.queued.fetch_add(count, Ordering::Release);
            self.sleeping.load(Ordering::Relaxed)
        };
        for _ in 0..count.min(sleeping) {
            self.ready.notify_one();
        }
    }
}

static POOL: LazyLock<Arc<Pool>> = LazyLock::new(|| {
    // One thread per physical core: threads of one core share its memory bandwidth, and the workers
    // stay where the scheduler first put them, so on a chiplet CPU (Ryzen 9 7950X) eight of them
    // sharing four cores of one chiplet copied 16 MiB in 0.28 to 0.54 ms, depending on the run,
    // where eight fresh threads took 0.36 ms and sixteen workers 0.31 ms. Asked once: the topology
    // and the cgroup limits cost tens of microseconds each.
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let threads = topology::physical_cores().map_or(threads.min(STAGING_THREADS), |cores| cores.min(threads)).clamp(1, MAX_THREADS);
    let pool = Arc::new(Pool { queue: Mutex::new(VecDeque::new()), ready: Condvar::new(), queued: AtomicUsize::new(0), sleeping: AtomicUsize::new(0), threads });
    for index in 1..threads {
        let pool = Arc::clone(&pool);
        // A worker that cannot start only means fewer threads: the caller copies whatever
        // nobody picked up.
        let _ = thread::Builder::new().name(format!("luma-copy-{index}")).spawn(move || {
            loop {
                pool.next().run();
            }
        });
    }
    pool
});

/// `dst.copy_from_slice(src)` split over several threads for large copies: one core cannot
/// saturate unified memory, and the page faults of a fresh destination spread out too. The
/// threads store past the cache (see [`copy_streaming`]). At most `STAGING_THREADS` threads
/// take part, each with one share of the copy: the transfers of a GPU backend through host
/// memory are bound by the bus, and the copy overlaps the device's work.
pub fn copy_parallel(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    copy_on(src, dst, STAGING_THREADS, false);
}

/// [`copy_parallel`] for a copy between two host buffers, which nothing but the host's memory
/// bandwidth limits: spread over one thread per physical core (at most `MAX_THREADS`), which
/// take chunks of `CHUNK` bytes as they get to them instead of a share each, so that a thread
/// the scheduler has put on a busy core, or on the same core as another copy thread, costs a
/// chunk and not a share.
pub fn copy_parallel_host(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    copy_on(src, dst, MAX_THREADS, true);
}

/// Copies `src` to `dst` on at most `most_threads` threads: each takes one even share, or, if
/// `chunks`, chunks of [`CHUNK`] bytes as it gets to them.
fn copy_on(src: &[u8], dst: &mut [MaybeUninit<u8>], most_threads: usize, chunks: bool) {
    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    if src.len() < PARALLEL_COPY_MIN {
        dst.write_copy_of_slice(src);
        return;
    }
    let pool = &*POOL;
    let parts = (src.len() / PART_MIN).clamp(1, pool.threads.min(most_threads));
    if parts == 1 {
        dst.write_copy_of_slice(src);
        return;
    }
    let job = Job { remaining: AtomicUsize::new(0), waiter: thread::current(), src: src.as_ptr(), dst: dst.as_mut_ptr(), len: src.len(), next: AtomicUsize::new(0) };
    let mut queued = VecDeque::with_capacity(parts - 1);
    let mut own = None;
    if chunks {
        for _ in 1..parts {
            queued.push_back(Part { src: src.as_ptr(), dst: dst.as_mut_ptr(), len: 0, job: &raw const job });
        }
    } else {
        let share = src.len().div_ceil(parts).next_multiple_of(4096);
        let mut shares = src.chunks(share).zip(dst.chunks_mut(share));
        own = shares.next();
        for (s, d) in shares {
            queued.push_back(Part { src: s.as_ptr(), dst: d.as_mut_ptr(), len: s.len(), job: &raw const job });
        }
    }
    // The count is complete before any worker can see a share.
    job.remaining.store(queued.len(), Ordering::Release);
    pool.push(&mut queued);
    match own {
        Some((own_src, own_dst)) => copy_streaming(own_src, own_dst),
        // SAFETY: the buffers are `len` bytes of two distinct slices, which outlive the call.
        None => unsafe { job.copy_chunks() },
    }
    // A waiting thread copies shares too (its own, or another call's), so a pool kept busy by
    // other calls cannot leave this one without a thread to run its shares.
    while job.remaining.load(Ordering::Acquire) != 0 {
        match pool.pop() {
            Some(part) => part.run(),
            None => thread::park(),
        }
    }
}

/// `dst.copy_from_slice(src)` with non-temporal stores: the destination lines are not read into
/// the cache before being overwritten, which a large readback (bound by host memory bandwidth
/// while the GPU's transfer writes the next chunk) would pay for with a third of its traffic.
#[cfg(target_arch = "x86_64")]
pub fn copy_streaming(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
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
pub fn copy_streaming(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    dst.write_copy_of_slice(src);
}

/// Backs a large readback destination with transparent huge pages: faulting in, and later
/// freeing, its 4 KiB pages one by one costs about as much as the copy itself. Destinations
/// below 8 MiB are left to the small pages (a huge page is 2 MiB).
#[cfg(target_os = "linux")]
pub fn advise_huge_pages(dst: &mut [MaybeUninit<u8>]) {
    const PAGE: usize = 4096;
    if dst.len() < 8 << 20 {
        return;
    }
    let start = (dst.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let end = (dst.as_mut_ptr() as usize + dst.len()) & !(PAGE - 1);
    // SAFETY: the page-aligned range lies within `dst`; `madvise` only changes how the kernel
    // backs it, never its contents, and failure is harmless.
    unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
pub fn advise_huge_pages(_: &mut [MaybeUninit<u8>]) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(len: usize) {
        let src: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8).collect();
        for copy in [copy_parallel, copy_parallel_host] {
            let mut dst = vec![MaybeUninit::<u8>::uninit(); len];
            advise_huge_pages(&mut dst);
            copy(&src, &mut dst);
            // SAFETY: the copy initialised every byte of `dst`.
            let out: Vec<u8> = dst.iter().map(|b| unsafe { b.assume_init() }).collect();
            assert!(out == src, "copy of {len} bytes differs");
        }
    }

    #[test]
    fn small_copy() {
        check(0);
        check(1000);
        check(PARALLEL_COPY_MIN - 1);
    }

    #[test]
    fn large_copy() {
        check(PARALLEL_COPY_MIN + 4099);
        check(3 * PARALLEL_COPY_MIN);
        // Around the chunk boundaries of a host copy: a last chunk of one byte, whole chunks.
        check(PARALLEL_COPY_MIN + CHUNK + 1);
        check(8 * CHUNK);
        check(17 * CHUNK - 1);
    }

    #[test]
    fn many_threads_copy_at_once() {
        // Calls share the workers: every one must still get exactly its own bytes, whoever copies which share.
        let threads: Vec<_> = (0..12).map(|t| std::thread::spawn(move || (0..6).for_each(|i| check(PARALLEL_COPY_MIN * (1 + (t + i) % 5) + 17 * t)))).collect();
        for thread in threads {
            thread.join().expect("a copying thread panicked");
        }
    }
}
