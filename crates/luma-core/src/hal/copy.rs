//! Host memory copies shared by every backend's upload and readback paths.

use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// Copies below this size stay on the calling thread.
const PARALLEL_COPY_MIN: usize = 2 << 20;

/// Smallest share of a parallel copy, so that a copy just above [`PARALLEL_COPY_MIN`] is not
/// cut into pieces that cost more to hand out than to copy.
const PART_MIN: usize = 512 << 10;

/// Most threads (the caller included) a copy is spread over.
const MAX_THREADS: usize = 8;

/// How long an idle worker polls for the next share before it parks.
const WORKER_SPIN: Duration = Duration::from_micros(200);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One share of a parallel copy, waiting for a worker.
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

/// The shares of one `copy_parallel` call still to be copied by workers.
struct Job {
    remaining: AtomicUsize,
    waiter: Thread,
}

impl Part {
    /// Copies the share, then tells its caller.
    fn run(self) {
        // SAFETY: `src` and `dst` are `len` bytes of two distinct buffers (see `Send` above); the
        // shares of a call do not overlap.
        unsafe {
            copy_streaming(std::slice::from_raw_parts(self.src, self.len), std::slice::from_raw_parts_mut(self.dst, self.len));
            // The waiter is cloned before the count drops: the job lives on the caller's stack,
            // which is free to go once the count reaches zero.
            let job = &*self.job;
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
            if spins % 64 == 0 && start.elapsed() >= WORKER_SPIN {
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
    // Asked once: the call reads the cgroup limits and costs tens of microseconds.
    let threads = thread::available_parallelism().map_or(1, |n| n.get().min(MAX_THREADS));
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
/// threads store past the cache (see [`copy_streaming`]).
pub fn copy_parallel(src: &[u8], dst: &mut [MaybeUninit<u8>]) {
    assert_eq!(src.len(), dst.len(), "source and destination lengths differ");
    if src.len() < PARALLEL_COPY_MIN {
        dst.write_copy_of_slice(src);
        return;
    }
    let pool = &*POOL;
    let parts = (src.len() / PART_MIN).clamp(1, pool.threads);
    if parts == 1 {
        dst.write_copy_of_slice(src);
        return;
    }
    let share = src.len().div_ceil(parts).next_multiple_of(4096);
    let job = Job { remaining: AtomicUsize::new(0), waiter: thread::current() };
    let mut shares = src.chunks(share).zip(dst.chunks_mut(share));
    let Some((own_src, own_dst)) = shares.next() else { return };
    let mut queued = VecDeque::with_capacity(parts - 1);
    for (s, d) in shares {
        queued.push_back(Part { src: s.as_ptr(), dst: d.as_mut_ptr(), len: s.len(), job: &raw const job });
    }
    // The count is complete before any worker can see a share.
    job.remaining.store(queued.len(), Ordering::Release);
    pool.push(&mut queued);
    copy_streaming(own_src, own_dst);
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
        let mut dst = vec![MaybeUninit::<u8>::uninit(); len];
        advise_huge_pages(&mut dst);
        copy_parallel(&src, &mut dst);
        // SAFETY: `copy_parallel` initialised every byte of `dst`.
        let out: Vec<u8> = dst.iter().map(|b| unsafe { b.assume_init() }).collect();
        assert!(out == src, "copy of {len} bytes differs");
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
