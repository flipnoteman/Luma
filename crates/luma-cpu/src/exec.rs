//! Executable memory (`ExecBlob`), aligned buffers (`AlignedBuf`), and multi-core `WorkgroupPool`.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use luma_core::{BackendError, NativeBuffer};

use crate::abi::{MATH_TABLE, WorkgroupCtx, WorkgroupFn};

const BUF_ALIGN: usize = 64;
const HUGE_PAGE_MIN: usize = 2 << 20;

/// 64-byte-aligned, zero-initialised host buffer.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
    size: usize,
}

// SAFETY: the buffer is plain owned heap memory; synchronisation of contents is the caller's job.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("size", &self.size)
            .finish()
    }
}

impl NativeBuffer for AlignedBuf {}

impl AlignedBuf {
    pub fn new(size: u64) -> Option<Self> {
        let len = usize::try_from(size).ok()?;
        let alloc_bytes = len.max(BUF_ALIGN).checked_next_multiple_of(BUF_ALIGN)?;
        let layout = Layout::from_size_align(alloc_bytes, BUF_ALIGN).ok()?;
        // SAFETY: layout has non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })?;
        if alloc_bytes >= HUGE_PAGE_MIN {
            // SAFETY: the range is the live allocation just made.
            let region = unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<MaybeUninit<u8>>(), alloc_bytes)
            };
            luma_core::hal::advise_huge_pages(region);
        }
        Some(Self {
            ptr,
            layout,
            size: len,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// # Safety
    /// `len <= self.len()` and no concurrent writer may alias the range.
    pub unsafe fn as_slice(&self, len: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    /// # Safety
    /// `len <= self.len()` and the caller must guarantee exclusive access to the range.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice(&self, len: usize) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }

    /// # Safety
    /// Same contract as [`Self::as_mut_slice`].
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_uninit_slice_mut(&self, len: usize) -> &mut [MaybeUninit<u8>] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<MaybeUninit<u8>>(), len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Machine code in an executable mapping.
pub struct ExecBlob {
    ptr: *mut u8,
    map_len: usize,
    bytes: Box<[u8]>,
}

// SAFETY: the mapping is immutable (read+execute) after construction.
unsafe impl Send for ExecBlob {}
unsafe impl Sync for ExecBlob {}

impl fmt::Debug for ExecBlob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecBlob")
            .field("len", &self.bytes.len())
            .finish()
    }
}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n > 0 { n as usize } else { 4096 }
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const PAGE_EXECUTE_READ: u32 = 0x20;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn VirtualAlloc(addr: *mut c_void, size: usize, ty: u32, protect: u32) -> *mut c_void;
        pub fn VirtualProtect(addr: *mut c_void, size: usize, new: u32, old: *mut u32) -> i32;
        pub fn VirtualFree(addr: *mut c_void, size: usize, ty: u32) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn FlushInstructionCache(process: *mut c_void, addr: *const c_void, size: usize)
        -> i32;
    }
}

impl ExecBlob {
    pub fn new(code: &[u8]) -> Result<Self, BackendError> {
        if code.is_empty() {
            return Err(BackendError::ShaderCompile(
                "empty machine code blob".into(),
            ));
        }
        let ptr = Self::map_exec(code)?;
        let map_len = Self::map_len(code.len());
        Ok(Self {
            ptr,
            map_len,
            bytes: code.into(),
        })
    }

    #[cfg(unix)]
    fn map_len(code_len: usize) -> usize {
        code_len.max(4096).next_multiple_of(page_size())
    }

    #[cfg(windows)]
    fn map_len(code_len: usize) -> usize {
        code_len
    }

    #[cfg(unix)]
    fn map_exec(code: &[u8]) -> Result<*mut u8, BackendError> {
        let map_len = Self::map_len(code.len());
        // SAFETY: anonymous private mapping; arguments are valid.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(BackendError::ShaderCompile(format!(
                "mmap of {map_len} bytes for machine code failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        // SAFETY: `raw` maps `map_len >= code.len()` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), raw.cast::<u8>(), code.len()) };
        // SAFETY: `raw`/`map_len` describe the mapping created above.
        if unsafe { libc::mprotect(raw, map_len, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
            let err = std::io::Error::last_os_error();
            // SAFETY: unmapping our own mapping.
            unsafe { libc::munmap(raw, map_len) };
            return Err(BackendError::ShaderCompile(format!(
                "mprotect(RX) for machine code failed: {err}"
            )));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(start: *const std::ffi::c_void, len: usize);
            }
            // SAFETY: range lies within the mapping.
            unsafe { sys_icache_invalidate(raw, code.len()) };
        }
        #[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
        {
            unsafe extern "C" {
                fn __clear_cache(start: *mut std::ffi::c_void, end: *mut std::ffi::c_void);
            }
            // SAFETY: range lies within the mapping.
            unsafe { __clear_cache(raw, raw.cast::<u8>().add(code.len()).cast()) };
        }
        Ok(raw.cast::<u8>())
    }

    #[cfg(windows)]
    fn map_exec(code: &[u8]) -> Result<*mut u8, BackendError> {
        use win::*;
        let fail = |what: &str| {
            BackendError::ShaderCompile(format!(
                "{what} for machine code failed: {}",
                std::io::Error::last_os_error()
            ))
        };
        // SAFETY: plain Win32 calls on a region we allocate and own.
        unsafe {
            let raw = VirtualAlloc(
                std::ptr::null_mut(),
                code.len(),
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            );
            if raw.is_null() {
                return Err(fail("VirtualAlloc"));
            }
            std::ptr::copy_nonoverlapping(code.as_ptr(), raw.cast::<u8>(), code.len());
            let mut old = 0u32;
            if VirtualProtect(raw, code.len(), PAGE_EXECUTE_READ, &mut old) == 0 {
                let err = fail("VirtualProtect");
                VirtualFree(raw, 0, MEM_RELEASE);
                return Err(err);
            }
            FlushInstructionCache(GetCurrentProcess(), raw, code.len());
            Ok(raw.cast::<u8>())
        }
    }

    pub fn as_fn(&self) -> WorkgroupFn {
        // SAFETY: the mapping holds a complete `extern "C" fn(*const WorkgroupCtx)` entry at offset 0.
        unsafe { std::mem::transmute::<*const u8, WorkgroupFn>(self.ptr) }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for ExecBlob {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: unmapping the mapping created in `new`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.map_len);
        }
        #[cfg(windows)]
        // SAFETY: releasing the region allocated in `new`.
        unsafe {
            let _ = self.map_len;
            win::VirtualFree(self.ptr.cast(), 0, win::MEM_RELEASE);
        }
    }
}

/// Learned execution cost of one kernel, in 1/1024 ns per workgroup (`0` = not measured yet).
///
/// Every dispatch times the work each participant does and folds it in, so [`WorkgroupPool`] can
/// run launches too small to pay for a hand-off inline and wake only as many workers as the
/// launch can keep busy.
#[derive(Debug, Default)]
pub struct CostHint(AtomicU64);

/// Nanoseconds of work one already-running (spinning) helper must get to be worth enlisting.
const HOT_NS: u64 = 5_000;
/// Nanoseconds of work per helper that justifies waking a sleeping worker (a futex round trip
/// costs tens of microseconds).
const COLD_NS: u64 = 16_000;
/// Prior cost of one invocation before the first measurement.
const PRIOR_NS_PER_INVOCATION: u64 = 1;
/// How long an idle worker or the waiting dispatcher polls before it sleeps on a condvar
/// (`LUMA_CPU_SPIN_US`). Long enough to bridge the host work between the launches of a loop.
const DEFAULT_SPIN: Duration = Duration::from_micros(1000);
/// Launches whose threads each work longer than this (nanoseconds) amortise a wake-up by
/// themselves: their workers sleep at once instead of polling, which leaves the cores to the
/// kernel (spinners on the other hyperthreads and a lower all-core clock slowed memory-bound
/// kernels by up to 2x).
/// Polling rounds (about 0.4 us each) that are a pure busy loop; later rounds yield the core.
const PAUSE_ROUNDS: u32 = 100;
const SPIN_BELOW_NS: u64 = 100_000;
/// Low bits of `Shared::slots`: helper slots still free; the bits above are the job generation.
const SLOT_BITS: u32 = 16;
const SLOT_MASK: u64 = (1 << SLOT_BITS) - 1;

impl CostHint {
    /// Estimated nanoseconds of work for `total` workgroups of `invocations` invocations.
    fn estimate_ns(&self, total: u64, invocations: u64) -> u64 {
        match self.0.load(Ordering::Relaxed) {
            0 => invocations.saturating_mul(PRIOR_NS_PER_INVOCATION),
            fixed => {
                ((u128::from(fixed) * u128::from(total)) >> 10).min(u128::from(u64::MAX)) as u64
            }
        }
    }

    fn record(&self, busy_ns: u64, total: u64) {
        let new = (u128::from(busy_ns) << 10) / u128::from(total.max(1));
        let new = new.clamp(1, u128::from(u64::MAX >> 1)) as u64;
        let old = self.0.load(Ordering::Relaxed);
        self.0.store(
            if old == 0 { new } else { (old + new) / 2 },
            Ordering::Relaxed,
        );
    }
}

/// One in-flight dispatch. Lives on the dispatcher's stack; a worker sees it only after claiming
/// one of its helper slots, and the dispatcher returns only once every claimed slot is finished.
struct Job {
    func: WorkgroupFn,
    grid_size: [u32; 4],
    workgroups: [u32; 4],
    bindings: *const *mut u8,
    binding_bytes: *const u64,
    lengths: *const u32,
    native: *const u8,
    shared_bytes: usize,
    scratch_bytes: usize,
    initialize_shared: bool,
    total: u64,
    chunk: u64,
    next: AtomicU64,
    /// Helpers that have not finished: the slots offered, less those found unclaimed when the job
    /// was closed, less the helpers that are done.
    pending: AtomicUsize,
    /// Nanoseconds every participant spent running workgroups.
    busy_ns: AtomicU64,
}

/// Keeps a hot atomic on its own cache line.
#[repr(align(64))]
struct Padded<T>(T);

struct Shared {
    /// `generation << SLOT_BITS | free helper slots` of the current job; no free slots once the
    /// job is closed or fully claimed. A worker claims a slot with a compare-exchange, so the
    /// dispatcher's closing swap decides exactly which workers took part.
    slots: Padded<AtomicU64>,
    /// The current job; valid for a worker that holds a slot of the current generation.
    job: AtomicPtr<Job>,
    /// Workers blocked on (or about to block on) `wake`.
    sleepers: AtomicUsize,
    /// The configured polling time (see [`DEFAULT_SPIN`]).
    spin: Duration,
    /// Nanoseconds a worker polls after finishing a job: `spin` after a short launch, none after
    /// a long one.
    worker_spin_ns: AtomicU64,
    dispatcher_sleeping: AtomicBool,
    shutdown: AtomicBool,
    wake_lock: Mutex<()>,
    wake: Condvar,
    done_lock: Mutex<()>,
    done: Condvar,
}

thread_local! {
    // u128 storage keeps both buffers 16-byte aligned.
    static LOCAL: RefCell<(Vec<u128>, Vec<u128>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn ensure(buf: &mut Vec<u128>, bytes: usize) {
    let words = bytes.div_ceil(16);
    if buf.len() < words {
        buf.resize(words, 0);
    }
}

/// Claims and runs chunks of `job` on the current thread; returns the nanoseconds spent.
///
/// # Safety
/// The pointers in `job` must stay valid for the duration of the call.
unsafe fn run_job(job: &Job) -> u64 {
    // Taken out of the cell (not borrowed across the kernel calls) so that a dispatch issued from
    // inside a kernel on this thread cannot hit a held borrow.
    let (mut shared, mut scratch) = LOCAL.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    ensure(&mut shared, job.shared_bytes);
    ensure(&mut scratch, job.scratch_bytes);
    let start = Instant::now();
    let shared_ptr = shared.as_mut_ptr().cast::<u8>();
    let scratch_ptr = scratch.as_mut_ptr().cast::<u8>();
    let mut ctx = WorkgroupCtx {
        grid_size: job.grid_size,
        num_workgroups: job.workgroups,
        workgroup_id: [0, 0, 0, 1],
        bindings: job.bindings,
        binding_bytes: job.binding_bytes,
        lengths: job.lengths,
        shared: shared_ptr,
        scratch: scratch_ptr,
        native: job.native,
        math: &MATH_TABLE,
    };
    let wx_count = u64::from(job.workgroups[0]);
    let wy_count = u64::from(job.workgroups[1]);
    loop {
        let start = job.next.fetch_add(job.chunk, Ordering::Relaxed);
        if start >= job.total {
            break;
        }
        let end = (start + job.chunk).min(job.total);
        for idx in start..end {
            let wy_wz = idx / wx_count;
            ctx.workgroup_id = [
                (idx % wx_count) as u32,
                (wy_wz % wy_count) as u32,
                (wy_wz / wy_count) as u32,
                1,
            ];
            if job.initialize_shared && job.shared_bytes > 0 {
                // SAFETY: `shared_ptr` holds at least `shared_bytes` bytes.
                unsafe { std::ptr::write_bytes(shared_ptr, 0, job.shared_bytes) };
            }
            // SAFETY: `ctx` is fully initialised and the compiled function honours the ABI.
            unsafe { (job.func)(&ctx) };
        }
    }
    let ns = start.elapsed().as_nanos() as u64;
    LOCAL.with(|cell| *cell.borrow_mut() = (shared, scratch));
    ns
}

/// One polling round of a spin wait: a busy pause for the first rounds, then the core is offered
/// to other threads between polls (a spinner must not hold a core an oversubscribed machine
/// needs for real work).
fn relax(round: &mut u32) {
    if *round < PAUSE_ROUNDS {
        for _ in 0..32 {
            std::hint::spin_loop();
        }
        *round += 1;
    } else {
        std::thread::yield_now();
    }
}

impl Shared {
    /// Takes a helper slot of a job this worker has not looked at yet, if one is free. `seen` is
    /// the last generation looked at; it is updated whenever a new one is observed.
    fn try_claim(&self, seen: &mut u64) -> bool {
        let mut cur = self.slots.0.load(Ordering::Acquire);
        loop {
            let generation = cur >> SLOT_BITS;
            if generation == *seen {
                return false;
            }
            if cur & SLOT_MASK == 0 {
                *seen = generation;
                return false;
            }
            match self.slots.0.compare_exchange_weak(
                cur,
                cur - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    *seen = generation;
                    return true;
                }
                Err(now) => cur = now,
            }
        }
    }

    fn claimable(&self, seen: u64) -> bool {
        let cur = self.slots.0.load(Ordering::SeqCst);
        cur >> SLOT_BITS != seen && cur & SLOT_MASK != 0
    }

    /// Blocks until a job with a free slot is open (or shutdown).
    fn sleep(&self, seen: u64) {
        let mut guard = self
            .wake_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        while !self.shutdown.load(Ordering::Relaxed) && !self.claimable(seen) {
            guard = self
                .wake
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner);
        }
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }
}

fn worker_loop(shared: Arc<Shared>) {
    let mut seen = 0u64;
    let mut deadline = Instant::now() + shared.spin;
    let mut round = 0u32;
    loop {
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        if shared.try_claim(&mut seen) {
            let job = shared.job.load(Ordering::Acquire);
            // SAFETY: holding a slot of the current generation keeps the dispatcher from
            // returning (and freeing the `Job`) until `pending` is decremented below.
            let busy = unsafe {
                let job = &*job;
                let busy = run_job(job);
                job.busy_ns.fetch_add(busy, Ordering::Relaxed);
                // The job must not be touched after this decrement.
                job.pending.fetch_sub(1, Ordering::SeqCst) == 1
            };
            if busy && shared.dispatcher_sleeping.load(Ordering::SeqCst) {
                drop(
                    shared
                        .done_lock
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner),
                );
                shared.done.notify_one();
            }
            deadline = Instant::now()
                + Duration::from_nanos(shared.worker_spin_ns.load(Ordering::Relaxed));
            continue;
        }
        if Instant::now() < deadline {
            relax(&mut round);
            continue;
        }
        shared.sleep(seen);
        // Woken for one job: if another worker took the slot, go back to sleep at once.
        deadline = Instant::now();
    }
}

/// Persistent multi-core workgroup executor.
pub struct WorkgroupPool {
    threads: usize,
    shared: Arc<Shared>,
    /// Serialises parallel dispatches; holds the generation counter of the last one.
    dispatch_lock: Mutex<u64>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

static POOL: LazyLock<Arc<WorkgroupPool>> = LazyLock::new(|| {
    let threads = std::env::var("LUMA_CPU_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let spin = std::env::var("LUMA_CPU_SPIN_US")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(DEFAULT_SPIN, Duration::from_micros);
    WorkgroupPool::with_spin(threads, spin)
});

impl WorkgroupPool {
    pub fn global() -> &'static Arc<WorkgroupPool> {
        &POOL
    }

    pub fn new(threads: usize) -> Arc<Self> {
        Self::with_spin(threads, DEFAULT_SPIN)
    }

    /// A pool of `threads` threads (the caller of `dispatch` is one) whose idle workers poll for
    /// `spin` before they sleep.
    pub fn with_spin(threads: usize, spin: Duration) -> Arc<Self> {
        let threads = threads.clamp(1, 4096);
        let shared = Arc::new(Shared {
            slots: Padded(AtomicU64::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            sleepers: AtomicUsize::new(0),
            spin,
            worker_spin_ns: AtomicU64::new(spin.as_nanos() as u64),
            dispatcher_sleeping: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            wake_lock: Mutex::new(()),
            wake: Condvar::new(),
            done_lock: Mutex::new(()),
            done: Condvar::new(),
        });
        let handles = (0..threads - 1)
            .filter_map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("luma-cpu-worker-{i}"))
                    .spawn(move || worker_loop(shared))
                    .ok()
            })
            .collect();
        Arc::new(Self {
            threads,
            shared,
            dispatch_lock: Mutex::new(0),
            handles: Mutex::new(handles),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// How many threads (the caller included) a launch of `est_ns` nanoseconds of work over
    /// `total` workgroups should use, and how many of them are asleep and must be woken.
    fn plan(&self, est_ns: u64, total: u64) -> (usize, usize) {
        let workers = self.threads - 1;
        let sleepers = self.shared.sleepers.load(Ordering::Relaxed).min(workers);
        let hot = workers - sleepers;
        let by_hot = 1 + hot.min((est_ns / HOT_NS).saturating_sub(1) as usize);
        let by_cold = (est_ns / COLD_NS).min(self.threads as u64) as usize;
        let participants = by_hot
            .max(by_cold)
            .min(total.min(self.threads as u64) as usize);
        (participants, (participants - 1).saturating_sub(hot))
    }

    /// Runs `func` once per workgroup and returns when all have finished. `cost` learns how long
    /// the kernel takes, which decides how many threads later launches of it use.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        func: WorkgroupFn,
        grid_size: [u32; 3],
        workgroups: [u32; 3],
        bindings: &[*mut u8],
        binding_bytes: &[u64],
        lengths: &[u32],
        shared_bytes: u32,
        scratch_bytes: u32,
        initialize_shared: bool,
        cost: &CostHint,
    ) {
        self.dispatch_native(func, std::ptr::null(), grid_size, workgroups, bindings, binding_bytes, lengths, shared_bytes, scratch_bytes, initialize_shared, cost);
    }

    /// [`WorkgroupPool::dispatch`] of a native kernel, whose function finds `native` in
    /// [`WorkgroupCtx::native`].
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_native(
        &self,
        func: WorkgroupFn,
        native: *const u8,
        grid_size: [u32; 3],
        workgroups: [u32; 3],
        bindings: &[*mut u8],
        binding_bytes: &[u64],
        lengths: &[u32],
        shared_bytes: u32,
        scratch_bytes: u32,
        initialize_shared: bool,
        cost: &CostHint,
    ) {
        let total = u64::from(workgroups[0]) * u64::from(workgroups[1]) * u64::from(workgroups[2]);
        if total == 0 {
            return;
        }
        let invocations = u64::from(grid_size[0])
            .saturating_mul(u64::from(grid_size[1]))
            .saturating_mul(u64::from(grid_size[2]));
        let est_ns = cost.estimate_ns(total, invocations);
        let (participants, wake) = if self.threads > 1 && total > 1 {
            self.plan(est_ns, total)
        } else {
            (1, 0)
        };
        let chunk = if participants > 1 {
            ((total as usize) / (participants * 4)).clamp(1, 64) as u64
        } else {
            total
        };
        let job = Job {
            func,
            grid_size: [grid_size[0], grid_size[1], grid_size[2], 1],
            workgroups: [workgroups[0], workgroups[1], workgroups[2], 1],
            bindings: bindings.as_ptr(),
            binding_bytes: binding_bytes.as_ptr(),
            lengths: lengths.as_ptr(),
            native,
            shared_bytes: shared_bytes as usize,
            scratch_bytes: scratch_bytes as usize,
            initialize_shared,
            total,
            chunk,
            next: AtomicU64::new(0),
            pending: AtomicUsize::new(participants - 1),
            busy_ns: AtomicU64::new(0),
        };
        let serial = if participants > 1 {
            self.dispatch_lock.try_lock().ok()
        } else {
            None
        };
        let Some(mut generation) = serial else {
            // SAFETY: the slices outlive the call.
            let busy = unsafe {
                run_job(&Job {
                    chunk: total,
                    ..job
                })
            };
            cost.record(busy, total);
            return;
        };
        *generation += 1;
        let shared = &*self.shared;
        let spin = if est_ns / participants as u64 > SPIN_BELOW_NS {
            0
        } else {
            shared.spin.as_nanos() as u64
        };
        shared.worker_spin_ns.store(spin, Ordering::Relaxed);
        shared
            .job
            .store(std::ptr::from_ref(&job).cast_mut(), Ordering::Relaxed);
        shared.slots.0.store(
            *generation << SLOT_BITS | (participants as u64 - 1),
            Ordering::SeqCst,
        );
        let sleepers = if wake > 0 {
            shared.sleepers.load(Ordering::SeqCst)
        } else {
            0
        };
        if sleepers > 0 {
            let _guard = shared
                .wake_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if wake * 2 >= sleepers {
                // Most of them are needed: one wake-up call instead of one syscall per worker.
                shared.wake.notify_all();
            } else {
                for _ in 0..wake {
                    shared.wake.notify_one();
                }
            }
        }
        // SAFETY: the slices outlive the call.
        let busy = unsafe { run_job(&job) };
        job.busy_ns.fetch_add(busy, Ordering::Relaxed);
        // Close the job: slots nobody claimed by now never will be, and every claimed one is
        // accounted for in `pending` until its worker is done with the `Job`.
        let closed = shared
            .slots
            .0
            .swap(*generation << SLOT_BITS, Ordering::SeqCst);
        let unclaimed = (closed & SLOT_MASK) as usize;
        if unclaimed > 0 {
            job.pending.fetch_sub(unclaimed, Ordering::SeqCst);
        }
        let mut deadline = None;
        let mut round = 0u32;
        while job.pending.load(Ordering::Acquire) != 0 {
            if *deadline.get_or_insert_with(|| Instant::now() + shared.spin) > Instant::now() {
                relax(&mut round);
                continue;
            }
            let mut guard = shared
                .done_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            shared.dispatcher_sleeping.store(true, Ordering::SeqCst);
            while job.pending.load(Ordering::SeqCst) != 0 {
                guard = shared
                    .done
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            shared.dispatcher_sleeping.store(false, Ordering::SeqCst);
        }
        cost.record(job.busy_ns.load(Ordering::Relaxed), total);
    }
}

impl Drop for WorkgroupPool {
    fn drop(&mut self) {
        {
            let _guard = self
                .shared
                .wake_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            self.shared.shutdown.store(true, Ordering::SeqCst);
            self.shared.wake.notify_all();
        }
        for h in self
            .handles
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .drain(..)
        {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buf_is_aligned_and_zeroed() {
        let buf = AlignedBuf::new(100).unwrap();
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        assert_eq!(buf.len(), 100);
        // SAFETY: exclusive test access within bounds.
        unsafe {
            assert!(buf.as_slice(100).iter().all(|&b| b == 0));
            buf.as_mut_slice(100)[7] = 42;
            assert_eq!(buf.as_slice(100)[7], 42);
        }
    }

    #[test]
    fn exec_blob_rejects_empty() {
        assert!(ExecBlob::new(&[]).is_err());
    }

    unsafe extern "C" fn test_wg(ctx: *const WorkgroupCtx) {
        // SAFETY: test harness passes a valid ctx with one u32 binding sized for all workgroups.
        unsafe {
            let ctx = &*ctx;
            let out = (*ctx.bindings).cast::<u32>();
            let n = ctx.num_workgroups;
            let id = ctx.workgroup_id;
            let lin = id[0] + n[0] * (id[1] + n[1] * id[2]);
            *out.add(lin as usize) = id[0] + id[1] * 1000 + id[2] * 1_000_000 + 1;
        }
    }

    /// A hint that makes every launch look like `ns` nanoseconds per workgroup.
    fn hint(ns: u64) -> CostHint {
        CostHint(AtomicU64::new(ns << 10))
    }

    fn check(data: &[u32], wg: [u32; 3]) {
        for z in 0..wg[2] {
            for y in 0..wg[1] {
                for x in 0..wg[0] {
                    let lin = (x + wg[0] * (y + wg[1] * z)) as usize;
                    assert_eq!(
                        data[lin],
                        x + y * 1000 + z * 1_000_000 + 1,
                        "workgroup {x},{y},{z}"
                    );
                }
            }
        }
    }

    fn run_with(pool: &WorkgroupPool, wg: [u32; 3], cost: &CostHint) {
        let count = (wg[0] * wg[1] * wg[2]) as usize;
        let buf = AlignedBuf::new(count as u64 * 4).unwrap();
        let bindings = [buf.as_ptr()];
        let bytes = [count as u64 * 4];
        pool.dispatch(
            test_wg,
            [1, 1, 1],
            wg,
            &bindings,
            &bytes,
            &[],
            16,
            16,
            true,
            cost,
        );
        // SAFETY: dispatch has completed.
        check(
            unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u32>(), count) },
            wg,
        );
    }

    fn run(pool: &WorkgroupPool, wg: [u32; 3]) {
        // Unmeasured, cheap and expensive kernels: inline, a few threads, all threads.
        for cost in [CostHint::default(), hint(1), hint(1_000_000)] {
            run_with(pool, wg, &cost);
        }
    }

    #[test]
    fn dispatch_zero_one_and_many_workgroups() {
        for (threads, spin) in [(1, 1000), (2, 1000), (4, 1000), (9, 1000), (4, 0), (9, 0)] {
            let pool = WorkgroupPool::with_spin(threads, Duration::from_micros(spin));
            run(&pool, [1, 1, 1]);
            run(&pool, [2, 1, 1]);
            run(&pool, [3, 1, 1]);
            run(&pool, [64, 2, 1]);
            run(&pool, [7, 5, 3]);
            run(&pool, [1000, 1, 1]);
            run(&pool, [0, 2, 1]);
            run(&pool, [2, 0, 1]);
            run(&pool, [2, 1, 0]);
        }
    }

    #[test]
    fn dispatch_learns_the_cost_of_a_kernel() {
        let pool = WorkgroupPool::new(4);
        let cost = CostHint::default();
        assert_eq!(cost.0.load(Ordering::Relaxed), 0);
        run_with(&pool, [64, 1, 1], &cost);
        assert_ne!(cost.0.load(Ordering::Relaxed), 0);
        assert!(cost.estimate_ns(64, 64) > 0);
    }

    #[test]
    fn small_launches_stay_on_the_calling_thread() {
        let pool = WorkgroupPool::new(8);
        // Little work: no helper is enlisted however many workgroups there are.
        assert_eq!(pool.plan(1_000, 64).0, 1);
        assert_eq!(pool.plan(0, 64).0, 1);
        // A lot of work: all threads, but never more than there are workgroups.
        assert_eq!(pool.plan(100_000_000, 64).0, 8);
        assert_eq!(pool.plan(100_000_000, 3).0, 3);
        assert_eq!(pool.plan(100_000_000, 1).0, 1);
    }

    #[test]
    fn dispatch_after_workers_went_to_sleep() {
        let pool = WorkgroupPool::with_spin(6, Duration::from_micros(50));
        run_with(&pool, [256, 1, 1], &hint(1_000_000));
        let give_up = Instant::now() + Duration::from_secs(20);
        while pool.shared.sleepers.load(Ordering::SeqCst) < 5 {
            assert!(Instant::now() < give_up, "workers never went to sleep");
            std::thread::sleep(Duration::from_millis(1));
        }
        for _ in 0..50 {
            run_with(&pool, [256, 1, 1], &hint(1_000_000));
        }
    }

    #[test]
    fn back_to_back_dispatches() {
        let pool = WorkgroupPool::new(8);
        for i in 0..2000u32 {
            let cost = if i % 3 == 0 {
                hint(100)
            } else {
                hint(1_000_000)
            };
            run_with(&pool, [1 + i % 40, 1, 1], &cost);
        }
    }

    #[test]
    fn concurrent_dispatch_from_several_threads() {
        let pool = WorkgroupPool::new(6);
        std::thread::scope(|scope| {
            for t in 0..5u32 {
                let pool = &pool;
                scope.spawn(move || {
                    for i in 0..300u32 {
                        let cost = if (i + t) % 2 == 0 {
                            hint(1_000_000)
                        } else {
                            hint(10)
                        };
                        run_with(pool, [1 + (i * 7 + t) % 90, 1 + i % 3, 1], &cost);
                    }
                });
            }
        });
    }

    static NESTED_POOL: LazyLock<Arc<WorkgroupPool>> = LazyLock::new(|| WorkgroupPool::new(4));

    unsafe extern "C" fn nested_wg(ctx: *const WorkgroupCtx) {
        // SAFETY: as in `test_wg`; the nested dispatch has its own buffer.
        unsafe {
            let outer = &*ctx;
            let out = (*outer.bindings).cast::<u32>();
            let id = outer.workgroup_id[0];
            // A dispatch issued from inside a workgroup (on the dispatching thread or on a worker)
            // finds the pool busy and runs inline.
            let inner = AlignedBuf::new(16 * 4).unwrap();
            let bindings = [inner.as_ptr()];
            let bytes = [16u64 * 4];
            NESTED_POOL.dispatch(
                test_wg,
                [1, 1, 1],
                [16, 1, 1],
                &bindings,
                &bytes,
                &[],
                16,
                16,
                true,
                &hint(1_000_000),
            );
            let data = std::slice::from_raw_parts(inner.as_ptr().cast::<u32>(), 16);
            let ok = data.iter().enumerate().all(|(i, &v)| v == i as u32 + 1);
            *out.add(id as usize) = if ok { id + 1 } else { 0 };
        }
    }

    #[test]
    fn nested_dispatch_runs_inline() {
        let buf = AlignedBuf::new(64 * 4).unwrap();
        let bindings = [buf.as_ptr()];
        let bytes = [64u64 * 4];
        NESTED_POOL.dispatch(
            nested_wg,
            [1, 1, 1],
            [64, 1, 1],
            &bindings,
            &bytes,
            &[],
            16,
            16,
            true,
            &hint(1_000_000),
        );
        // SAFETY: dispatch has completed.
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u32>(), 64) };
        assert!(
            data.iter().enumerate().all(|(i, &v)| v == i as u32 + 1),
            "{data:?}"
        );
    }

    #[test]
    fn dropping_the_pool_stops_its_workers() {
        let pool = WorkgroupPool::new(5);
        run_with(&pool, [64, 1, 1], &hint(1_000_000));
        drop(pool);
        let pool = WorkgroupPool::with_spin(5, Duration::from_micros(50));
        run_with(&pool, [64, 1, 1], &hint(1_000_000));
        std::thread::sleep(Duration::from_millis(5));
        drop(pool);
    }
}
