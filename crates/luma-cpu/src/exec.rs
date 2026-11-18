//! Executable memory (`ExecBlob`), aligned buffers (`AlignedBuf`), and multi-core `WorkgroupPool`.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::JoinHandle;

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

/// One in-flight dispatch. Lives on the dispatcher's stack; workers see it by address only while
/// `Shared::state.job` is set and `active` accounts for them.
struct Job {
    func: WorkgroupFn,
    grid_size: [u32; 4],
    workgroups: [u32; 4],
    bindings: *const *mut u8,
    binding_bytes: *const u64,
    lengths: *const u32,
    shared_bytes: usize,
    scratch_bytes: usize,
    initialize_shared: bool,
    total: u64,
    chunk: u64,
    next: AtomicU64,
}

#[derive(Default)]
struct State {
    generation: u64,
    job: Option<usize>,
    active: usize,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
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

/// Claims and runs chunks of `job` on the current thread.
///
/// # Safety
/// The pointers in `job` must stay valid for the duration of the call.
unsafe fn run_job(job: &Job) {
    LOCAL.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (shared, scratch) = &mut *guard;
        ensure(shared, job.shared_bytes);
        ensure(scratch, job.scratch_bytes);
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
    });
}

fn worker_loop(shared: Arc<Shared>) {
    let mut seen = 0u64;
    loop {
        let addr = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if st.shutdown {
                    return;
                }
                if st.generation != seen {
                    seen = st.generation;
                    if let Some(addr) = st.job {
                        st.active += 1;
                        break addr;
                    }
                    continue;
                }
                st = shared.wake.wait(st).unwrap();
            }
        };
        // SAFETY: the dispatcher keeps the `Job` alive until `active` returns to zero.
        unsafe { run_job(&*(addr as *const Job)) };
        let mut st = shared.state.lock().unwrap();
        st.active -= 1;
        if st.active == 0 {
            shared.done.notify_all();
        }
    }
}

/// Persistent multi-core workgroup executor.
pub struct WorkgroupPool {
    threads: usize,
    shared: Arc<Shared>,
    dispatch_lock: Mutex<()>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

static POOL: LazyLock<Arc<WorkgroupPool>> = LazyLock::new(|| {
    let threads = std::env::var("LUMA_CPU_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    WorkgroupPool::new(threads)
});

impl WorkgroupPool {
    pub fn global() -> &'static Arc<WorkgroupPool> {
        &POOL
    }

    pub fn new(threads: usize) -> Arc<Self> {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
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
            dispatch_lock: Mutex::new(()),
            handles: Mutex::new(handles),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `func` once per workgroup and returns when all have finished.
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
    ) {
        let total = u64::from(workgroups[0]) * u64::from(workgroups[1]) * u64::from(workgroups[2]);
        if total == 0 {
            return;
        }
        let parallel = total >= 4 && self.threads > 1;
        let chunk = if parallel {
            ((total as usize) / (self.threads * 4)).clamp(1, 64) as u64
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
            shared_bytes: shared_bytes as usize,
            scratch_bytes: scratch_bytes as usize,
            initialize_shared,
            total,
            chunk,
            next: AtomicU64::new(0),
        };
        let Ok(_serial) = (if parallel {
            self.dispatch_lock.try_lock()
        } else {
            Err(std::sync::TryLockError::WouldBlock)
        }) else {
            // SAFETY: slices outlive the call.
            unsafe { run_job(&job) };
            return;
        };
        {
            let mut st = self.shared.state.lock().unwrap();
            st.job = Some(&job as *const Job as usize);
            st.generation += 1;
        }
        self.shared.wake.notify_all();
        // SAFETY: slices outlive the call.
        unsafe { run_job(&job) };
        let mut st = self.shared.state.lock().unwrap();
        // No new worker may pick the job up after this point.
        st.job = None;
        while st.active > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
    }
}

impl Drop for WorkgroupPool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.wake.notify_all();
        for h in self.handles.get_mut().unwrap().drain(..) {
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

    fn run(pool: &WorkgroupPool, wg: [u32; 3]) {
        let count = (wg[0] * wg[1] * wg[2]) as usize;
        let buf = AlignedBuf::new(count as u64 * 4).unwrap();
        let bindings = [buf.as_ptr()];
        let bytes = [count as u64 * 4];
        pool.dispatch(test_wg, [1, 1, 1], wg, &bindings, &bytes, &[], 16, 16, true);
        // SAFETY: dispatch has completed.
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u32>(), count) };
        for z in 0..wg[2] {
            for y in 0..wg[1] {
                for x in 0..wg[0] {
                    let lin = (x + wg[0] * (y + wg[1] * z)) as usize;
                    assert_eq!(data[lin], x + y * 1000 + z * 1_000_000 + 1);
                }
            }
        }
    }

    #[test]
    fn dispatch_single_and_many_workgroups() {
        for threads in [1, 4] {
            let pool = WorkgroupPool::new(threads);
            run(&pool, [1, 1, 1]);
            run(&pool, [64, 2, 1]);
            run(&pool, [0, 2, 1]);
        }
    }
}
