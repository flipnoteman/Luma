//! Positional file I/O: a small thread pool that splits extents into jobs.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::JoinHandle;

/// Size of one pool job: an extent is split into chunks of this many bytes.
pub const CHUNK_BYTES: u64 = 4 << 20;

/// A byte range of a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// First byte of the range.
    pub offset: u64,
    /// Length of the range in bytes.
    pub len: u64,
}

/// A raw pointer to staging memory handed to an engine, valid for `len` bytes.
#[derive(Clone, Copy, Debug)]
pub struct StagePtr {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the constructor's contract makes the creator responsible for the
// memory being valid and unaliased until the ticket completes, on any thread.
unsafe impl Send for StagePtr {}

impl StagePtr {
    /// Wraps `ptr`/`len`.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and writes of `len` bytes, and that memory
    /// must stay valid and untouched by anyone else until the ticket of the
    /// operation it is passed to has completed.
    pub unsafe fn new(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Length of the staging region in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

struct DoneState {
    pending: usize,
    bytes: u64,
    error: Option<io::Error>,
}

struct Done {
    state: Mutex<DoneState>,
    cv: Condvar,
}

impl Done {
    fn new(pending: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(DoneState { pending, bytes: 0, error: None }),
            cv: Condvar::new(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DoneState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn failed(&self) -> bool {
        self.lock().error.is_some()
    }

    fn finish_job(&self, bytes: u64, r: io::Result<()>) {
        let mut s = self.lock();
        match r {
            Ok(()) => s.bytes += bytes,
            Err(e) => {
                if s.error.is_none() {
                    s.error = Some(e);
                }
            }
        }
        s.pending -= 1;
        if s.pending == 0 {
            self.cv.notify_all();
        }
    }

    fn result(s: &DoneState) -> io::Result<u64> {
        match &s.error {
            None => Ok(s.bytes),
            Some(e) => Err(match e.raw_os_error() {
                Some(code) => io::Error::from_raw_os_error(code),
                None => io::Error::new(e.kind(), e.to_string()),
            }),
        }
    }
}

/// A handle to an in-flight read or write. Cloning shares the operation.
#[derive(Clone)]
pub struct IoTicket(Arc<Done>);

impl std::fmt::Debug for IoTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoTicket").field("pending", &self.0.lock().pending).finish()
    }
}

impl IoTicket {
    fn completed(r: io::Result<u64>) -> Self {
        let done = Done::new(0);
        {
            let mut s = done.lock();
            match r {
                Ok(n) => s.bytes = n,
                Err(e) => s.error = Some(e),
            }
        }
        Self(done)
    }

    fn poll(&self) -> Option<io::Result<u64>> {
        let s = self.0.lock();
        (s.pending == 0).then(|| Done::result(&s))
    }

    fn wait(&self) -> io::Result<u64> {
        let mut s = self.0.lock();
        while s.pending > 0 {
            s = self.0.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        Done::result(&s)
    }
}

/// Access-pattern hints for the page cache (`posix_fadvise` on Linux, a no-op elsewhere).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// Expect sequential access.
    Sequential,
    /// Expect random access.
    Random,
    /// The range will be needed soon.
    WillNeed,
    /// The range will not be needed; drop it from the cache.
    DontNeed,
}

/// An open file with its path and access mode.
#[derive(Debug)]
pub struct IoFile {
    file: File,
    path: PathBuf,
    writable: bool,
    len: AtomicU64,
}

impl IoFile {
    /// Opens an existing file for reading, and for writing when `writable`.
    pub fn open(path: &Path, writable: bool) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(writable).open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, path: path.to_path_buf(), writable, len: AtomicU64::new(len) })
    }

    /// Creates (or truncates) a writable file of `len` bytes.
    pub fn create(path: &Path, len: u64) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        file.set_len(len)?;
        Ok(Self { file, path: path.to_path_buf(), writable: true, len: AtomicU64::new(len) })
    }

    /// Creates a writable file of `len` bytes; fails with `AlreadyExists` if the path exists.
    pub fn create_new(path: &Path, len: u64) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create_new(true).open(path)?;
        file.set_len(len)?;
        Ok(Self { file, path: path.to_path_buf(), writable: true, len: AtomicU64::new(len) })
    }

    /// The path the file was opened with.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file length in bytes (as of open, or the last [`IoFile::set_len`]).
    pub fn len(&self) -> u64 {
        self.len.load(Ordering::Relaxed)
    }

    /// Whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the file was opened for writing.
    pub fn writable(&self) -> bool {
        self.writable
    }

    /// The underlying handle, for advisory locking.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Tells the page cache how `e` will be accessed (`e.len == 0` means to the end of the
    /// file). Best effort: errors are ignored; a no-op off Linux.
    pub fn advise(&self, e: Extent, a: Advice) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let flag = match a {
                Advice::Sequential => libc::POSIX_FADV_SEQUENTIAL,
                Advice::Random => libc::POSIX_FADV_RANDOM,
                Advice::WillNeed => libc::POSIX_FADV_WILLNEED,
                Advice::DontNeed => libc::POSIX_FADV_DONTNEED,
            };
            // SAFETY: plain syscall on a descriptor we own.
            let r = unsafe {
                libc::posix_fadvise(self.file.as_raw_fd(), e.offset as libc::off_t, e.len as libc::off_t, flag)
            };
            if r != 0 {
                log::debug!("posix_fadvise({:?}) on {} failed: errno {r}", a, self.path.display());
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (e, a);
    }

    /// Flushes file data (not necessarily metadata) to stable storage.
    pub fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// Sets the file length, zero-extending or truncating.
    pub fn set_len(&self, n: u64) -> io::Result<()> {
        self.file.set_len(n)?;
        self.len.store(n, Ordering::Relaxed);
        Ok(())
    }

    /// Reserves disk space for the first `n` bytes without changing the file length
    /// (`fallocate` with `KEEP_SIZE` on Linux), so that a full disk fails here and not at a later
    /// write. A file system without `fallocate` is not an error; a no-op off Linux.
    pub fn preallocate(&self, n: u64) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: plain syscall on a descriptor we own. KEEP_SIZE leaves the length alone.
            let r = unsafe { libc::fallocate(self.file.as_raw_fd(), libc::FALLOC_FL_KEEP_SIZE, 0, n as libc::off_t) };
            if r != 0 {
                let error = io::Error::last_os_error();
                return match error.raw_os_error() {
                    Some(libc::EOPNOTSUPP | libc::ENOSYS | libc::EINVAL) => {
                        log::debug!("fallocate({n}) on {} is not supported: {error}", self.path.display());
                        Ok(())
                    }
                    _ => Err(error),
                };
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = n;
        Ok(())
    }
}

/// Alignment an engine needs for direct I/O. All zero-cost (1) for the pool engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoAlign {
    /// Required file offset alignment in bytes.
    pub offset: u64,
    /// Required length granularity in bytes.
    pub len: u64,
    /// Required memory address alignment in bytes.
    pub memory: usize,
}

impl IoAlign {
    /// No alignment requirement.
    pub const NONE: IoAlign = IoAlign { offset: 1, len: 1, memory: 1 };
}

/// An asynchronous positional read/write engine.
pub trait TileIo: Send + Sync + std::fmt::Debug {
    /// Starts reading `extent` of `file` into `dst` (`dst.len()` must equal `extent.len`).
    fn read(&self, file: &Arc<IoFile>, extent: Extent, dst: StagePtr) -> IoTicket;
    /// Starts writing `src` to `extent` of `file` (`src.len()` must equal `extent.len`).
    fn write(&self, file: &Arc<IoFile>, extent: Extent, src: StagePtr) -> IoTicket;
    /// `Some(result)` if the operation has completed (bytes transferred), `None` otherwise.
    fn poll(&self, t: &IoTicket) -> Option<io::Result<u64>>;
    /// Parks until the operation completes and returns its result.
    fn wait(&self, t: &IoTicket) -> io::Result<u64>;
    /// Alignment requirements of this engine.
    fn align(&self) -> IoAlign;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Read,
    Write,
}

struct Job {
    file: Arc<IoFile>,
    kind: Kind,
    offset: u64,
    stage: StagePtr,
    done: Arc<Done>,
}

struct Queue {
    jobs: VecDeque<Job>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    cv: Condvar,
    jobs_run: AtomicU64,
}

/// A thread pool engine: extents are split into [`CHUNK_BYTES`] jobs spread over the threads.
pub struct PoolIo {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    chunk: u64,
}

impl std::fmt::Debug for PoolIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolIo").field("threads", &self.workers.len()).field("chunk", &self.chunk).finish()
    }
}

impl PoolIo {
    /// A pool of `threads` (at least one) threads named `luma-io-N`.
    pub fn new(threads: usize) -> Self {
        Self::with_chunk(threads, CHUNK_BYTES)
    }

    fn with_chunk(threads: usize, chunk: u64) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue { jobs: VecDeque::new(), shutdown: false }),
            cv: Condvar::new(),
            jobs_run: AtomicU64::new(0),
        });
        let workers = (0..threads.max(1))
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("luma-io-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("luma-io: cannot spawn an I/O thread")
            })
            .collect();
        Self { shared, workers, chunk: chunk.max(1) }
    }

    fn submit(&self, file: &Arc<IoFile>, extent: Extent, stage: StagePtr, kind: Kind) -> IoTicket {
        if extent.len != stage.len as u64 {
            return IoTicket::completed(Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "extent length differs from the staging buffer length",
            )));
        }
        if kind == Kind::Write && !file.writable {
            return IoTicket::completed(Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "write to a file opened read-only",
            )));
        }
        match extent.offset.checked_add(extent.len) {
            Some(end) if end <= i64::MAX as u64 => {}
            _ => {
                return IoTicket::completed(Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "extent end exceeds the maximum file offset",
                )));
            }
        }
        if extent.len == 0 {
            return IoTicket::completed(Ok(0));
        }
        let njobs = extent.len.div_ceil(self.chunk) as usize;
        let done = Done::new(njobs);
        {
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            for i in 0..njobs as u64 {
                let off = i * self.chunk;
                let len = self.chunk.min(extent.len - off) as usize;
                q.jobs.push_back(Job {
                    file: file.clone(),
                    kind,
                    offset: extent.offset + off,
                    stage: StagePtr { ptr: stage.ptr.wrapping_add(off as usize), len },
                    done: done.clone(),
                });
            }
        }
        self.shared.cv.notify_all();
        IoTicket(done)
    }

    #[cfg(test)]
    fn jobs_run(&self) -> u64 {
        self.shared.jobs_run.load(Ordering::Relaxed)
    }
}

impl Drop for PoolIo {
    fn drop(&mut self) {
        self.shared.queue.lock().unwrap_or_else(|e| e.into_inner()).shutdown = true;
        self.shared.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared) {
    loop {
        let job = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(j) = q.jobs.pop_front() {
                    break j;
                }
                if q.shutdown {
                    return;
                }
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        shared.jobs_run.fetch_add(1, Ordering::Relaxed);
        let Job { file, kind, offset, stage, done } = job;
        let len = stage.len;
        // Once one job has failed the ticket's result is decided: skip the rest.
        let r = if done.failed() { Ok(()) } else { run_job(&file, kind, offset, stage) };
        // The job's hold on the file goes before the ticket completes: whoever sees the ticket
        // complete may close the file at once, and an open descriptor would keep its lock.
        drop(file);
        done.finish_job(len as u64, r);
    }
}

fn run_job(file: &IoFile, kind: Kind, offset: u64, stage: StagePtr) -> io::Result<()> {
    let file = &file.file;
    match kind {
        Kind::Read => read_full(stage.ptr, stage.len, offset, |p, n, o| read_at(file, p, n, o)),
        Kind::Write => write_full(stage.ptr, stage.len, offset, |b, o| write_at(file, b, o)),
    }
}

#[cfg(unix)]
fn read_at(file: &File, p: *mut u8, n: usize, o: u64) -> io::Result<usize> {
    use std::os::fd::AsRawFd;
    // SAFETY: `p` is valid for `n` bytes (StagePtr contract); pread writes through the raw
    // pointer, so uninitialised staging memory is fine.
    let r = unsafe { libc::pread(file.as_raw_fd(), p.cast(), n, o as libc::off_t) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(r as usize) }
}

#[cfg(windows)]
fn read_at(file: &File, p: *mut u8, n: usize, o: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    // SAFETY: `p` is valid for `n` bytes (StagePtr contract).
    file.seek_read(unsafe { std::slice::from_raw_parts_mut(p, n) }, o)
}

#[cfg(unix)]
fn write_at(file: &File, b: &[u8], o: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(file, b, o)
}

#[cfg(windows)]
fn write_at(file: &File, b: &[u8], o: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(file, b, o)
}

/// Reads exactly `len` bytes at `offset` through `read_at`, retrying `Interrupted` and short
/// reads; a zero-byte read before the end is `UnexpectedEof`.
fn read_full(
    mut dst: *mut u8,
    mut len: usize,
    mut offset: u64,
    mut read_at: impl FnMut(*mut u8, usize, u64) -> io::Result<usize>,
) -> io::Result<()> {
    while len > 0 {
        match read_at(dst, len, offset) {
            Ok(0) => {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "file ended before the extent did"));
            }
            Ok(n) => {
                dst = dst.wrapping_add(n);
                len -= n;
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Writes exactly `len` bytes at `offset` through `write_at`, retrying `Interrupted` and
/// short writes; a zero-byte write is `WriteZero`.
fn write_full(
    mut src: *const u8,
    mut len: usize,
    mut offset: u64,
    mut write_at: impl FnMut(&[u8], u64) -> io::Result<usize>,
) -> io::Result<()> {
    while len > 0 {
        // SAFETY: `src` is valid for `len` bytes (StagePtr contract).
        let buf = unsafe { std::slice::from_raw_parts(src, len) };
        match write_at(buf, offset) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "the file accepted no bytes")),
            Ok(n) => {
                src = src.wrapping_add(n);
                len -= n;
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl TileIo for PoolIo {
    fn read(&self, file: &Arc<IoFile>, extent: Extent, dst: StagePtr) -> IoTicket {
        self.submit(file, extent, dst, Kind::Read)
    }

    fn write(&self, file: &Arc<IoFile>, extent: Extent, src: StagePtr) -> IoTicket {
        self.submit(file, extent, src, Kind::Write)
    }

    fn poll(&self, t: &IoTicket) -> Option<io::Result<u64>> {
        t.poll()
    }

    fn wait(&self, t: &IoTicket) -> io::Result<u64> {
        t.wait()
    }

    fn align(&self) -> IoAlign {
        IoAlign::NONE
    }
}

fn thread_count(env: Option<&str>, available: usize) -> usize {
    let default = available.clamp(1, 8);
    match env {
        None => default,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                log::warn!("LUMA_IO_THREADS={s:?} is not a positive integer; using {default}");
                default
            }
        },
    }
}

/// The process-wide [`PoolIo`]; threads = `LUMA_IO_THREADS`, default `min(8, available_parallelism)`.
pub fn default_engine() -> Arc<dyn TileIo> {
    static ENGINE: LazyLock<Arc<PoolIo>> = LazyLock::new(|| {
        let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
        Arc::new(PoolIo::new(thread_count(std::env::var("LUMA_IO_THREADS").ok().as_deref(), avail)))
    });
    ENGINE.clone()
}

/// Reads `extent` of `file` into `buf` and waits. `buf.len()` must equal `extent.len`.
/// On `Ok` every byte of `buf` is initialised.
pub fn read_blocking(
    engine: &dyn TileIo,
    file: &Arc<IoFile>,
    extent: Extent,
    buf: &mut [MaybeUninit<u8>],
) -> io::Result<u64> {
    // SAFETY: `buf` is borrowed for the whole call, which waits for the ticket.
    let stage = unsafe { StagePtr::new(buf.as_mut_ptr().cast(), buf.len()) };
    let t = engine.read(file, extent, stage);
    engine.wait(&t)
}

/// Writes `src` to `extent` of `file` and waits. `src.len()` must equal `extent.len`.
pub fn write_blocking(engine: &dyn TileIo, file: &Arc<IoFile>, extent: Extent, src: &[u8]) -> io::Result<u64> {
    // SAFETY: `src` is borrowed for the whole call, which waits for the ticket; the engine
    // only reads through the pointer of a write.
    let stage = unsafe { StagePtr::new(src.as_ptr().cast_mut(), src.len()) };
    let t = engine.write(file, extent, stage);
    engine.wait(&t)
}

/// What [`FailingIo`] does to the selected operation.
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Do nothing and fail with this kind.
    Fail(io::ErrorKind),
    /// Transfer only the first `keep` bytes, then fail with `UnexpectedEof`.
    Short {
        /// Bytes actually transferred before the error.
        keep: u64,
    },
}

/// Which operations a [`FailingIo`] sabotages: the `n`-th (0-based) read and/or write.
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FailPlan {
    /// Fault for the n-th read, as `(n, fault)`.
    pub read: Option<(u64, Fault)>,
    /// Fault for the n-th write, as `(n, fault)`.
    pub write: Option<(u64, Fault)>,
}

/// An engine wrapper that fails or shortens the n-th read and/or write of its inner engine.
#[cfg(feature = "testing")]
#[derive(Debug)]
pub struct FailingIo<E> {
    inner: E,
    plan: FailPlan,
    reads: AtomicU64,
    writes: AtomicU64,
}

#[cfg(feature = "testing")]
impl<E: TileIo> FailingIo<E> {
    /// Wraps `inner`, applying `plan`.
    pub fn new(inner: E, plan: FailPlan) -> Self {
        Self { inner, plan, reads: AtomicU64::new(0), writes: AtomicU64::new(0) }
    }

    /// Reads issued so far (including the faulted one).
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::SeqCst)
    }

    /// Writes issued so far (including the faulted one).
    pub fn writes(&self) -> u64 {
        self.writes.load(Ordering::SeqCst)
    }

    fn apply(
        &self,
        counter: &AtomicU64,
        slot: Option<(u64, Fault)>,
        extent: Extent,
        stage: StagePtr,
        op: impl Fn(Extent, StagePtr) -> IoTicket,
    ) -> IoTicket {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        match slot {
            Some((at, Fault::Fail(kind))) if at == n => {
                IoTicket::completed(Err(io::Error::new(kind, "injected I/O failure")))
            }
            Some((at, Fault::Short { keep })) if at == n => {
                let keep = keep.min(extent.len);
                // SAFETY: a prefix of the caller's region, which they keep valid until completion.
                let part = unsafe { StagePtr::new(stage.ptr, keep as usize) };
                let t = op(Extent { offset: extent.offset, len: keep }, part);
                match self.inner.wait(&t) {
                    Ok(_) => IoTicket::completed(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "injected short transfer",
                    ))),
                    Err(e) => IoTicket::completed(Err(e)),
                }
            }
            _ => op(extent, stage),
        }
    }
}

#[cfg(feature = "testing")]
impl<E: TileIo> TileIo for FailingIo<E> {
    fn read(&self, file: &Arc<IoFile>, extent: Extent, dst: StagePtr) -> IoTicket {
        self.apply(&self.reads, self.plan.read, extent, dst, |e, p| self.inner.read(file, e, p))
    }

    fn write(&self, file: &Arc<IoFile>, extent: Extent, src: StagePtr) -> IoTicket {
        self.apply(&self.writes, self.plan.write, extent, src, |e, p| self.inner.write(file, e, p))
    }

    fn poll(&self, t: &IoTicket) -> Option<io::Result<u64>> {
        self.inner.poll(t)
    }

    fn wait(&self, t: &IoTicket) -> io::Result<u64> {
        self.inner.wait(t)
    }

    fn align(&self) -> IoAlign {
        self.inner.align()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);

    impl Temp {
        fn new(name: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!("luma-io-{}-{n}-{name}", std::process::id())))
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8).collect()
    }

    fn file_with(t: &Temp, data: &[u8]) -> Arc<IoFile> {
        std::fs::write(&t.0, data).unwrap();
        Arc::new(IoFile::open(&t.0, false).unwrap())
    }

    #[test]
    fn a_large_read_is_split_over_the_pool() {
        let t = Temp::new("large");
        let data = pattern(17 << 20);
        let f = file_with(&t, &data);
        let pool = PoolIo::new(4);
        let mut buf = vec![MaybeUninit::<u8>::uninit(); data.len()];
        let n = read_blocking(&pool, &f, Extent { offset: 0, len: data.len() as u64 }, &mut buf).unwrap();
        assert_eq!(n, data.len() as u64);
        // SAFETY: the read succeeded, so every byte is initialised.
        let got: Vec<u8> = buf.iter().map(|b| unsafe { b.assume_init() }).collect();
        assert!(got == data);
        assert_eq!(pool.jobs_run(), 5); // 4 x 4 MiB + 1 MiB
        let names: Vec<_> = pool.workers.iter().map(|w| w.thread().name().unwrap().to_string()).collect();
        assert_eq!(names, ["luma-io-0", "luma-io-1", "luma-io-2", "luma-io-3"]);
    }

    #[test]
    fn eintr_and_short_reads_are_retried() {
        let src = pattern(100);
        let mut dst = vec![0u8; 100];
        let mut calls = 0;
        read_full(dst.as_mut_ptr(), 100, 0, |p, n, o| {
            calls += 1;
            if calls % 2 == 1 {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let k = n.min(7);
            // SAFETY: `p` is valid for `n >= k` bytes and `o + k <= 100`.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(o as usize), p, k) };
            Ok(k)
        })
        .unwrap();
        assert_eq!(dst, src);
        assert!(calls > 28);

        let mut out = vec![0u8; 100];
        let mut calls = 0;
        write_full(src.as_ptr(), 100, 0, |b, o| {
            calls += 1;
            if calls % 3 == 0 {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            let k = b.len().min(9);
            out[o as usize..o as usize + k].copy_from_slice(&b[..k]);
            Ok(k)
        })
        .unwrap();
        assert_eq!(out, src);

        let e = write_full(src.as_ptr(), 1, 0, |_, _| Ok(0)).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::WriteZero);
    }

    #[test]
    fn a_read_past_the_end_is_unexpected_eof() {
        let t = Temp::new("eof");
        let f = file_with(&t, &pattern(1000));
        let pool = PoolIo::new(2);
        let mut buf = vec![MaybeUninit::<u8>::uninit(); 200];
        let e = read_blocking(&pool, &f, Extent { offset: 900, len: 200 }, &mut buf).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        let e = read_blocking(&pool, &f, Extent { offset: 5000, len: 200 }, &mut buf).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn the_first_error_wins_and_the_ticket_completes() {
        let t = Temp::new("first");
        let f = file_with(&t, &pattern(1 << 12));
        let pool = PoolIo::with_chunk(3, 512);
        // 8 jobs of 512 bytes; everything from offset 2048 on is past the end.
        let mut buf = vec![MaybeUninit::<u8>::uninit(); 1 << 13];
        // SAFETY: `buf` outlives the wait below.
        let stage = unsafe { StagePtr::new(buf.as_mut_ptr().cast(), buf.len()) };
        let ticket = pool.read(&f, Extent { offset: 2048, len: 1 << 13 }, stage);
        let e = pool.wait(&ticket).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        // Completed: polling and waiting again give the same error; all 16 jobs were retired.
        assert_eq!(pool.poll(&ticket).unwrap().unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(pool.wait(&ticket.clone()).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(pool.jobs_run(), 16);

        // Writes to a read-only file fail at once.
        let e = write_blocking(&pool, &f, Extent { offset: 0, len: 4 }, &[1, 2, 3, 4]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
        // A length mismatch is refused.
        let e = write_blocking(&pool, &f, Extent { offset: 0, len: 5 }, &[1, 2, 3, 4]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn writes_round_trip_and_create_sets_the_length() {
        let t = Temp::new("write");
        let f = Arc::new(IoFile::create(&t.0, 10_000).unwrap());
        assert_eq!((f.len(), f.writable()), (10_000, true));
        f.preallocate(20_000).unwrap();
        assert_eq!(std::fs::metadata(&t.0).unwrap().len(), 10_000);
        let pool = default_engine();
        let data = pattern(3000);
        write_blocking(&*pool, &f, Extent { offset: 100, len: 3000 }, &data).unwrap();
        f.sync_data().unwrap();
        f.advise(Extent { offset: 0, len: 0 }, Advice::DontNeed);
        let mut buf = vec![MaybeUninit::<u8>::uninit(); 3000];
        read_blocking(&*pool, &f, Extent { offset: 100, len: 3000 }, &mut buf).unwrap();
        // SAFETY: the read succeeded.
        let got: Vec<u8> = buf.iter().map(|b| unsafe { b.assume_init() }).collect();
        assert_eq!(got, data);
        f.set_len(50).unwrap();
        assert_eq!(f.len(), 50);
        assert_eq!(pool.align(), IoAlign::NONE);
    }

    #[test]
    fn thread_count_parses_the_env_value() {
        assert_eq!(thread_count(None, 64), 8);
        assert_eq!(thread_count(None, 2), 2);
        assert_eq!(thread_count(Some("3"), 64), 3);
        assert_eq!(thread_count(Some("0"), 4), 4);
        assert_eq!(thread_count(Some("x"), 4), 4);
    }

    #[cfg(feature = "testing")]
    #[test]
    fn the_failing_engine_fails_the_nth_operation() {
        let t = Temp::new("failing");
        let data = pattern(256);
        let f = file_with(&t, &data);
        let w = Temp::new("failing-w");
        let wf = Arc::new(IoFile::create(&w.0, 256).unwrap());
        let plan = FailPlan {
            read: Some((1, Fault::Fail(io::ErrorKind::Other))),
            write: Some((0, Fault::Short { keep: 10 })),
        };
        let io = FailingIo::new(PoolIo::new(1), plan);
        let ext = Extent { offset: 0, len: 256 };
        let mut buf = vec![MaybeUninit::<u8>::uninit(); 256];
        assert_eq!(read_blocking(&io, &f, ext, &mut buf).unwrap(), 256);
        let e = read_blocking(&io, &f, ext, &mut buf).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::Other);
        assert_eq!(read_blocking(&io, &f, ext, &mut buf).unwrap(), 256);
        assert_eq!(io.reads(), 3);

        let e = write_blocking(&io, &wf, ext, &data).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        let on_disk = std::fs::read(&w.0).unwrap();
        assert_eq!(&on_disk[..10], &data[..10]);
        assert!(on_disk[10..].iter().all(|&b| b == 0));
        assert_eq!(write_blocking(&io, &wf, ext, &data).unwrap(), 256);
        assert_eq!(io.writes(), 2);
        assert_eq!(std::fs::read(&w.0).unwrap(), data);
    }
}
