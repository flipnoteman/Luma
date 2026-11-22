//! Kernels compiled on a pool of worker threads.
//!
//! A backend whose compiler is slow (NVVM and the driver's PTX JIT, Cranelift) hands the work
//! to [`RawKernel::spawn`] and returns at once; the kernel is waited for where it is first needed
//! ([`RawKernel::resolve`]). Kernels a recorded step needs compile side by side while the step is
//! still being traced.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use super::{BackendError, NativeKernel};

/// Queued jobs per worker beyond which a caller compiles on its own thread instead of queueing.
const QUEUE_PER_WORKER: usize = 8;

/// Stack of a worker: compilers recurse over the kernel's expression trees.
const WORKER_STACK: usize = 32 << 20;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A queued job that a worker or the thread waiting for its result runs, whichever gets there first.
struct Task(Mutex<Option<Job>>);

impl Task {
    fn run(&self) {
        let job = lock(&self.0).take();
        if let Some(job) = job {
            job();
            // Where the driver's handlers are glibc's (see `hook_exit`); a registration per job
            // elsewhere would only fill the handler table.
            #[cfg(target_os = "linux")]
            if pool().is_some() {
                hook_exit();
            }
        }
    }
}

#[derive(Default)]
struct State {
    queue: VecDeque<Arc<Task>>,
    /// Jobs a worker is running.
    running: usize,
    /// Workers started (they are never stopped) and the ones of them waiting for a job.
    started: usize,
    waiting: usize,
    /// The process is exiting: nothing is queued any more.
    closed: bool,
}

struct Pool {
    state: Mutex<State>,
    wake: Condvar,
    idle: Condvar,
    workers: usize,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Worker count: `LUMA_COMPILE_THREADS` if set (`0` compiles every kernel on the calling thread),
/// otherwise the available parallelism.
fn worker_count() -> usize {
    std::env::var("LUMA_COMPILE_THREADS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = worker_count();
        (workers > 0).then(|| Pool { state: Mutex::new(State::default()), wake: Condvar::new(), idle: Condvar::new(), workers })
    })
    .as_ref()
}

/// How long exit waits for the kernels being compiled.
const EXIT_WAIT: Duration = Duration::from_secs(20);

/// Set when the exit handler has run: it is registered many times (see [`hook_exit`]) and does
/// its work once.
static EXITED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

/// A driver's own exit handler frees what a compile still running on a worker uses, which
/// crashes the process (a recorded step that is never replayed leaves its kernels compiling).
/// This handler is registered after the driver's, so it runs first: it drops the jobs still
/// queued and waits for the running ones. It has to stay registered after the driver's (see
/// [`hook_exit`]).
extern "C" fn at_exit() {
    if EXITED.swap(true, Ordering::AcqRel) {
        return;
    }
    let Some(pool) = pool() else { return };
    let mut state = lock(&pool.state);
    state.closed = true;
    state.queue.clear();
    let deadline = Instant::now() + EXIT_WAIT;
    while state.running > 0 {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else { return };
        state = pool.idle.wait_timeout(state, left).unwrap_or_else(PoisonError::into_inner).0;
    }
}

/// Registers the exit handler as the newest one. `exit` runs handlers in the reverse order of
/// their registration and the driver registers its own while it works (the PTX JIT,
/// `libnvidia-ptxjitcompiler`, is loaded by the first kernel a job loads, and registers the
/// destructors of its state then): a handler registered before them would run after them, with
/// the compiler's state freed under the job still running. So it is registered again whenever a
/// job has finished, and the older registrations return at once ([`EXITED`]).
fn hook_exit() {
    // SAFETY: `at_exit` is a plain `extern "C"` function that does not unwind.
    unsafe {
        atexit(at_exit);
    }
}

/// Registers the exit handler for the first job, which may still be running at exit.
fn hook_exit_once() {
    static HOOKED: OnceLock<()> = OnceLock::new();
    HOOKED.get_or_init(hook_exit);
}

/// Starts the worker `index`; it takes jobs until the process exits.
fn spawn_worker(pool: &'static Pool, index: usize) -> std::io::Result<()> {
    std::thread::Builder::new().name(format!("luma-compile-{index}")).stack_size(WORKER_STACK).spawn(move || {
        let mut state = lock(&pool.state);
        loop {
            if let Some(task) = state.queue.pop_front() {
                state.running += 1;
                drop(state);
                task.run();
                state = lock(&pool.state);
                state.running -= 1;
                pool.idle.notify_all();
            } else {
                state.waiting += 1;
                state = pool.wake.wait(state).unwrap_or_else(PoisonError::into_inner);
                state.waiting -= 1;
            }
        }
    })?;
    Ok(())
}

/// Runs `task` on a worker, or on the calling thread when the pool is off, closed or its queue is
/// full. Workers start on demand, when no waiting one can take the job: a compiler's first run
/// on a thread costs milliseconds (its per-thread state, the driver's binding of the context), so
/// a kernel compiled by itself must reach a worker that has compiled one before, not a new one.
fn submit(task: Arc<Task>) {
    let Some(pool) = pool() else { return task.run() };
    hook_exit_once();
    let mut state = lock(&pool.state);
    if state.closed || state.queue.len() >= pool.workers * QUEUE_PER_WORKER {
        drop(state);
        return task.run();
    }
    state.queue.push_back(task);
    if state.waiting >= state.queue.len() {
        drop(state);
        pool.wake.notify_one();
    } else if state.started < pool.workers {
        // Without a worker the queue would never drain: a failed start of the first one runs
        // the job here, any other costs parallelism only.
        if spawn_worker(pool, state.started).is_ok() {
            state.started += 1;
        } else if state.started == 0 {
            let task = state.queue.pop_back();
            drop(state);
            if let Some(task) = task {
                task.run();
            }
        }
    }
}

/// The result of a compilation that may still be running.
pub(super) struct Slot {
    value: OnceLock<Result<Box<dyn NativeKernel>, BackendError>>,
    gate: Mutex<()>,
    done: Condvar,
    /// The job that fills the slot, while it is queued (see [`Slot::wait`]).
    task: OnceLock<Weak<Task>>,
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value.get() {
            Some(Ok(native)) => native.fmt(f),
            Some(Err(error)) => write!(f, "failed: {error:?}"),
            None => f.write_str("compiling"),
        }
    }
}

impl Slot {
    pub(super) fn ready(native: impl NativeKernel) -> Slot {
        let slot = Slot::empty();
        let _ = slot.value.set(Ok(Box::new(native)));
        slot
    }

    fn empty() -> Slot {
        Slot { value: OnceLock::new(), gate: Mutex::new(()), done: Condvar::new(), task: OnceLock::new() }
    }

    fn complete(&self, result: Result<Box<dyn NativeKernel>, BackendError>) {
        let _gate = lock(&self.gate);
        let _ = self.value.set(result);
        self.done.notify_all();
    }

    /// Whether the compilation has finished.
    pub(super) fn is_ready(&self) -> bool {
        self.value.get().is_some()
    }

    /// Waits for the compilation to finish. A job no worker has begun is compiled here: the thread
    /// that needs the kernel at once (a single kernel, not one of a traced step) neither waits for a
    /// worker to wake nor pays for a thread that never compiled before.
    pub(super) fn wait(&self) -> &Result<Box<dyn NativeKernel>, BackendError> {
        if let Some(value) = self.value.get() {
            return value;
        }
        if let Some(task) = self.task.get().and_then(Weak::upgrade) {
            task.run();
            if let Some(value) = self.value.get() {
                return value;
            }
        }
        let mut gate = lock(&self.gate);
        loop {
            if let Some(value) = self.value.get() {
                return value;
            }
            gate = self.done.wait(gate).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload.downcast_ref::<&str>().copied().or_else(|| payload.downcast_ref::<String>().map(String::as_str)).unwrap_or("unknown panic")
}

/// Fills a slot with an error if dropped before it was filled (a job dropped at exit).
struct Completer(Option<Arc<Slot>>);

impl Completer {
    fn complete(mut self, result: Result<Box<dyn NativeKernel>, BackendError>) {
        if let Some(slot) = self.0.take() {
            slot.complete(result);
        }
    }
}

impl Drop for Completer {
    fn drop(&mut self) {
        if let Some(slot) = self.0.take() {
            slot.complete(Err(BackendError::ShaderCompile("the kernel was not compiled: the process is exiting".into())));
        }
    }
}

/// Starts `compile` on the pool and returns the slot it fills.
pub(super) fn spawn<N: NativeKernel>(compile: impl FnOnce() -> Result<N, BackendError> + Send + 'static) -> Arc<Slot> {
    let slot = Arc::new(Slot::empty());
    let target = Completer(Some(Arc::clone(&slot)));
    let task = Arc::new(Task(Mutex::new(Some(Box::new(move || {
        let result = match catch_unwind(AssertUnwindSafe(compile)) {
            Ok(result) => result.map(|native| Box::new(native) as Box<dyn NativeKernel>),
            Err(payload) => Err(BackendError::ShaderCompile(format!("the kernel compiler panicked: {}", panic_message(&*payload)))),
        };
        target.complete(result);
    })))));
    let _ = slot.task.set(Arc::downgrade(&task));
    submit(task);
    slot
}
