//! Kernels compiled on a pool of worker threads.
//!
//! A backend whose compiler is slow (NVVM and the driver's PTX JIT, Cranelift) hands the work
//! to [`RawKernel::spawn`] and returns at once; the kernel is waited for where it is first needed
//! ([`RawKernel::resolve`]). Kernels a recorded step needs compile side by side while the step is
//! still being traced.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

use super::{BackendError, NativeKernel};

/// Queued jobs per worker beyond which a caller compiles on its own thread instead of queueing.
const QUEUE_PER_WORKER: usize = 8;

/// Stack of a worker: compilers recurse over the kernel's expression trees.
const WORKER_STACK: usize = 32 << 20;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
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
        (workers > 0).then(|| Pool { queue: Mutex::new(VecDeque::new()), wake: Condvar::new(), workers })
    })
    .as_ref()
}

fn start_workers(pool: &'static Pool) {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        for index in 0..pool.workers {
            let spawned = std::thread::Builder::new().name(format!("luma-compile-{index}")).stack_size(WORKER_STACK).spawn(move || loop {
                let job = {
                    let mut queue = lock(&pool.queue);
                    loop {
                        if let Some(job) = queue.pop_front() {
                            break job;
                        }
                        queue = pool.wake.wait(queue).unwrap_or_else(PoisonError::into_inner);
                    }
                };
                job();
            });
            // Without a worker the queue would never drain: callers fall back to compiling inline
            // once it is full, so a failed spawn costs parallelism only.
            drop(spawned);
        }
    });
}

/// Runs `job` on a worker, or on the calling thread when the pool is off or its queue is full.
fn submit(job: Job) {
    let Some(pool) = pool() else { return job() };
    start_workers(pool);
    {
        let mut queue = lock(&pool.queue);
        if queue.len() < pool.workers * QUEUE_PER_WORKER {
            queue.push_back(job);
            drop(queue);
            pool.wake.notify_one();
            return;
        }
    }
    job();
}

/// The result of a compilation that may still be running.
pub(super) struct Slot {
    value: OnceLock<Result<Box<dyn NativeKernel>, BackendError>>,
    gate: Mutex<()>,
    done: Condvar,
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
        Slot { value: OnceLock::new(), gate: Mutex::new(()), done: Condvar::new() }
    }

    fn complete(&self, result: Result<Box<dyn NativeKernel>, BackendError>) {
        let _gate = lock(&self.gate);
        let _ = self.value.set(result);
        self.done.notify_all();
    }

    /// Waits for the compilation to finish.
    pub(super) fn wait(&self) -> &Result<Box<dyn NativeKernel>, BackendError> {
        if let Some(value) = self.value.get() {
            return value;
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

/// Starts `compile` on the pool and returns the slot it fills.
pub(super) fn spawn<N: NativeKernel>(compile: impl FnOnce() -> Result<N, BackendError> + Send + 'static) -> std::sync::Arc<Slot> {
    let slot = std::sync::Arc::new(Slot::empty());
    let target = std::sync::Arc::clone(&slot);
    submit(Box::new(move || {
        let result = match catch_unwind(AssertUnwindSafe(compile)) {
            Ok(result) => result.map(|native| Box::new(native) as Box<dyn NativeKernel>),
            Err(payload) => Err(BackendError::ShaderCompile(format!("the kernel compiler panicked: {}", panic_message(&*payload)))),
        };
        target.complete(result);
    }));
    slot
}
