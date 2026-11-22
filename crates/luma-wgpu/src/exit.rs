//! NVIDIA's Linux driver hangs, aborts or segfaults in `exit` when its `atexit` handler races
//! its worker threads: the `[vk..]` ones and the compiler workers it starts while a device or a
//! pipeline is created (they keep the name of the thread that started them). This handler,
//! registered later and so run first, waits until the GPU and those workers are idle; it
//! destroys nothing (see `IDLE` in `device.rs`). The compiler workers keep going after the
//! pipelines they were started for were created (a backlog of recompiles of cached pipelines);
//! when one is still busy after [`DRIVER_WAIT`] the driver's handler would hang (it waits for a
//! lock the worker holds) or free memory the worker uses, so the process ends with `_exit`, with
//! the status `exit` was given, instead of running the driver's handler.
//!
//! "Registered later" has to stay true while the driver works. `exit` runs the handlers newest
//! first and the driver registers more of its own as it goes: `libnvidia-gpucomp`, its compiler,
//! registers the destructors of its static state when the first pipeline is compiled, long after
//! the device exists. A handler registered once, with the device, ran after them: it found the
//! state a compiler worker was still using freed and the worker crashed (`SIGSEGV` in
//! `libnvidia-gpucomp`, while the handler wrote the pipeline cache). The handler is therefore
//! registered again after every call into the driver ([`driver_call`]); the newest registration
//! does the work and the older ones return at once.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::device::PipelineStore;

/// The devices to wait for at exit. Devices of instances with validation on (debug builds) are
/// left out: with two or more devices in the process, the Khronos validation layer no longer
/// finds the later ones during `exit` and aborts in `vkWaitSemaphores`.
static DEVICES: Mutex<Vec<wgpu::Device>> = Mutex::new(Vec::new());
/// The pipeline caches the exit handler writes.
static STORES: Mutex<Vec<Arc<PipelineStore>>> = Mutex::new(Vec::new());
/// Set when the exit handler has run: it is registered many times (see [`arm`]) and does its
/// work once.
static DONE: AtomicBool = AtomicBool::new(false);

const GPU_WAIT: Duration = Duration::from_secs(2);
/// How long the driver's workers must stay idle before exit goes ahead.
const QUIET: Duration = Duration::from_millis(50);
const SAMPLE_EVERY: Duration = Duration::from_millis(5);
/// How long a busy driver worker is waited for before the process is ended without the driver's
/// handler.
const DRIVER_WAIT: Duration = Duration::from_secs(2);

/// The threads the driver started during calls into it that [`driver_call`] wraps.
static DRIVER_THREADS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

unsafe extern "C" {
    /// glibc: like `atexit`, but the handler gets the status `exit` was called with.
    fn on_exit(callback: extern "C" fn(std::ffi::c_int, *mut std::ffi::c_void), arg: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn _exit(status: std::ffi::c_int) -> !;
}

/// The ids of the threads of this process.
fn task_ids() -> HashSet<u32> {
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else { return HashSet::new() };
    dir.flatten().filter_map(|entry| entry.file_name().to_str()?.parse().ok()).collect()
}

/// Runs `call`, a call into the driver (creating a device or a pipeline), and remembers the
/// threads that exist after it and did not before as the driver's workers, which the exit
/// handler waits for. The driver may have registered exit handlers of its own during the call,
/// so the exit handler is registered again after it (see [`arm`]).
pub(crate) fn driver_call<R>(call: impl FnOnce() -> R) -> R {
    let before = task_ids();
    let result = call();
    let after = task_ids();
    if let Ok(mut known) = DRIVER_THREADS.lock() {
        known.retain(|id| after.contains(id));
        known.extend(after.difference(&before));
    }
    arm();
    result
}

/// Registers the exit handler as the newest one. `exit` runs the handlers in the reverse order of
/// their registration: one registered before the driver's would run after them, when the driver
/// has freed what its compiler workers still use. A registration costs a few nanoseconds and a
/// list slot; all but the first handler to run return at once ([`DONE`]).
fn arm() {
    // SAFETY: `at_exit` is a plain `extern "C"` function that never unwinds (it catches
    // panics) and only touches statics that outlive `exit`.
    unsafe {
        on_exit(at_exit, std::ptr::null_mut());
    }
}

/// Makes the exit handler wait for `device` too (unless its instance has `validated` layers, see
/// [`DEVICES`]), registering the handler.
pub(crate) fn track(device: &wgpu::Device, validated: bool) {
    if !validated
        && let Ok(mut devices) = DEVICES.lock()
    {
        devices.push(device.clone());
    }
    arm();
}

/// Has the exit handler write the pipeline cache of `store` too. Not for devices of instances
/// with validation layers (see [`track`]): at exit the layer no longer finds them and aborts on
/// the first call.
pub(crate) fn flush_at_exit(store: &Arc<PipelineStore>) {
    if store.is_active()
        && let Ok(mut stores) = STORES.lock()
    {
        stores.push(Arc::clone(store));
    }
}

/// Writes every cache registered with [`flush_at_exit`] that has unwritten pipelines.
fn flush_all() {
    // `try_lock`: this runs at exit, when a thread still running may hold the list.
    if let Ok(stores) = STORES.try_lock() {
        for store in stores.iter() {
            store.flush();
        }
    }
}

/// `0` skips the wait, for short-lived programs that accept the driver's exit race.
const OPT_OUT_ENV: &str = "LUMA_EXIT_WAIT";

extern "C" fn at_exit(status: std::ffi::c_int, _arg: *mut std::ffi::c_void) {
    if DONE.swap(true, Ordering::AcqRel) || std::env::var_os(OPT_OUT_ENV).is_some_and(|value| value == "0") {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // `try_lock`: a thread still running at exit may hold it, and exit must not wait on it.
        if let Ok(devices) = DEVICES.try_lock() {
            for device in devices.iter() {
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(GPU_WAIT) });
            }
        }
        flush_all();
        // A process that opened several devices (a test binary whose tests each have a context)
        // has a `[vkps] Update` thread per device, and the driver's own handler segfaulted in one
        // of them on every exit of the 400 unit tests on an RTX 3090 (driver 615.71) even though
        // all were idle: with more than one device the driver's handler is never run.
        let several = DEVICES.try_lock().is_ok_and(|devices| devices.len() > 1);
        if several || !wait_for_driver_workers() {
            // SAFETY: the GPU is idle and the pipeline cache is written. What is left to run is
            // the driver's own handler, which would hang on a lock the busy worker holds or free
            // memory it uses; what other handlers release the kernel releases too.
            unsafe { _exit(status) }
        }
    }));
}

/// Waits until the driver's workers are idle; `false` if some are still busy after [`DRIVER_WAIT`].
fn wait_for_driver_workers() -> bool {
    let Some(mut last) = driver_activity() else { return true };
    let start = Instant::now();
    let mut quiet_since = Instant::now();
    while start.elapsed() < DRIVER_WAIT {
        std::thread::sleep(SAMPLE_EVERY);
        let Some(now) = driver_activity() else { return true };
        if now.running || now.ticks != last.ticks {
            quiet_since = Instant::now();
        } else if quiet_since.elapsed() >= QUIET {
            return true;
        }
        last = now;
    }
    false
}

struct Activity {
    /// CPU time of the workers so far, in clock ticks.
    ticks: u64,
    running: bool,
}

/// The activity of the driver's worker threads, `None` if the process has none.
fn driver_activity() -> Option<Activity> {
    let mut found = None;
    let started = DRIVER_THREADS.try_lock().map(|known| known.clone()).unwrap_or_default();
    for entry in std::fs::read_dir("/proc/self/task").ok()?.flatten() {
        let path = entry.path();
        let Ok(comm) = std::fs::read_to_string(path.join("comm")) else { continue };
        let started_by_driver = entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()).is_some_and(|tid| started.contains(&tid));
        if !comm.starts_with("[vk") && !started_by_driver {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(path.join("stat")) else { continue };
        // After the parenthesized name: state, then utime and stime as the 12th and 13th fields.
        let Some((_, rest)) = stat.rsplit_once(')') else { continue };
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let ticks = |at: usize| fields.get(at).and_then(|field| field.parse::<u64>().ok()).unwrap_or(0);
        let activity = found.get_or_insert(Activity { ticks: 0, running: false });
        activity.ticks += ticks(11) + ticks(12);
        activity.running |= fields.first() == Some(&"R");
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_started_during_a_driver_call_are_the_drivers() {
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let mut spawned = None;
        let worker = driver_call(|| {
            let handle = std::thread::spawn(move || {
                let _ = wait.recv();
            });
            // The thread exists once its id shows in the task list: give it a moment to start.
            std::thread::sleep(Duration::from_millis(50));
            spawned = Some(());
            handle
        });
        assert!(spawned.is_some());
        let known = DRIVER_THREADS.lock().map(|known| known.clone()).unwrap_or_default();
        assert!(!known.is_empty(), "the thread started inside the call is remembered");
        let _ = go.send(());
        let _ = worker.join();
    }

    /// `exit` runs the handlers newest first and the driver registers its own while it works: a
    /// handler armed again afterwards must run before them, one armed only once runs after.
    /// Each case runs in a child process that ends with `exit`.
    #[test]
    fn arming_again_puts_the_handler_before_later_registrations() {
        const CHILD: &str = "LUMA_EXIT_ORDER_CHILD";
        const NAME: &str = "exit::tests::arming_again_puts_the_handler_before_later_registrations";
        unsafe extern "C" {
            fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
        }
        /// Stands for a driver's handler: reports whether ours has run by now.
        extern "C" fn driver_handler() {
            println!("driver handler: ours has run = {}", DONE.load(Ordering::SeqCst));
        }
        if let Ok(case) = std::env::var(CHILD) {
            arm();
            // SAFETY: `driver_handler` is a plain `extern "C"` function that does not unwind.
            unsafe { atexit(driver_handler) };
            if case == "again" {
                arm();
            }
            std::process::exit(0);
        }
        let run = |case: &str| {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture", "--test-threads=1"])
                .env(CHILD, case)
                .output()
                .unwrap();
            String::from_utf8_lossy(&output.stdout).into_owned()
        };
        assert!(run("again").contains("ours has run = true"), "armed again after the driver's handler");
        assert!(run("once").contains("ours has run = false"), "armed only before the driver's handler");
    }
}
