//! NVIDIA's Linux driver hangs, aborts or segfaults in `exit` when its `atexit` handler races
//! its worker threads: the `[vk..]` ones and the compiler workers it starts while a device or a
//! pipeline is created (they keep the name of the thread that started them). This handler,
//! registered later and so run first, waits until the GPU and those workers are idle; it
//! destroys nothing (see `IDLE` in `handle.rs`). The compiler workers keep going after the
//! pipelines they were started for were created (a backlog of recompiles of cached pipelines);
//! when one is still busy after [`DRIVER_WAIT`] the driver's handler would hang (it waits for a
//! lock the worker holds) or free memory the worker uses, so the process ends with `_exit`, with
//! the status `exit` was given, instead of running the driver's handler.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

/// The devices to wait for at exit. Devices of instances with validation on (debug builds) are
/// left out: with two or more devices in the process, the Khronos validation layer no longer
/// finds the later ones during `exit` and aborts in `vkWaitSemaphores`.
static DEVICES: Mutex<Vec<wgpu::Device>> = Mutex::new(Vec::new());
static REGISTER: Once = Once::new();

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
/// handler waits for.
pub(crate) fn driver_call<R>(call: impl FnOnce() -> R) -> R {
    let before = task_ids();
    let result = call();
    let after = task_ids();
    if let Ok(mut known) = DRIVER_THREADS.lock() {
        known.retain(|id| after.contains(id));
        known.extend(after.difference(&before));
    }
    result
}

/// Makes the exit handler wait for `device` too (unless its instance has `validated` layers, see
/// [`DEVICES`]), registering the handler on first use.
pub(crate) fn track(device: &wgpu::Device, validated: bool) {
    if !validated
        && let Ok(mut devices) = DEVICES.lock()
    {
        devices.push(device.clone());
    }
    REGISTER.call_once(|| {
        // SAFETY: `at_exit` is a plain `extern "C"` function that never unwinds (it catches
        // panics) and only touches statics that outlive `exit`.
        unsafe {
            on_exit(at_exit, std::ptr::null_mut());
        }
    });
}

/// `0` skips the wait, for short-lived programs that accept the driver's exit race.
const OPT_OUT_ENV: &str = "LUMA_EXIT_WAIT";

extern "C" fn at_exit(status: std::ffi::c_int, _arg: *mut std::ffi::c_void) {
    if std::env::var_os(OPT_OUT_ENV).is_some_and(|value| value == "0") {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // `try_lock`: a thread still running at exit may hold it, and exit must not wait on it.
        if let Ok(devices) = DEVICES.try_lock() {
            for device in devices.iter() {
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(GPU_WAIT) });
            }
        }
        super::cache::flush_all();
        if !wait_for_driver_workers() {
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
}
