//! NVIDIA's Linux driver hangs, aborts or segfaults in `exit` when its `atexit` handler races
//! its `[vk..]` worker threads. This handler, registered later and so run first, waits until
//! the GPU and those workers are idle; it destroys nothing (see `IDLE` in `handle.rs`).

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
const DRIVER_WAIT: Duration = Duration::from_secs(2);

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
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
            atexit(at_exit);
        }
    });
}

/// `0` skips the wait, for short-lived programs that accept the driver's exit race.
const OPT_OUT_ENV: &str = "LUMA_EXIT_WAIT";

extern "C" fn at_exit() {
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
        wait_for_driver_workers();
    }));
}

fn wait_for_driver_workers() {
    let Some(mut last) = driver_activity() else { return };
    let start = Instant::now();
    let mut quiet_since = Instant::now();
    while start.elapsed() < DRIVER_WAIT {
        std::thread::sleep(SAMPLE_EVERY);
        let Some(now) = driver_activity() else { return };
        if now.running || now.ticks != last.ticks {
            quiet_since = Instant::now();
        } else if quiet_since.elapsed() >= QUIET {
            return;
        }
        last = now;
    }
}

struct Activity {
    /// CPU time of the workers so far, in clock ticks.
    ticks: u64,
    running: bool,
}

/// The activity of the driver's worker threads, `None` if the process has none.
fn driver_activity() -> Option<Activity> {
    let mut found = None;
    for entry in std::fs::read_dir("/proc/self/task").ok()?.flatten() {
        let path = entry.path();
        let Ok(comm) = std::fs::read_to_string(path.join("comm")) else { continue };
        if !comm.starts_with("[vk") {
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
