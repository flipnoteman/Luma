pub(crate) mod built;
#[cfg(target_os = "linux")]
pub(crate) mod exit;
#[cfg(not(target_os = "linux"))]
pub(crate) mod exit {
    pub(crate) fn driver_call<R>(call: impl FnOnce() -> R) -> R {
        call()
    }
}
pub(crate) mod cache;
pub mod handle;
mod fma;
pub mod hash;
pub(crate) mod pager;
pub mod pool;

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, ignoring poisoning: every mutex in the crate guards data that stays consistent
/// if a holder panicked.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
