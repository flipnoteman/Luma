pub(crate) mod built;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) mod exit;
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) mod exit {
    pub(crate) fn driver_call<R>(call: impl FnOnce() -> R) -> R {
        call()
    }
}
pub(crate) mod cache;
pub(crate) mod handle;
mod fma;
pub(crate) mod hash;
pub(crate) mod pager;
pub(crate) mod pool;

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, ignoring poisoning: every mutex in the crate guards data that stays consistent
/// if a holder panicked.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
