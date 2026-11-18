pub(crate) mod built;
pub(crate) mod fma;
pub(crate) mod hash;
pub(crate) mod lower;
pub(crate) mod pager;
pub(crate) mod pool;

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, ignoring poisoning: every mutex in the crate guards data that stays consistent
/// if a holder panicked.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
