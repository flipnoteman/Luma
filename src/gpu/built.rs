//! A cache of values that cost a lot to build (compiled kernels), built outside the cache's lock.
//!
//! Compiling one kernel takes from a millisecond to many seconds in the driver. A cache that is
//! one mutex around a map and compiles under it makes every thread of a context wait for every
//! compile, hits included. Here a miss only claims the entry: threads that miss on different keys
//! build in parallel, and a thread that asks for a key somebody is building waits for that one
//! build instead of repeating it.

use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::gpu::hash::FxMap;
use crate::gpu::lock;

/// One key's value: empty until a build succeeded. The mutex is held while it is built.
type Slot<V> = Arc<Mutex<Option<V>>>;

/// Values by key, built at most once each (a failed build leaves the key empty: the next ask builds again).
#[derive(Debug)]
pub(crate) struct BuildCache<K, V> {
    slots: Mutex<FxMap<K, Slot<V>>>,
}

impl<K, V> Default for BuildCache<K, V> {
    fn default() -> Self {
        BuildCache { slots: Mutex::new(FxMap::default()) }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> BuildCache<K, V> {
    /// The value of `key`, from `build` unless it is built already; also whether `build` ran
    /// here. `build` runs without the lock of the cache: only a thread asking for this key waits
    /// for it.
    pub(crate) fn get_or_build(&self, key: &K, build: impl FnOnce() -> Result<V>) -> Result<(V, bool)> {
        let slot = {
            let mut slots = lock(&self.slots);
            match slots.get(key) {
                Some(slot) => Arc::clone(slot),
                None => Arc::clone(slots.entry(key.clone()).or_default()),
            }
        };
        let mut value = lock(&slot);
        if let Some(built) = value.as_ref() {
            return Ok((built.clone(), false));
        }
        let built = build()?;
        *value = Some(built.clone());
        Ok((built, true))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::error::LumaError;

    #[test]
    fn a_value_is_built_once_however_many_threads_ask() {
        let cache = BuildCache::<u32, u32>::default();
        let builds = AtomicUsize::new(0);
        let built_here: usize = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        let (value, built) = cache
                            .get_or_build(&1, || {
                                builds.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(Duration::from_millis(20));
                                Ok(7)
                            })
                            .unwrap();
                        assert_eq!(value, 7);
                        usize::from(built)
                    })
                })
                .collect();
            threads.into_iter().map(|thread| thread.join().unwrap()).sum()
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(built_here, 1, "exactly one caller reports having built it");
    }

    /// Two threads that need different values are inside their builds at the same time (one that waited
    /// for the other would never see the other inside, and gives up after five seconds).
    #[test]
    fn different_keys_build_at_the_same_time() {
        let cache = BuildCache::<u32, u32>::default();
        let inside = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for key in 0..2u32 {
                let (cache, inside) = (&cache, &inside);
                scope.spawn(move || {
                    cache
                        .get_or_build(&key, || {
                            inside.fetch_add(1, Ordering::SeqCst);
                            let deadline = Instant::now() + Duration::from_secs(5);
                            while inside.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
                                std::thread::yield_now();
                            }
                            Ok(key)
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(inside.load(Ordering::SeqCst), 2);
        // A hit on one key does not wait for a build of another.
        let building = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                cache
                    .get_or_build(&9, || {
                        building.wait();
                        release.wait();
                        Ok(9)
                    })
                    .unwrap();
            });
            building.wait();
            let (value, built) = cache.get_or_build(&0, || Ok(100)).unwrap();
            assert_eq!((value, built), (0, false));
            release.wait();
        });
    }

    #[test]
    fn a_failed_build_is_tried_again() {
        let cache = BuildCache::<u32, u32>::default();
        assert!(matches!(cache.get_or_build(&3, || Err(LumaError::Gpu("no".into()))), Err(LumaError::Gpu(_))));
        assert_eq!(cache.get_or_build(&3, || Ok(5)).unwrap(), (5, true));
        assert_eq!(cache.get_or_build(&3, || Ok(6)).unwrap(), (5, false));
    }
}
