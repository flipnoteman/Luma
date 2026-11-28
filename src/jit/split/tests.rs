//! Failures of a commit: the outputs are poisoned and the error reaches everything that uses them.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::SplitMode;
use crate::{Array, Context, ContextOptions};

/// Slab at which the next stage fails (`usize::MAX`: never). One test at a time sets it (they are serialised by `LOCK`).
static FAIL_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn fail_here(slab: usize) -> bool {
    slab >= FAIL_AT.load(Ordering::Relaxed)
}

#[test]
fn a_failing_commit_poisons_the_outputs() {
    let _serial = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1024).split(SplitMode::Auto)).expect("a device is available");
    let n = 5 * 256 + 3;
    let x = Array::from_slice_on(&ctx, &vec![1.0f32; n], n).unwrap();
    let y = &x * 2.0;
    FAIL_AT.store(3, Ordering::Relaxed);
    // The commit runs when the GPU is idle (`eval`) or when the data is wanted; the error comes back from the call that triggered it.
    let evaluated = y.eval();
    FAIL_AT.store(usize::MAX, Ordering::Relaxed);
    let wanted = y.to_vec();
    let error = evaluated.err().or(wanted.err()).expect("the commit failed");
    assert!(format!("{error}").contains("injected failure"), "{error}");
    // Everything the failed steps wrote stays failed, whoever asks.
    assert!(y.to_vec().is_err());
    // A dependent step fails its own commit the same way.
    let z = &y + 1.0;
    assert!(z.to_vec().is_err());
    // The input is intact and the context works.
    assert_eq!(x.to_vec().unwrap(), vec![1.0f32; n]);
    let fresh = Array::from_slice_on(&ctx, &vec![3.0f32; n], n).unwrap();
    assert_eq!((&fresh + 1.0).to_vec().unwrap(), vec![4.0f32; n]);
}
