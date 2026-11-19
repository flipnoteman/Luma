//! `Context::memory_stats` and `Context::trim`: dropped arrays leave idle buffers in the pool,
//! `trim` gives them back to the device, and array data keeps its memory.

use luma::{Array, Context};

#[test]
fn trim_releases_idle_buffers_and_keeps_array_data() {
    let ctx = Context::new().unwrap();
    let len = 1 << 18;
    let kept = Array::from_slice_on(&ctx, &vec![1.0f32; len], len).unwrap();
    let doubled = (&kept * 2.0).to_vec().unwrap();
    assert_eq!(doubled[len - 1], 2.0);
    let before = ctx.memory_stats();
    assert!(before.resident_bytes >= 4 * len as u64, "{before:?}");

    // Multipliers other than 1 (`x * 1` is no operation and makes no buffer); the sync makes the
    // evaluations happen even while the GPU is busy, when `eval` may postpone them.
    let temporaries: Vec<Array<f32>> = (2..6).map(|k| &kept * k as f32).collect();
    for t in &temporaries {
        t.eval().unwrap();
    }
    ctx.sync().unwrap();
    drop(temporaries);
    let idle = ctx.memory_stats();
    assert!(idle.pooled_bytes >= 4 * 4 * len as u64, "dropped arrays are pooled: {idle:?}");

    ctx.trim().unwrap();
    let trimmed = ctx.memory_stats();
    assert_eq!(trimmed.pooled_bytes, 0, "{trimmed:?}");
    assert_eq!(trimmed.resident_bytes, before.resident_bytes, "array data keeps its memory");
    assert_eq!(kept.to_vec().unwrap()[len / 2], 1.0);
}

/// A compiled loop keeps work buffers (the stop rule's history among them) for as long as it
/// lives: they count as resident, and leave the count when the loop is dropped.
#[test]
fn a_compiled_loop_counts_the_buffers_it_keeps() {
    use luma::{Until, compile_until_all};

    type One = (Array<f32>,);
    let ctx = Context::new().unwrap();
    let state = (Array::from_slice_on(&ctx, &[0.0f32; 8], 8).unwrap(),);
    state.0.eval().unwrap();
    ctx.sync().unwrap();
    let before = ctx.memory_stats().resident_bytes;

    let steps = 100_000;
    let until = Until::new(steps).record("v", |s: &One| s.0.max()).stop_when(|h| h["v"].last().ge(1e9));
    let compiled = compile_until_all(&state, until, |s: &One, k| (&s.0 + k,)).unwrap();
    let kept = ctx.memory_stats().resident_bytes - before;
    assert!(kept >= 4 * steps as u64, "the history of {steps} steps is device memory: {kept} bytes counted");

    drop(compiled);
    assert!(ctx.memory_stats().resident_bytes <= before, "the loop's buffers leave the count with it");
}
