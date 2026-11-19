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

    let temporaries: Vec<Array<f32>> = (0..4).map(|k| &kept * k as f32).collect();
    for t in &temporaries {
        t.eval().unwrap();
    }
    drop(temporaries);
    let idle = ctx.memory_stats();
    assert!(idle.pooled_bytes >= 4 * 4 * len as u64, "dropped arrays are pooled: {idle:?}");

    ctx.trim().unwrap();
    let trimmed = ctx.memory_stats();
    assert_eq!(trimmed.pooled_bytes, 0, "{trimmed:?}");
    assert_eq!(trimmed.resident_bytes, before.resident_bytes, "array data keeps its memory");
    assert_eq!(kept.to_vec().unwrap()[len / 2], 1.0);
}
