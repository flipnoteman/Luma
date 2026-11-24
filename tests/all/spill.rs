//! Array data beyond the memory budget moves to host memory and comes back on demand.

use luma::{Array, Context, ContextOptions};

/// A context with 64 KiB segments and a 1 MiB budget, so small arrays already spill.
fn small_budget() -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(1 << 16).memory_budget(1 << 20))
    .unwrap()
}

/// Distinct, exactly representable values.
fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + salt * 131) % 1021) as f32).collect()
}

#[test]
fn arrays_larger_than_the_budget_compute_correctly() {
    let ctx = small_budget();
    let len = 1 << 20;
    let (x, y) = (values(len, 1), values(len, 2));
    let a = Array::from_slice_on(&ctx, &x, len).unwrap();
    let b = Array::from_slice_on(&ctx, &y, len).unwrap();

    let c = &a * 2.0 + &b;
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x * 2.0 + y).collect();
    assert_eq!(c.to_vec().unwrap(), expected);

    let sum = c.sum().item().unwrap();
    let reference: f64 = expected.iter().map(|&v| f64::from(v)).sum();
    assert!((f64::from(sum) - reference).abs() <= reference * 1e-4, "{sum} vs {reference}");
    assert!(ctx.spilled_bytes() > 0);
}

#[test]
fn many_single_segment_arrays_survive_spilling() {
    let ctx = small_budget();
    let len = 16_384;
    let arrays: Vec<Array<f32>> = (0..40)
        .map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap())
        .collect();
    for (k, array) in arrays.iter().enumerate() {
        assert_eq!(array.to_vec().unwrap(), values(len, k), "array {k}");
    }
    assert!(ctx.spilled_bytes() > 0);
}

#[test]
fn spilled_arrays_keep_working_as_operands() {
    let ctx = small_budget();
    let len = 16_384;
    let mut arrays: Vec<Array<f32>> = (0..40)
        .map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap())
        .collect();
    assert!(ctx.spilled_bytes() > 0);

    // The first arrays were created long ago and have been moved out.
    let fresh = values(len, 99);
    arrays[0].write(&fresh).unwrap();
    assert_eq!(arrays[0].to_vec().unwrap(), fresh);

    arrays[1] += 1.0;
    let expected: Vec<f32> = values(len, 1).iter().map(|v| v + 1.0).collect();
    assert_eq!(arrays[1].to_vec().unwrap(), expected);

    let sum = (&arrays[2] + &arrays[3]).to_vec().unwrap();
    let expected: Vec<f32> = values(len, 2).iter().zip(values(len, 3)).map(|(a, b)| a + b).collect();
    assert_eq!(sum, expected);
}

#[test]
fn a_default_context_does_not_spill_small_workloads() {
    let ctx = Context::new().unwrap();
    let len = 1 << 17;
    let (x, y) = (values(len, 1), values(len, 2));
    let a = Array::from_slice_on(&ctx, &x, len).unwrap();
    let b = Array::from_slice_on(&ctx, &y, len).unwrap();
    let c = &a * 2.0 + &b;
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x * 2.0 + y).collect();
    assert_eq!(c.to_vec().unwrap(), expected);
    assert_eq!(ctx.spilled_bytes(), 0);
}

#[test]
fn eviction_while_tracing_an_iterate_step_completes() {
    let ctx = small_budget();
    let len = 16_384;
    // The state is the oldest resident segment; the fillers bring the context to its budget.
    let x = Array::from_slice_on(&ctx, &values(len, 0), len).unwrap();
    let fillers: Vec<Array<f32>> = (0..14).map(|k| Array::from_slice_on(&ctx, &values(len, k + 1), len).unwrap()).collect();
    let step = |[u]: &[Array<f32>; 1]| {
        let a = (u * 0.5 + 1.0).cache();
        let b = (&a * 0.5 + u).cache();
        let c = (&b * 0.25 + &a).cache();
        [c * 0.5]
    };
    let [got] = luma::iterate_all([&x], 5, step);
    let mut want = values(len, 0);
    for _ in 0..5 {
        for v in &mut want {
            let a = *v * 0.5 + 1.0;
            let b = a * 0.5 + *v;
            *v = (b * 0.25 + a) * 0.5;
        }
    }
    let got = got.to_vec().unwrap();
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= w.abs() * 1e-5 + 1e-5));
    for (k, f) in fillers.iter().enumerate() {
        assert_eq!(f.to_vec().unwrap(), values(len, k + 1));
    }
}

#[test]
fn readbacks_in_a_loop_under_a_budget_do_not_deadlock() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = small_budget();
        let len = 1 << 17;
        // Arrays of half the budget each: every step has to evict while it speculates.
        let held: Vec<Array<f32>> = (0..6).map(|k| Array::from_slice_on(&ctx, &values(len, k), len).unwrap()).collect();
        let mut x = Array::from_slice_on(&ctx, &values(len, 9), len).unwrap();
        let mut total = 0.0f64;
        // Enough steps for the loop to be recognised and its readbacks speculated, which is when
        // the eviction path re-entered the speculation lock.
        for step in 0..24 {
            x = &x * 0.5 + &held[step % held.len()];
            total += f64::from(x.sum().item().unwrap());
        }
        // Destroy the device's objects before reporting: a context still being dropped by this
        // thread when the test process exits races the exit-time teardown of the Vulkan driver.
        drop((held, x, ctx));
        let _ = done.send(total);
    });
    // Generous: a debug build on a small CI runner's software device is slow; a deadlock never ends.
    let total = finished.recv_timeout(std::time::Duration::from_secs(240)).expect("deadlock in a readback under a memory budget");
    assert!(total > 0.0);
}

/// `ctx` with 64 KiB segments, a 1 MiB device budget and `host` bytes of host budget.
fn with_host_budget(host: u64) -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(1 << 16).memory_budget(1 << 20).host_budget(Some(host))).unwrap()
}

/// Reads every array through a kernel (which only reads it), so a spilled one comes back.
fn sweep(arrays: &[Array<f32>]) -> Vec<Vec<f32>> {
    arrays.iter().map(|array| (array * 2.0).to_vec().unwrap()).collect()
}

fn twenty(ctx: &Context) -> Vec<Array<f32>> {
    (0..20).map(|k| Array::from_slice_on(ctx, &values(16_384, k), 16_384).unwrap()).collect()
}

#[test]
fn evicting_unchanged_data_reads_nothing_back() {
    let ctx = with_host_budget(1 << 30);
    let arrays = twenty(&ctx);
    assert!(ctx.spilled_bytes() > 0);
    let first = sweep(&arrays);
    let before = ctx.memory_stats();
    let second = sweep(&arrays);
    let after = ctx.memory_stats();
    assert_eq!(first, second);
    assert!(after.clean_evictions > before.clean_evictions, "{before:?} {after:?}");
    assert_eq!(after.readback_evictions, before.readback_evictions, "unchanged data was read back");
    assert!(after.host_copy_bytes > 0);
    for (k, got) in second.iter().enumerate() {
        let expected: Vec<f32> = values(16_384, k).iter().map(|v| v * 2.0).collect();
        assert_eq!(got, &expected, "array {k}");
    }
}

#[test]
fn a_write_invalidates_the_host_copy() {
    let ctx = with_host_budget(1 << 30);
    let mut arrays = twenty(&ctx);
    sweep(&arrays);
    sweep(&arrays);
    // Every array holds a copy by now; change one on the device.
    arrays[7] += 1.0;
    // Materialise the sum, so the kernel writes a buffer instead of fusing into the sweep.
    arrays[7].to_vec().unwrap();
    let before = ctx.memory_stats();
    sweep(&arrays);
    sweep(&arrays);
    let again = sweep(&arrays);
    let after = ctx.memory_stats();
    assert_eq!(after.readback_evictions - before.readback_evictions, 1, "{before:?} {after:?}");
    for (k, got) in again.iter().enumerate() {
        let expected: Vec<f32> = values(16_384, k).iter().map(|v| (v + if k == 7 { 1.0 } else { 0.0 }) * 2.0).collect();
        assert_eq!(got, &expected, "array {k}");
    }
}

#[test]
fn the_host_budget_drops_copies_first() {
    // Room for the spilled arrays only: no copy is kept, every eviction reads back.
    let ctx = with_host_budget(1 << 16);
    let arrays = twenty(&ctx);
    let first = sweep(&arrays);
    let second = sweep(&arrays);
    let stats = ctx.memory_stats();
    assert_eq!(first, second);
    assert_eq!(stats.host_copy_bytes, 0, "{stats:?}");
    assert_eq!(stats.clean_evictions, 0, "{stats:?}");

    // Room for a few copies: they never exceed what is left of the budget.
    let ctx = with_host_budget(6 << 16);
    let arrays = twenty(&ctx);
    let first = sweep(&arrays);
    let second = sweep(&arrays);
    let stats = ctx.memory_stats();
    assert_eq!(first, second);
    assert!(stats.host_copy_bytes <= 6 << 16, "{stats:?}");
    assert!(stats.host_copy_bytes > 0 || stats.spilled_bytes > 0);
}
