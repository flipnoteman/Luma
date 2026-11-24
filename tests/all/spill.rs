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
    let before = ctx.paging_stats();
    let second = sweep(&arrays);
    let after = ctx.paging_stats();
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
    let before = ctx.paging_stats();
    sweep(&arrays);
    sweep(&arrays);
    let again = sweep(&arrays);
    let after = ctx.paging_stats();
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
    let stats = ctx.paging_stats();
    assert_eq!(first, second);
    assert_eq!(stats.host_copy_bytes, 0, "{stats:?}");
    assert_eq!(stats.clean_evictions, 0, "{stats:?}");

    // Room for a few copies: they never exceed what is left of the budget.
    let ctx = with_host_budget(6 << 16);
    let arrays = twenty(&ctx);
    let first = sweep(&arrays);
    let second = sweep(&arrays);
    let stats = ctx.paging_stats();
    assert_eq!(first, second);
    assert!(stats.host_copy_bytes <= 6 << 16, "{stats:?}");
    assert!(stats.host_copy_bytes > 0 || ctx.memory_stats().spilled_bytes > 0);
}

/// Elements of an array that is one megabyte: big enough that recomputing it costs less than moving it over the link.
const MEGABYTE: usize = 1 << 18;

/// A context whose segments hold a megabyte and whose budget holds six of them.
fn lineage_context() -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(1 << 20).memory_budget(6 << 20)).unwrap()
}

fn megabyte(salt: usize) -> Vec<f32> {
    values(MEGABYTE, salt)
}

/// Creates arrays until `done` holds of the context's statistics: the least recently used segment is evicted first.
fn pressure(ctx: &Context, done: impl Fn(&luma::PagingStats) -> bool) -> Vec<Array<f32>> {
    let mut fillers = Vec::new();
    while !done(&ctx.paging_stats()) {
        assert!(fillers.len() < 24, "{:?}", ctx.paging_stats());
        fillers.push(Array::from_slice_on(ctx, &megabyte(100 + fillers.len()), MEGABYTE).unwrap());
    }
    fillers
}

/// `c = a * 2 + b` of two fresh arrays, computed and evicted by lineage, with the arrays and the fillers that kept the pressure on.
fn evicted_by_lineage(ctx: &Context) -> (Array<f32>, Array<f32>, Array<f32>, Vec<Array<f32>>) {
    let a = Array::from_slice_on(ctx, &megabyte(1), MEGABYTE).unwrap();
    let b = Array::from_slice_on(ctx, &megabyte(2), MEGABYTE).unwrap();
    let c = &a * 2.0 + &b;
    c.eval().unwrap();
    ctx.flush().unwrap();
    let fillers = pressure(ctx, |stats| stats.recompute_evictions >= 1);
    (a, b, c, fillers)
}

fn expected_c(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a * 2.0 + b).collect()
}

#[test]
fn a_cheap_intermediate_is_recomputed_not_read_back() {
    let ctx = lineage_context();
    let before = ctx.paging_stats().readback_evictions;
    let (_a, _b, c, _fillers) = evicted_by_lineage(&ctx);
    let stats = ctx.paging_stats();
    assert_eq!(stats.recompute_evictions, 1, "{stats:?}");
    assert_eq!(stats.readback_evictions, before, "the intermediate was read back: {stats:?}");
    assert!(stats.lineage_pinned_bytes >= 2 << 20, "{stats:?}");
    let got = c.to_vec().unwrap();
    let want = expected_c(&megabyte(1), &megabyte(2));
    assert!(got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()), "the recomputed value differs");
}

#[test]
fn writing_an_input_materialises_dependents_first() {
    let ctx = lineage_context();
    let (mut a, _b, c, _fillers) = evicted_by_lineage(&ctx);
    a.write(&megabyte(7)).unwrap();
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)), "the dependent saw the new input");
    assert_eq!(a.to_vec().unwrap(), megabyte(7));
}

#[test]
fn a_two_cycle_write_does_not_resurrect_stale_lineage() {
    let ctx = lineage_context();
    let (mut a, _b, c, fillers) = evicted_by_lineage(&ctx);
    let old = expected_c(&megabyte(1), &megabyte(2));
    a.write(&megabyte(7)).unwrap();
    assert_eq!(c.to_vec().unwrap(), old);
    // Make `c` the least recently used segment again, with both inputs resident, and push it
    // out: it has no lineage any more, so it is read back.
    drop(fillers);
    // A kernel pins what it reads; a host read of an idle buffer does not.
    (&a * 2.0).to_vec().unwrap();
    (&_b * 2.0).to_vec().unwrap();
    let before = ctx.paging_stats();
    let _fresh = pressure(&ctx, |stats| stats.readback_evictions + stats.recompute_evictions > before.readback_evictions + before.recompute_evictions);
    let after = ctx.paging_stats();
    assert_eq!(after.recompute_evictions, before.recompute_evictions, "{before:?} {after:?}");
    assert_eq!(after.readback_evictions, before.readback_evictions + 1, "{before:?} {after:?}");
    assert_eq!(c.to_vec().unwrap(), old, "the second round used the new input");
}

/// `c` resident, with its lineage, when the input is written; evicting it afterwards must not recompute it.
#[test]
fn writing_an_input_of_a_resident_dependent_drops_its_lineage() {
    let ctx = lineage_context();
    let mut a = Array::from_slice_on(&ctx, &megabyte(1), MEGABYTE).unwrap();
    let b = Array::from_slice_on(&ctx, &megabyte(2), MEGABYTE).unwrap();
    let c = &a * 2.0 + &b;
    c.eval().unwrap();
    ctx.flush().unwrap();
    a.write(&megabyte(7)).unwrap();
    let _fillers = pressure(&ctx, |stats| stats.readback_evictions >= 1);
    let stats = ctx.paging_stats();
    assert_eq!(stats.recompute_evictions, 0, "{stats:?}");
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
}

/// A compound assignment writes through the kernel's own pin.
#[test]
fn an_in_place_update_of_an_input_keeps_the_dependent_old() {
    let ctx = lineage_context();
    let (mut a, _b, c, _fillers) = evicted_by_lineage(&ctx);
    a += 1.0;
    a.eval().unwrap();
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
    let new: Vec<f32> = megabyte(1).iter().map(|v| v + 1.0).collect();
    assert_eq!(a.to_vec().unwrap(), new);
}

#[test]
fn dropping_an_input_keeps_the_dependent_correct() {
    let ctx = lineage_context();
    let (a, b, c, _fillers) = evicted_by_lineage(&ctx);
    drop(a);
    drop(b);
    assert!(ctx.paging_stats().lineage_pinned_bytes >= 2 << 20);
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
    assert_eq!(ctx.paging_stats().lineage_pinned_bytes, 0, "the recomputed segment still holds its inputs");
}

#[test]
fn no_lineage_eviction_happens_while_recording() {
    let ctx = lineage_context();
    let x = Array::from_slice_on(&ctx, &megabyte(0), MEGABYTE).unwrap();
    let fillers: Vec<Array<f32>> = (0..3).map(|k| Array::from_slice_on(&ctx, &megabyte(k + 1), MEGABYTE).unwrap()).collect();
    let step = |[u]: &[Array<f32>; 1]| {
        let a = (u * 0.5 + 1.0).cache();
        let b = (&a * 0.5 + u).cache();
        let c = (&b * 0.25 + &a).cache();
        [c * 0.5]
    };
    let [got] = luma::iterate_all([&x], 5, step);
    let mut want = megabyte(0);
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
        assert_eq!(f.to_vec().unwrap(), megabyte(k + 1));
    }
    assert_eq!(ctx.paging_stats().recompute_evictions, 0, "something recorded during a trace was evicted by lineage");
}

/// `c = a * 2 + b`, then `write` on `a` (through some path), then pressure that pushes `c` out
/// again, then a read of `c`: it must have the old values. `evicted`: `c` is evicted by lineage
/// before the write (else it is resident, with its lineage, at the write).
fn a_write_path_keeps_the_dependent_old(evicted: bool, write: impl FnOnce(&mut Array<f32>, &Context)) {
    let ctx = lineage_context();
    let (mut a, b, c, _fillers) = if evicted {
        evicted_by_lineage(&ctx)
    } else {
        let a = Array::from_slice_on(&ctx, &megabyte(1), MEGABYTE).unwrap();
        let b = Array::from_slice_on(&ctx, &megabyte(2), MEGABYTE).unwrap();
        let c = &a * 2.0 + &b;
        c.eval().unwrap();
        ctx.flush().unwrap();
        (a, b, c, Vec::new())
    };
    write(&mut a, &ctx);
    // Keep both inputs the most recently used, so that `c` is the next victim whatever it costs.
    (&a * 2.0).to_vec().unwrap();
    (&b * 2.0).to_vec().unwrap();
    let before = ctx.paging_stats();
    let _more = pressure(&ctx, |stats| stats.readback_evictions + stats.recompute_evictions > before.readback_evictions + before.recompute_evictions);
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
}

#[test]
fn a_region_assignment_keeps_the_dependent_old() {
    for evicted in [true, false] {
        a_write_path_keeps_the_dependent_old(evicted, |a, _| a.slice_mut(0, 0..4096).assign(-1.0).unwrap());
    }
}

#[test]
fn a_scatter_keeps_the_dependent_old() {
    for evicted in [true, false] {
        a_write_path_keeps_the_dependent_old(evicted, |a, ctx| {
            let at = Array::<u32>::from_slice_on(ctx, &[0, 7, 4096, 99_999], 4).unwrap();
            a.put(&at, -1.0).unwrap();
        });
    }
}

#[test]
fn a_scatter_add_keeps_the_dependent_old() {
    for evicted in [true, false] {
        a_write_path_keeps_the_dependent_old(evicted, |a, ctx| {
            let at = Array::<u32>::from_slice_on(ctx, &[0, 7, 4096, 99_999], 4).unwrap();
            a.put_add(&at, 1.0).unwrap();
        });
    }
}

#[test]
fn a_compound_assignment_keeps_the_dependent_old() {
    for evicted in [true, false] {
        a_write_path_keeps_the_dependent_old(evicted, |a, _| {
            *a += 1.0;
            a.eval().unwrap();
        });
    }
}

#[test]
fn a_write_keeps_the_dependent_old() {
    for evicted in [true, false] {
        a_write_path_keeps_the_dependent_old(evicted, |a, _| a.write(&megabyte(7)).unwrap());
    }
}

/// A solely owned input consumed by a store kernel gives its buffer to the result.
#[test]
fn an_input_taken_over_by_a_kernel_keeps_the_dependent_old() {
    let ctx = lineage_context();
    let a = Array::from_slice_on(&ctx, &megabyte(1), MEGABYTE).unwrap();
    let b = Array::from_slice_on(&ctx, &megabyte(2), MEGABYTE).unwrap();
    let c = &a * 2.0 + &b;
    c.eval().unwrap();
    ctx.flush().unwrap();
    let d = a * 3.0;
    d.eval().unwrap();
    ctx.flush().unwrap();
    (&b * 2.0).to_vec().unwrap();
    let before = ctx.paging_stats();
    let _more = pressure(&ctx, |stats| stats.readback_evictions + stats.recompute_evictions > before.readback_evictions + before.recompute_evictions);
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
    assert_eq!(d.to_vec().unwrap(), megabyte(1).iter().map(|v| v * 3.0).collect::<Vec<f32>>());
}

/// A state advanced by a loop, whose launches write buffers that the loop pins once.
#[test]
fn a_loop_state_written_by_held_pins_keeps_the_dependent_old() {
    let ctx = lineage_context();
    let b = Array::from_slice_on(&ctx, &megabyte(2), MEGABYTE).unwrap();
    let mut x = Array::from_slice_on(&ctx, &megabyte(1), MEGABYTE).unwrap();
    let c = &x * 2.0 + &b;
    c.eval().unwrap();
    ctx.flush().unwrap();
    for _ in 0..30 {
        x = &x * 0.5 + &b;
        x.eval().unwrap();
        (&x * 1.0).to_vec().unwrap();
    }
    (&b * 2.0).to_vec().unwrap();
    let before = ctx.paging_stats();
    let _more = pressure(&ctx, |stats| stats.readback_evictions + stats.recompute_evictions > before.readback_evictions + before.recompute_evictions);
    assert_eq!(c.to_vec().unwrap(), expected_c(&megabyte(1), &megabyte(2)));
}
