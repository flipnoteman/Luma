use super::batch::{BATCH_LIMIT, SUBMIT_HOLD};
use super::transfer::{STAGING_USAGE, SYNC_SPIN_MIN, WAIT_SAMPLES};
use super::*;
use crate::array::Array;

/// A context with the calibration off and the cost model pinned, so that kernel counts do not
/// depend on the machine.
fn ctx() -> Context {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    ctx.set_cost_model(CostModel::TEST);
    ctx
}

#[test]
fn spin_budget_follows_the_recent_waits_up_to_the_cap() {
    let waits = WaitHistory::new();
    assert_eq!(waits.budget(), SYNC_SPIN_MIN);

    // Short waits keep the minimum; one of 1 ms makes the next ones poll for 1.5 ms.
    waits.record(Duration::from_micros(80));
    assert_eq!(waits.budget(), SYNC_SPIN_MIN);
    waits.record(Duration::from_millis(1));
    assert_eq!(waits.budget(), Duration::from_micros(1500));

    // Waits too long to be caught by polling fall back to the minimum.
    waits.record(Duration::from_millis(3));
    assert_eq!(waits.budget(), SYNC_SPIN_MIN);
    waits.record(Duration::from_secs(5));
    assert_eq!(waits.budget(), SYNC_SPIN_MIN);

    // The outlier is forgotten after `WAIT_SAMPLES` more waits.
    for _ in 0..WAIT_SAMPLES {
        waits.record(Duration::from_micros(200));
    }
    assert_eq!(waits.budget(), SYNC_SPIN_MIN);
    for _ in 0..WAIT_SAMPLES {
        waits.record(Duration::from_millis(2));
    }
    assert_eq!(waits.budget(), Duration::from_millis(3));
}

#[test]
fn recycled_buffers_are_reused_per_size_class_and_usage() {
    let ctx = ctx();

    let a = ctx.alloc_buffer(2400, STORAGE_USAGE).unwrap();
    assert_eq!(a.size(), 4096);
    ctx.recycle(a.clone());

    // Same class and usage: the recycled buffer comes back, even for a different byte size.
    let b = ctx.alloc_buffer(4000, STORAGE_USAGE).unwrap();
    assert_eq!(a, b);

    // Different usage or different class: a new buffer.
    ctx.recycle(b.clone());
    assert_ne!(ctx.alloc_buffer(4000, STAGING_USAGE).unwrap(), a);
    assert_ne!(ctx.alloc_buffer(4100, STORAGE_USAGE).unwrap(), a);
    // ... while the recycled one is still waiting in the pool.
    assert_eq!(ctx.alloc_buffer(4000, STORAGE_USAGE).unwrap(), a);
}

#[test]
fn dropping_an_array_returns_its_buffer_to_the_pool() {
    let ctx = ctx();

    let array = Array::<u32>::zeros_on(&ctx, 600).unwrap();
    let storage = array.buffer().unwrap();
    drop(array);

    let reused = Array::<u32>::zeros_on(&ctx, 1000).unwrap();
    assert_eq!(reused.buffer().unwrap(), storage);
}

#[test]
fn dropped_buffer_is_reused_while_its_readers_are_still_pending() {
    let ctx = ctx();

    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let storage = a.buffer().unwrap();
    let doubled = &a * 2.0;
    // Operators are lazy: record the multiplication (without waiting for it) so that it is
    // pending in the batch, reading a's buffer.
    assert!(doubled.ready().is_ok());
    assert!(ctx.pending_dispatches() > 0, "the multiplication must still be pending");
    drop(a);

    // The new array takes over a's buffer, although the multiplication has not run yet. (A
    // zero-copy buffer is only reusable after the next submission.)
    let b = Array::from_slice_on(&ctx, &[10.0f32, 20.0, 30.0], 3).unwrap();
    if !ctx.zero_copy() {
        assert_eq!(b.buffer().unwrap(), storage);
    }
    assert!(ctx.pending_dispatches() > 0);

    assert_eq!(doubled.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn flush_submits_the_pending_batch_and_keeps_the_results() {
    let ctx = ctx();
    ctx.flush().unwrap(); // nothing pending
    assert_eq!(ctx.pending_dispatches(), 0);

    let mut x = Array::<f32>::zeros_on(&ctx, 16).unwrap();
    // A view sharing the buffer makes every compound assignment a dispatch of its own.
    let view = x.reshape(16);
    x += 2.0;
    x *= 3.0;
    assert_eq!(ctx.pending_dispatches(), 2);
    ctx.flush().unwrap();
    assert_eq!(ctx.pending_dispatches(), 0);

    x += 1.0; // recorded after the flush, runs after it
    assert_eq!(ctx.pending_dispatches(), 1);
    assert_eq!(view.to_vec().unwrap(), vec![7.0; 16]);
}

/// Steps of a loop that keep the GPU busy long enough for `eval` to hold some back.
fn eval_busy_loop(ctx: &Context, x: &mut Array<f32>, steps: usize) {
    for _ in 0..steps {
        *x = &*x * 1.0 + 1.0;
        x.eval().unwrap();
    }
    assert!(ctx.pending_dispatches() < SUBMIT_HOLD + 1);
}

#[test]
fn flush_hands_over_what_eval_held_back() {
    let ctx = ctx();
    let mut x = Array::<f32>::zeros_on(&ctx, 1 << 22).unwrap();
    eval_busy_loop(&ctx, &mut x, 40);
    ctx.flush().unwrap();
    assert_eq!(ctx.pending_dispatches(), 0);
    assert_eq!(x.to_vec().unwrap()[..4], [40.0; 4]);
}

#[test]
fn a_waiting_call_completes_a_batch_that_eval_held_back() {
    let ctx = ctx();
    let mut x = Array::<f32>::zeros_on(&ctx, 1 << 22).unwrap();
    eval_busy_loop(&ctx, &mut x, 40);
    assert_eq!(x.to_vec().unwrap()[1 << 21], 40.0);
    assert_eq!(ctx.pending_dispatches(), 0);
}

#[test]
fn eval_hands_over_one_array_and_leaves_other_lazy_arrays_lazy() {
    let ctx = ctx();
    let x = Array::<f32>::full_on(&ctx, 16, 1.0).unwrap();
    let wanted = &x + 1.0;
    let unrelated = &x * 3.0;
    wanted.eval().unwrap();
    // Unlike `flush`, `eval` does not evaluate the other array.
    assert!(wanted.pending().is_none());
    assert!(unrelated.pending().is_some());
    ctx.flush().unwrap();
    assert_eq!(ctx.pending_dispatches(), 0);
    assert!(unrelated.pending().is_none());
    assert_eq!(wanted.to_vec().unwrap(), vec![2.0; 16]);
    assert_eq!(unrelated.to_vec().unwrap(), vec![3.0; 16]);
}

#[test]
fn one_pass_per_batch_and_bound_state_survive_interleaved_commands() {
    // Uploads and clears between launches end the pass; the launches after them start a new
    // one and must bind everything again.
    let ctx = ctx();
    let mut x = Array::<f32>::zeros_on(&ctx, 8).unwrap();
    for i in 0..50 {
        x += 1.0;
        let y = Array::from_slice_on(&ctx, &[i as f32; 8], 8).unwrap();
        x += &y;
        let z = Array::<f32>::zeros_on(&ctx, 8).unwrap();
        x += &z;
    }
    let want: f32 = (0..50).map(|i| 1.0 + i as f32).sum();
    assert_eq!(x.to_vec().unwrap(), vec![want; 8]);
}

#[test]
fn batch_is_submitted_at_the_dispatch_limit() {
    let ctx = ctx();
    let mut x = Array::<f32>::zeros_on(&ctx, 8).unwrap();
    // A second array shares the buffer, so every `+=` is a dispatch of its own.
    let _view = x.reshape(8);
    for i in 0..BATCH_LIMIT * 2 + 5 {
        x += 1.0;
        assert!(ctx.pending_dispatches() < BATCH_LIMIT, "after {i} operations");
    }
    assert_eq!(x.to_vec().unwrap(), vec![(BATCH_LIMIT * 2 + 5) as f32; 8]);
    assert_eq!(ctx.pending_dispatches(), 0);
}

#[test]
fn uniform_slots_beyond_the_prefix_copy_reach_the_gpu() {
    // The first slots of a batch are copied by its own command buffer, the rest by a separate
    // one. A launch takes one slot and the grid of the shared size one more; with 256-byte
    // slots `UNIFORM_PREFIX` holds 15 launches.
    let ctx = ctx();
    for launches in [1usize, 15, 16, 17, 40, 100] {
        let mut x = Array::<f32>::zeros_on(&ctx, 8).unwrap();
        let _view = x.reshape(8);
        for i in 1..=launches {
            x += i as f32;
        }
        let want: f32 = (1..=launches).map(|i| i as f32).sum();
        assert_eq!(x.to_vec().unwrap(), vec![want; 8], "{launches} launches");
    }
}

#[test]
fn postponing_stops_under_memory_pressure() {
    use super::batch::postpone;
    assert!(postpone(true, 0, false));
    assert!(!postpone(true, 0, true));
    assert!(!postpone(false, 0, false));
    assert!(!postpone(true, SUBMIT_HOLD, false));
}

#[test]
fn calibration_off_gives_the_profile_fallback() {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    assert_eq!(ctx.cost_model(), CostModel::fallback(ctx.tuning().profile));
}

#[test]
fn calibration_is_measured_once_per_device() {
    let first = Context::with_options(ContextOptions::from_env().calibrate(true)).unwrap();
    let cell = first.device_cost();
    let measured = *cell.get().expect("the context calibrated its device");
    assert_eq!(measured, first.cost_model());
    drop(first);

    let calibrations = crate::jit::cost::CALIBRATIONS.load(std::sync::atomic::Ordering::Relaxed);
    let second = Context::with_options(ContextOptions::from_env().calibrate(true)).unwrap();
    // A pooled device keeps its calibration: nothing was measured again.
    if Arc::ptr_eq(&cell, &second.device_cost()) {
        assert_eq!(second.cost_model(), measured);
        assert_eq!(crate::jit::cost::CALIBRATIONS.load(std::sync::atomic::Ordering::Relaxed), calibrations);
    }
}

#[test]
fn the_cost_model_can_be_overridden_in_tests() {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    let model = CostModel { launch_us: 123.0, ..CostModel::TEST };
    ctx.set_cost_model(model);
    assert_eq!(ctx.cost_model(), model);
}
