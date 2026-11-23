use std::time::Duration;

use super::batch::{BATCH_LIMIT, SUBMIT_HOLD};
use super::transfer::{SYNC_SPIN_MIN, WAIT_SAMPLES};
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
    assert_ne!(ctx.alloc_buffer(4000, BufferUsage::MAP_READ | BufferUsage::COPY_DST).unwrap(), a);
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
    let second = Context::with_options(ContextOptions::from_env().calibrate(true)).unwrap();
    // A pooled device keeps its calibration: the second context reads the same cell.
    if Arc::ptr_eq(&cell, &second.device_cost()) {
        assert_eq!(second.cost_model(), measured);
    }
}

#[test]
fn the_cost_model_can_be_overridden_in_tests() {
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    let model = CostModel::TEST.with(|m| { m.launch_us = 123.0; });
    ctx.set_cost_model(model);
    assert_eq!(ctx.cost_model(), model);
}

/// Every thread that shares a context has a recording of its own: one thread recording neither stops the
/// others from starting theirs nor captures their launches, and a thread's recording cannot be started twice.
#[test]
fn threads_record_side_by_side() {
    let ctx = ctx();
    let threads = 4;
    let barrier = std::sync::Barrier::new(threads);
    // Nothing is asserted before the last barrier: a thread that stopped would leave the others waiting.
    let outcomes: Vec<_> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..threads)
            .map(|t| {
                let (ctx, barrier) = (&ctx, &barrier);
                scope.spawn(move || {
                    let x = Array::from_slice_on(ctx, &vec![1.0f32; 16 + t], 16 + t).unwrap();
                    x.eval().unwrap();
                    let began = ctx.begin_recording(Vec::new());
                    let twice = ctx.begin_recording(Vec::new());
                    // Everyone is recording now.
                    barrier.wait();
                    for _ in 0..t + 1 {
                        let y = &x * 2.0;
                        crate::jit::force(&[&y]).unwrap_or_else(|_| panic!("thread {t} could not launch"));
                    }
                    barrier.wait();
                    let (here, recorded) = (ctx.recording_here(), ctx.recorded());
                    let commands = ctx.end_recording().map(|recording| recording.cmds.len());
                    let (still_here, ended_twice) = (ctx.recording_here(), ctx.end_recording().is_some());
                    // Once it ended, the thread can record again.
                    let again = ctx.begin_recording(Vec::new()) && ctx.end_recording().is_some();
                    (began, twice, here, recorded, commands, still_here, ended_twice, again)
                })
            })
            .collect();
        threads.into_iter().map(|thread| thread.join().unwrap()).collect()
    });
    for (t, (began, twice, here, recorded, commands, still_here, ended_twice, again)) in outcomes.into_iter().enumerate() {
        assert!(began, "thread {t} could not start a recording");
        assert!(!twice, "a thread has one recording per context");
        assert!(here);
        assert_eq!(recorded, t + 1, "thread {t} captured launches of others or lost its own");
        assert_eq!(commands, Some(t + 1));
        assert!(!still_here && !ended_twice);
        assert!(again);
    }
}

/// A recording that was aborted stays out of the way of a new one until its owner ends it, and only the
/// owner's launches ran in the meantime.
#[test]
fn aborting_a_recording_concerns_only_its_thread() {
    let ctx = ctx();
    let x = Array::from_slice_on(&ctx, &[1.0f32; 8], 8).unwrap();
    x.eval().unwrap();
    assert!(ctx.begin_recording(Vec::new()));
    let y = &x * 3.0;
    crate::jit::force(&[&y]).unwrap_or_else(|_| panic!("launch"));
    assert_eq!(ctx.recorded(), 1);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // Another thread's sync and readback do not abort this thread's recording.
            let z = Array::from_slice_on(&ctx, &[2.0f32; 8], 8).unwrap();
            assert_eq!((&z + 1.0).to_vec().unwrap(), vec![3.0; 8]);
            ctx.sync().unwrap();
        });
    });
    assert!(ctx.recording_here());
    assert_eq!(ctx.recorded(), 1);
    // Its own readback does: the captured launch runs, and the recording is aborted, not ended.
    assert_eq!(y.to_vec().unwrap(), vec![3.0; 8]);
    assert!(!ctx.recording_here());
    assert!(!ctx.begin_recording(Vec::new()), "an aborted recording stays until its owner ends it");
    assert!(ctx.end_recording().is_none());
    assert!(ctx.begin_recording(Vec::new()));
    assert!(ctx.end_recording().is_some());
}

/// A kernel of luma's own whose module declares a uniform its entry point never reads: the
/// pipeline layout has no binding for it, so the launch binds the arguments the shader uses and
/// skips the rest, whichever of them it is.
#[test]
fn a_launch_binds_exactly_the_bindings_the_shader_reads() {
    let ctx = ctx();
    let build = |source: &str| kernel::build_core(&ctx, &KernelKey::new(source.to_owned(), "main", None), Some("probe"), luma_core::KernelTrust::Trusted).unwrap();
    // The uniform at the end is unread.
    let tail = build(
        "struct Params { s0: u32, }
         @group(1) @binding(0) var<storage, read_write> out0: array<u32>;
         @group(1) @binding(1) var<storage, read> in0: array<u32>;
         @group(1) @binding(2) var<uniform> p: Params;
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
             if (gid.x >= luma.size.x) { return; }
             out0[gid.x] = in0[gid.x] + 1u;
         }",
    );
    // The input in the middle is unread.
    let middle = build(
        "struct Params { s0: u32, }
         @group(1) @binding(0) var<storage, read_write> out0: array<u32>;
         @group(1) @binding(1) var<storage, read> in0: array<u32>;
         @group(1) @binding(2) var<uniform> p: Params;
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
             if (gid.x >= luma.size.x) { return; }
             out0[gid.x] = p.s0;
         }",
    );
    let input = ctx.alloc_buffer(256, STORAGE_USAGE).unwrap();
    ctx.upload(&input, bytemuck::cast_slice(&[41u32; 64])).unwrap();
    let uniform = 7u32.to_le_bytes();
    let outputs: Vec<Buffer> = (0..4).map(|_| ctx.alloc_buffer(256, STORAGE_USAGE).unwrap()).collect();
    fn args<'a>(out: &'a Buffer, input: &'a Buffer, uniform: &'a [u8]) -> [RawArg<'a>; 3] {
        [RawArg::write(out, 256), RawArg::read(input, 256), RawArg::Uniform(uniform)]
    }

    // Good launches around bad ones, all in one batch: the bad ones are errors of their own, and
    // the work recorded before and after them still runs.
    ctx.launch(&tail, [1, 1, 1], [64, 1, 1], &args(&outputs[0], &input, &uniform)).unwrap();
    let short = [RawArg::write(&outputs[1], 256), RawArg::read(&input, 256)];
    assert!(matches!(ctx.launch(&tail, [1, 1, 1], [64, 1, 1], &short), Err(LumaError::KernelArgCount { expected: 3, actual: 2 })));
    let wrong = [RawArg::write(&outputs[1], 256), RawArg::Uniform(&uniform), RawArg::Uniform(&uniform)];
    assert!(matches!(ctx.launch(&tail, [1, 1, 1], [64, 1, 1], &wrong), Err(LumaError::KernelBinding(_))));
    ctx.launch(&middle, [1, 1, 1], [64, 1, 1], &args(&outputs[2], &input, &uniform)).unwrap();
    ctx.launch(&tail, [1, 1, 1], [64, 1, 1], &args(&outputs[3], &input, &uniform)).unwrap();

    assert_eq!(ctx.read::<u32>(&outputs[0], 256).unwrap(), vec![42; 64]);
    assert_eq!(ctx.read::<u32>(&outputs[2], 256).unwrap(), vec![7; 64]);
    assert_eq!(ctx.read::<u32>(&outputs[3], 256).unwrap(), vec![42; 64]);
}

/// A kernel binds byte ranges of one buffer: it reads one range and writes another of the same
/// buffer, the bind-group cache tells two offsets of a buffer apart, and a misaligned offset is
/// a binding error.
#[test]
fn a_launch_binds_byte_ranges_of_a_buffer() {
    let ctx = ctx();
    let core = kernel::build_core(
        &ctx,
        &KernelKey::new(
            "@group(1) @binding(0) var<storage, read_write> out0: array<u32>;
             @group(1) @binding(1) var<storage, read> in0: array<u32>;
             @compute @workgroup_size(64)
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                 if (gid.x >= luma.size.x) { return; }
                 out0[gid.x] = in0[gid.x] + 1u;
             }"
            .to_owned(),
            "main",
            None,
        ),
        Some("ranges"),
        luma_core::KernelTrust::Trusted,
    )
    .unwrap();
    let align = ctx.info().limits.storage_offset_align.max(256);
    let words = (align / 4) as usize;
    // [A | B | C | D]: A and C hold data, B and D are written.
    let buffer = ctx.alloc_buffer(4 * align, STORAGE_USAGE).unwrap();
    let a: Vec<u32> = (0..words as u32).collect();
    let c: Vec<u32> = (0..words as u32).map(|i| i * 10).collect();
    let mut data = vec![0u32; 4 * words];
    data[..words].copy_from_slice(&a);
    data[2 * words..3 * words].copy_from_slice(&c);
    ctx.upload(&buffer, bytemuck::cast_slice(&data)).unwrap();
    let range = |at: u64| RawArg::storage_at(&buffer, at * align, align, false);
    let target = |at: u64| RawArg::storage_at(&buffer, at * align, align, true);
    let size = [words as u32, 1, 1];
    let groups = [(words as u32).div_ceil(64), 1, 1];
    ctx.launch(&core, groups, size, &[target(1), range(0)]).unwrap();
    ctx.launch(&core, groups, size, &[target(3), range(2)]).unwrap();
    // The first launch again, so the cached bind group of (buffer, 0..) is hit after (buffer, 2..).
    ctx.launch(&core, groups, size, &[target(1), range(0)]).unwrap();
    let got = ctx.read::<u32>(&buffer, 4 * align).unwrap();
    assert_eq!(&got[..words], &a[..], "A is untouched");
    assert_eq!(got[words..2 * words], a.iter().map(|x| x + 1).collect::<Vec<_>>()[..]);
    assert_eq!(&got[2 * words..3 * words], &c[..], "C is untouched");
    assert_eq!(got[3 * words..], c.iter().map(|x| x + 1).collect::<Vec<_>>()[..]);

    let misaligned = [RawArg::storage_at(&buffer, 2 * align + 4, 4, true), range(0)];
    let result = ctx.launch(&core, [1, 1, 1], [1, 1, 1], &misaligned);
    assert!(matches!(result, Err(LumaError::KernelBinding(_))), "{result:?}");
    let outside = [RawArg::storage_at(&buffer, 3 * align, 2 * align, true), range(0)];
    assert!(matches!(ctx.launch(&core, [1, 1, 1], [1, 1, 1], &outside), Err(LumaError::KernelBinding(_))));
    // Overlapping ranges of one buffer still alias.
    let alias = [RawArg::storage_at(&buffer, 0, 2 * align, true), range(1)];
    assert!(matches!(ctx.launch(&core, [1, 1, 1], [1, 1, 1], &alias), Err(LumaError::KernelArgAlias { .. })));
}

/// The device reports its free memory on CUDA and Metal, and it drops after an allocation.
#[test]
fn memory_now_follows_allocations() {
    let ctx = ctx();
    let Some(before) = ctx.hal().memory_now() else {
        assert!(matches!(ctx.backend(), BackendKind::Wgpu | BackendKind::Cpu));
        assert_eq!(ctx.memory_stats().device_free_bytes, None);
        return;
    };
    assert!(before.free <= before.total);
    let held = ctx.alloc_buffer(256 << 20, STORAGE_USAGE).unwrap();
    ctx.upload(&held, &vec![1u8; 256 << 20]).unwrap();
    ctx.sync().unwrap();
    let after = ctx.memory_stats().device_free_bytes.unwrap();
    assert!(after <= before.free - (128 << 20), "free memory {after} after, {} before", before.free);
    drop(held);
}

/// Slots for the live intervals of transient buffers: intervals that overlap, even at one
/// command, never share a slot, and as few slots are used as the greatest number of intervals
/// live at once.
#[test]
fn interval_packing_never_overlaps_and_is_minimal() {
    use super::record::pack_intervals;

    // A chain: each interval touches the next at one command.
    let chain: Vec<(usize, usize)> = (0..8).map(|i| (i, i + 1)).collect();
    let slots = pack_intervals(&chain);
    assert_eq!(slots.iter().max(), Some(&1), "{slots:?}");

    // The first one is live across all the others.
    let mut late = chain.clone();
    late.push((0, 9));
    let slots = pack_intervals(&late);
    let long = slots[8];
    assert!(slots[..8].iter().all(|&slot| slot != long), "{slots:?}");
    assert_eq!(slots.iter().max(), Some(&2), "{slots:?}");

    // Strictly before: ending at a command and starting at it overlap; one command apart do not.
    assert_eq!(pack_intervals(&[(0, 3), (3, 5)]), [0, 1]);
    assert_eq!(pack_intervals(&[(0, 3), (4, 5)]), [0, 0]);
    assert_eq!(pack_intervals(&[(5, 5), (0, 1)]), [0, 0]);
    assert!(pack_intervals(&[]).is_empty());

    // Against brute force on pseudo-random intervals.
    let mut state = 12345u64;
    let mut next = |bound: u64| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) % bound) as usize
    };
    for _ in 0..200 {
        let intervals: Vec<(usize, usize)> = (0..1 + next(30))
            .map(|_| {
                let start = next(40);
                (start, start + next(10))
            })
            .collect();
        let slots = pack_intervals(&intervals);
        for (i, a) in intervals.iter().enumerate() {
            for (j, b) in intervals.iter().enumerate().skip(i + 1) {
                if slots[i] == slots[j] {
                    assert!(a.1 < b.0 || b.1 < a.0, "{a:?} and {b:?} share slot {}", slots[i]);
                }
            }
        }
        let live = (0..60).map(|at| intervals.iter().filter(|&&(s, e)| s <= at && at <= e).count()).max().unwrap();
        assert_eq!(slots.iter().max().map_or(0, |last| last + 1), live);
    }
}

/// The scratch of an operation counts as resident while the operation holds it, and the counter
/// of the single-pass reductions (the context keeps it for life) counts once it exists.
#[test]
fn scratch_and_the_reduce_counter_are_counted_while_held() {
    let ctx = ctx();
    let resident = || ctx.pager().resident_bytes();
    let before = resident();
    let mut scratch = Scratch::new(&ctx);
    let buffer = scratch.take(1000).unwrap();
    assert_eq!(resident(), before + buffer.size());
    let other = scratch.take(5000).unwrap();
    assert_eq!(resident(), before + buffer.size() + other.size());
    drop((buffer, other));
    drop(scratch);
    assert_eq!(resident(), before, "given back to the pool, which counts them as pooled");
    assert!(ctx.pooled_bytes() > 0);

    // Only backends with single-pass reductions make the counter (the wgpu one cannot fill a
    // buffer at creation on a discrete device).
    if !ctx.tuning().single_pass_reduce {
        return;
    }
    let counter = ctx.reduce_counter().unwrap().size();
    assert_eq!(resident(), before + counter, "the counter stays for the life of the context");
    ctx.reduce_counter().unwrap();
    assert_eq!(resident(), before + counter, "made once");
}

/// A context is freed with its last handle, whatever ran on it: nothing process-wide may keep one
/// alive. The kernels of the dense linear algebra were once cached in a process-wide table whose
/// kernels hold their context, so a finished context stayed alive, and with it its GPU device,
/// which is then never handed to the next context; the test suite of a machine with many threads
/// ran out of the 63 or 64 devices the NVIDIA driver gives a process.
#[test]
fn a_context_that_ran_linear_algebra_is_freed_with_its_last_handle() {
    let ctx = ctx();
    let weak = Arc::downgrade(&ctx.inner);
    let a = Array::from_slice_on(&ctx, &[4.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0], [3, 3]).unwrap();
    let l = a.cholesky().to_vec().unwrap();
    assert!((l[0] - 2.0).abs() < 1e-6, "the factor was computed: {l:?}");
    let lu = a.lu();
    lu.lu.to_vec().unwrap();
    drop((a, lu));
    drop(ctx);
    assert!(weak.upgrade().is_none(), "something outlives its context");
}
