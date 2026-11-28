//! Schedules (`LUMA_SPLIT`): kernels over arrays of several segments are deferred and run segment by segment. These tests open
//! their contexts in `split` mode `Auto` (and `Off` where they compare); the pinned tests of `chunked.rs` and `spill.rs` run with
//! both through `LUMA_SPLIT`.

use luma::{select, Array, Context, ContextOptions, SplitMode};

/// A context of 4096-byte segments (1024 `f32`), deferring.
fn deferring() -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(4096).split(SplitMode::Auto)).expect("a device is available")
}

fn kernel_major() -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(4096).split(SplitMode::Off)).expect("a device is available")
}

const N: usize = 5 * 1024 + 17;

fn data() -> Vec<f32> {
    (0..N).map(|i| ((i * 37) % 101) as f32 * 0.125 - 5.0).collect()
}

#[test]
fn a_chain_over_several_segments_runs_as_a_schedule() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = ((&x * 2.0 + 1.0).sqrt() - &x).abs();
    assert_eq!(ctx.stream_stats().schedules, 0, "nothing ran yet");
    let got = y.to_vec().unwrap();
    let stats = ctx.stream_stats();
    assert!(stats.schedules >= 1, "{stats:?}");
    assert!(stats.steps >= 1 && stats.slabs >= 6, "{stats:?}");
    let off = kernel_major();
    let x = Array::from_slice_on(&off, &data(), N).unwrap();
    let want = ((&x * 2.0 + 1.0).sqrt() - &x).abs().to_vec().unwrap();
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn single_segment_arrays_never_defer() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data()[..500], 500).unwrap();
    let _ = ((&x + 1.0) * 3.0).to_vec().unwrap();
    assert_eq!(ctx.stream_stats().schedules, 0);
}

#[test]
fn maps_are_bit_equal_to_the_kernel_major_path() {
    let on = deferring();
    let off = kernel_major();
    let xs = data();
    let us: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let is: Vec<i32> = us.iter().map(|&u| u as i32 >> 7).collect();
    let run = |ctx: &Context| -> (Vec<u32>, Vec<u32>, Vec<i32>, Vec<u32>) {
        let x = Array::from_slice_on(ctx, &xs, N).unwrap();
        let u = Array::from_slice_on(ctx, &us, N).unwrap();
        let i = Array::from_slice_on(ctx, &is, N).unwrap();
        let a = select(&x.gt(0.5), &x * 3.0 - 1.0, x.abs() + 0.25).to_vec().unwrap();
        let b = (&u ^ (&u >> 3u32)) + 7u32;
        let c = &i * 3 - (&i >> 2);
        let d = (&x * 2.0 + 1.0).sqrt() * 0.5 + &x;
        (a.iter().map(|v| v.to_bits()).collect(), b.to_vec().unwrap(), c.to_vec().unwrap(), d.to_vec().unwrap().iter().map(|v| v.to_bits()).collect())
    };
    assert_eq!(run(&on), run(&off));
    assert!(on.stream_stats().schedules > 0);
    assert_eq!(off.stream_stats().schedules, 0);
}

#[test]
fn a_write_after_a_deferred_read_does_not_change_it() {
    let ctx = deferring();
    let mut x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = &x * 2.0;
    let z = y.exp();
    // `z` is lazy and reads `x`; writing `x` evaluates it first.
    let fresh: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    x.write(&fresh).unwrap();
    let want: Vec<f32> = data().iter().map(|v| (v * 2.0).exp()).collect();
    let got = z.to_vec().unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "{g} {w}");
    }
    assert_eq!(x.to_vec().unwrap(), fresh);
}

#[test]
fn a_write_after_an_evaluated_deferred_step_does_not_change_it() {
    let ctx = deferring();
    let mut x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = &x * 2.0;
    y.eval().unwrap();
    // The step reading `x` is in the schedule; the write must commit it first.
    let fresh = vec![1.0f32; N];
    x.write(&fresh).unwrap();
    let want: Vec<f32> = data().iter().map(|v| v * 2.0).collect();
    assert_eq!(y.to_vec().unwrap(), want);
    assert_eq!(x.to_vec().unwrap(), fresh);
}

#[test]
fn compound_assignment_commits_readers_first() {
    let ctx = deferring();
    let mut x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = &x + 1.0;
    y.eval().unwrap();
    x += 10.0;
    let want_y: Vec<f32> = data().iter().map(|v| v + 1.0).collect();
    let want_x: Vec<f32> = data().iter().map(|v| v + 10.0).collect();
    assert_eq!(y.to_vec().unwrap(), want_y);
    assert_eq!(x.to_vec().unwrap(), want_x);
}

#[test]
fn flush_sync_and_eval_commit() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = (&x * 2.0).abs();
    y.eval().unwrap();
    ctx.flush().unwrap();
    let after_flush = ctx.stream_stats().schedules;
    assert!(after_flush >= 1);
    let z = (&x + 3.0).abs();
    z.eval().unwrap();
    ctx.sync().unwrap();
    assert!(ctx.stream_stats().schedules > after_flush);
    drop(y);
    drop(z);
}

#[test]
fn a_size_bound_commits() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let mut ys = Vec::new();
    for k in 0..600 {
        let y = &x + k as f32;
        y.eval().unwrap();
        ys.push(y);
    }
    assert!(ctx.stream_stats().schedules >= 1, "a long schedule commits before anyone asks");
    assert_eq!(ys[599].to_vec().unwrap()[3], data()[3] + 599.0);
}

#[test]
fn committing_from_another_thread() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = (&x * 4.0).abs();
    y.eval().unwrap();
    let want: Vec<f32> = data().iter().map(|v| (v * 4.0).abs()).collect();
    let got = std::thread::scope(|scope| scope.spawn(|| y.to_vec().unwrap()).join().unwrap());
    assert_eq!(got, want);
}

#[test]
fn a_thread_that_ends_commits_its_steps() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let y = (&x * 5.0).abs();
                y.eval().unwrap();
                y
            })
            .join()
            .unwrap()
    });
    let want: Vec<f32> = data().iter().map(|v| (v * 5.0).abs()).collect();
    assert_eq!(y.to_vec().unwrap(), want);
}

#[test]
fn reductions_and_global_ops_commit_what_they_read() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = (&x * 2.0).abs();
    y.eval().unwrap();
    let sum = y.sum().item().unwrap();
    let want: f32 = data().iter().map(|v| (v * 2.0).abs()).sum();
    assert!((sum - want).abs() <= 1e-3 * want.abs(), "{sum} {want}");
}

#[test]
fn an_off_context_never_defers() {
    let ctx = kernel_major();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let _ = (&x * 2.0).to_vec().unwrap();
    assert_eq!(ctx.stream_stats(), Default::default());
}

/// An expression too long for one kernel is cut into several; what the cuts store is read by the next kernel and by nothing else.
/// Every round reads the previous result twice, so a cut cannot overwrite its input.
fn long_chain(ctx: &Context, rounds: usize) -> Vec<f32> {
    let x = Array::from_slice_on(ctx, &data(), N).unwrap();
    let mut y = x.clone();
    for k in 0..rounds {
        y = (&y * 1.0009 + (k % 5) as f32 * 0.25).abs() - (&y * 0.001);
    }
    y.to_vec().unwrap()
}

#[test]
fn dead_temporaries_are_never_materialised() {
    let ctx = deferring();
    let got = long_chain(&ctx, 300);
    let off = kernel_major();
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), long_chain(&off, 300).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let stats = ctx.stream_stats();
    assert!(stats.steps >= 2, "the chain is cut into several kernels: {stats:?}");
    // The cuts are read by the next kernel only: one stage, one pass over the six segments, and no buffers for the cuts but the last.
    assert_eq!(stats.stages, 1, "{stats:?}");
    assert_eq!(stats.slabs, 6, "{stats:?}");
    assert!(stats.transients + 1 >= stats.steps, "{stats:?}");
    assert!(stats.transient_bytes_elided >= (stats.steps - 1) * 4 * N as u64, "{stats:?}");
}

#[test]
fn a_halo_edge_cuts_the_stage() {
    let on = deferring();
    on.set_assume_idle(true);
    let off = kernel_major();
    off.set_assume_idle(true);
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &data(), N).unwrap();
        let a = &x * 2.0;
        a.eval().unwrap();
        let b = a.shift(1) + &a;
        b.eval().unwrap();
        let c = &b - 1.0;
        c.to_vec().unwrap()
    };
    assert_eq!(run(&on), run(&off));
    let stats = on.stream_stats();
    assert!(stats.stages >= 2, "a read through a shift of an array written in the stage is a cut: {stats:?}");
}

#[test]
fn in_place_chains_match_the_kernel_major_path() {
    let on = deferring();
    let off = kernel_major();
    let run = |ctx: &Context| {
        let mut x = Array::from_slice_on(ctx, &data(), N).unwrap();
        x += 1.0;
        x *= 3.0;
        let y = &x + 2.0;
        x -= 0.5;
        (x.to_vec().unwrap(), y.to_vec().unwrap())
    };
    assert_eq!(run(&on), run(&off));
}

#[test]
fn decisions_record_one_schedule_per_commit() {
    let ctx = deferring();
    ctx.record_decisions(true);
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let y = (&x * 2.0).abs();
    let _ = y.to_vec().unwrap();
    let schedule: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == luma::DecisionSite::Schedule).collect();
    assert_eq!(schedule.len(), 1, "{schedule:?}");
    assert!(schedule[0].chosen.starts_with("stages=1 slabs=6 window=1"), "{}", schedule[0].chosen);
}

#[test]
fn a_chain_over_an_array_above_the_budget_uploads_each_segment_once() {
    let options = |mode| ContextOptions::from_env().chunk_bytes(1 << 16).memory_budget(1 << 20).split(mode);
    let n = 1 << 20; // 4 MiB of f32 in 64 segments, four times the budget
    let xs: Vec<f32> = (0..n).map(|i| (i % 1013) as f32 * 0.01).collect();
    let run = |mode| {
        let ctx = Context::with_options(options(mode)).expect("a device is available");
        let x = Array::from_slice_on(&ctx, &xs, n).unwrap();
        let before = ctx.paging_stats().uploaded_bytes;
        let mut y = x.clone();
        for k in 0..400 {
            y = (&y * 1.0009 + (k % 5) as f32 * 0.25).abs();
        }
        let got = y.to_vec().unwrap();
        (got, ctx.paging_stats().uploaded_bytes - before, ctx.stream_stats())
    };
    let (on, on_up, stats) = run(SplitMode::Auto);
    let (off, off_up, _) = run(SplitMode::Off);
    assert_eq!(on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), off.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    if ctx_has_budget_paging(on_up, off_up) {
        assert!(on_up <= 4 * n as u64 + (1 << 16), "each input segment crosses once per chain: {on_up} {stats:?}");
        assert!(stats.steps >= 2 && stats.stages == 1, "{stats:?}");
        assert!(off_up >= 2 * on_up, "kernel-major pages the input in once per kernel: off {off_up}, on {on_up}");
    }
}

/// Devices that keep nothing under a budget (software ones) never page: nothing to compare there.
fn ctx_has_budget_paging(on: u64, off: u64) -> bool {
    on > 0 || off > 0
}

#[test]
fn recording_disables_deferral_and_starts_from_committed_state() {
    let on = deferring();
    let off = kernel_major();
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &data(), N).unwrap();
        // A pending streamed step, then a traced loop on a small array and on the streamed one.
        let pending = &x * 2.0;
        pending.eval().unwrap();
        let small = Array::from_slice_on(ctx, &data()[..300], 300).unwrap();
        let looped_small = small.iterate(6, |a| a * 1.01 + 0.5).to_vec().unwrap();
        let looped_big = x.iterate(4, |a| a * 0.99 + 0.25).to_vec().unwrap();
        (looped_small, looped_big, pending.to_vec().unwrap())
    };
    assert_eq!(run(&on), run(&off));
}

#[test]
fn plan_cache_does_not_cache_deferred_evaluations() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    for k in 0..6 {
        let y = ((&x * 2.0 + 1.0).sqrt() + k as f32).abs();
        assert_eq!(y.to_vec().unwrap().len(), N);
    }
    assert_eq!(ctx.plan_cache_hits(), 0, "a deferred evaluation is never a cached plan");
}

fn above_budget(mode: SplitMode, budget: u64) -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(1 << 16).memory_budget(budget).split(mode)).expect("a device is available")
}

/// `y = |x * 1.5 + 1| - x` over `n` elements, a new array: the input streams in and the output streams out.
fn map_chain(ctx: &Context, n: usize) -> (Vec<f32>, luma::StreamStats) {
    let xs: Vec<f32> = (0..n).map(|i| (i % 1013) as f32 * 0.01 - 3.0).collect();
    let x = Array::from_slice_on(ctx, &xs, n).unwrap();
    let y = (&x * 1.5 + 1.0).abs() - &x;
    let got = y.to_vec().unwrap();
    (got, ctx.stream_stats())
}

#[test]
fn a_streamed_stage_matches_kernel_major_and_moves_each_segment_once() {
    let n = 1 << 20; // 4 MiB of f32 in 64 segments, four times the budget
    let (on, stats) = map_chain(&above_budget(SplitMode::Auto, 1 << 20), n);
    let (off, _) = map_chain(&above_budget(SplitMode::Off, 1 << 20), n);
    assert_eq!(on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), off.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(stats.stages, 1, "{stats:?}");
    eprintln!("{stats:?}");
    if stats.streamed > 0 {
        assert_eq!(stats.streamed, 2, "the input streams in and the output out: {stats:?}");
        assert!(stats.ring_bytes_peak > 0 && stats.ring_bytes_peak <= 1 << 20, "{stats:?}");
        assert!(stats.upload_bytes <= 4 * n as u64 + (1 << 16), "each input segment crosses once: {stats:?}");
        assert!(stats.download_bytes >= 4 * n as u64 && stats.download_bytes <= 5 * n as u64, "the output goes to host blocks once: {stats:?}");
    }
}

#[test]
fn an_infeasible_stage_falls_back_in_order() {
    let n = 1 << 20;
    let seg = 1u64 << 16;
    // Two slabs in flight of x and y need 4 segments; with room for 3 the window drops to one slab.
    let (on, stats) = map_chain(&above_budget(SplitMode::Auto, 3 * seg), n);
    let (off, _) = map_chain(&above_budget(SplitMode::Off, 3 * seg), n);
    assert_eq!(on, off);
    if stats.streamed > 0 {
        assert!(stats.window1_fallbacks >= 1 && stats.cuts == 0, "window 1 comes first: {stats:?}");
    }
    // One slab of each of x and y does not fit in one segment: the single step fails and names itself.
    let ctx = above_budget(SplitMode::Auto, seg);
    let xs = vec![1.0f32; n];
    let x = Array::from_slice_on(&ctx, &xs, n).unwrap();
    let error = (&x * 2.0).abs().to_vec().err();
    assert!(error.is_none_or(|e| e.to_string().contains("budget") || e.to_string().contains("memory")));
    // The chain's stage needs a slab of the input, of the result and of each buffer the cuts store (five segments); one of its kernels
    // needs four. With room for four the stage is cut in two, each half needing less.
    let chain = |mode| {
        let ctx = above_budget(mode, 4 * seg);
        let x = Array::from_slice_on(&ctx, &xs, n).unwrap();
        let mut y = x.clone();
        for k in 0..300 {
            y = (&y * 1.0009 + (k % 5) as f32 * 0.25).abs() - (&y * 0.001);
        }
        (y.to_vec().unwrap(), ctx.stream_stats())
    };
    let (got, stats) = chain(SplitMode::Auto);
    assert!(stats.cuts >= 1 || stats.streamed == 0, "a stage that does not fit is cut: {stats:?}");
    assert_eq!(got, chain(SplitMode::Off).0);
}

/// A store into a multi-segment file array that is dropped before the schedule committed still reaches the file: a segment with a home
/// is never a transient.
#[test]
fn a_store_into_a_dropped_file_array_reaches_the_file() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let path = std::env::temp_dir().join(format!("luma-split-test-{}-store-drop", std::process::id()));
    let mut target = Array::<f32>::create_on(&ctx, &path, N, luma::FileOptions::raw()).unwrap();
    (&x * 2.0 + 1.0).store(&mut target).unwrap();
    drop(target);
    let want: Vec<f32> = data().iter().map(|v| v * 2.0 + 1.0).collect();
    let bytes = std::fs::read(&path).unwrap();
    let got: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.luma", path.display()));
    assert_eq!(got, want);
}

// -- review fixes (switch-on bugs) --

const ROWS: usize = 8;
const COLS: usize = 700;

fn big_data() -> Vec<f32> {
    (0..ROWS * COLS).map(|i| ((i * 29) % 97) as f32 * 0.25 - 3.0).collect()
}

fn row_data(k: usize) -> Vec<f32> {
    (0..COLS).map(|i| ((i * 7 + k) % 13) as f32 * 0.5).collect()
}

fn broadcast_sum(row: &[f32]) -> Vec<f32> {
    big_data().iter().enumerate().map(|(i, v)| v + row[i % COLS]).collect()
}

/// `big` has several segments, `s` one: a deferred `big + s` reads `s`, and `s * 2` takes `s`'s buffer over at once (it is not
/// deferred: nothing it touches is streamed). The takeover must not run before the step that reads `s` did.
fn takeover_while_a_step_reads(ctx: &Context, how: u8) {
    let big = Array::from_slice_on(ctx, &big_data(), [COLS, ROWS]).unwrap();
    let s = Array::from_slice_on(ctx, &row_data(1), [COLS, 1]).unwrap();
    let y = &big + &s;
    let t = s * 2.0;
    let mut t_values = None;
    match how {
        // `y`'s step waits in the schedule (`eval_all` leaves it open); asking for `t` evaluates it at once, over `s`'s buffer.
        0 => {
            luma::eval_all(&[&y]).unwrap();
            t_values = Some(t.to_vec().unwrap());
        }
        1 => luma::eval_all(&[&y, &t]).unwrap(),
        2 => {
            y.eval().unwrap();
            t.eval().unwrap();
            ctx.flush().unwrap();
        }
        _ => {
            y.eval().unwrap();
            ctx.flush().unwrap();
            t.eval().unwrap();
        }
    }
    assert_eq!(y.to_vec().unwrap(), broadcast_sum(&row_data(1)));
    assert_eq!(t_values.unwrap_or_else(|| t.to_vec().unwrap()), row_data(1).iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

#[test]
fn an_in_place_takeover_waits_for_a_deferred_step_that_reads_the_buffer() {
    for how in 0..4 {
        // A busy device postpones `eval`: the step waits in the schedule (on an idle one it would commit at once).
        let busy = deferring();
        busy.set_assume_busy(true);
        takeover_while_a_step_reads(&busy, how);
        let idle = deferring();
        idle.set_assume_idle(true);
        takeover_while_a_step_reads(&idle, how);
        takeover_while_a_step_reads(&kernel_major(), how);
    }
}

#[test]
fn a_takeover_inside_the_deferring_schedule_keeps_the_step_order() {
    // `x` is read by the first step and taken over by the second, both deferred into one schedule: no commit is needed in between.
    let ctx = deferring();
    let off = kernel_major();
    let run = |ctx: &Context| {
        let big = Array::from_slice_on(ctx, &data(), N).unwrap();
        let x = Array::from_slice_on(ctx, &data(), N).unwrap();
        let y = &big + &x;
        let z = x * 3.0;
        luma::eval_all(&[&y, &z]).unwrap();
        (y.to_vec().unwrap(), z.to_vec().unwrap())
    };
    assert_eq!(run(&ctx), run(&off));
}

#[test]
fn a_takeover_on_one_thread_commits_the_schedule_of_another() {
    let ctx = deferring();
    ctx.set_assume_busy(true);
    let big = Array::from_slice_on(&ctx, &big_data(), [COLS, ROWS]).unwrap();
    let s = Array::from_slice_on(&ctx, &row_data(2), [COLS, 1]).unwrap();
    let y = &big + &s;
    y.eval().unwrap();
    // `y`'s step waits in this thread's schedule; the other thread takes `s` over.
    let t = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let t = s * 2.0;
                t.eval().unwrap();
                t.to_vec().unwrap()
            })
            .join()
            .unwrap()
    });
    assert_eq!(t, row_data(2).iter().map(|v| v * 2.0).collect::<Vec<_>>());
    assert_eq!(y.to_vec().unwrap(), broadcast_sum(&row_data(2)));
}

/// Two threads defer, take over, and wait for each other's schedules: no deadlock, and every result is right.
#[test]
fn two_threads_settling_each_others_schedules_do_not_deadlock() {
    use std::sync::mpsc;
    use std::time::Duration;
    let ctx = deferring();
    ctx.set_assume_busy(true);
    let big = Array::from_slice_on(&ctx, &big_data(), [COLS, ROWS]).unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let (to_b, from_a) = mpsc::channel::<(Array<f32>, usize)>();
        let (to_a, from_b) = mpsc::channel::<(Array<f32>, usize)>();
        let (ctx, big) = (&ctx, &big);
        let a_done = done_tx.clone();
        scope.spawn(move || {
            for k in 0..40 {
                let s = Array::from_slice_on(ctx, &row_data(k), [COLS, 1]).unwrap();
                let y = big + &s;
                y.eval().unwrap();
                let t = s * 2.0;
                t.eval().unwrap();
                assert_eq!(t.to_vec().unwrap(), row_data(k).iter().map(|v| v * 2.0).collect::<Vec<_>>());
                to_b.send((y, k)).unwrap();
                if let Ok((w, j)) = from_b.try_recv() {
                    assert_eq!(w.to_vec().unwrap(), broadcast_sum(&row_data(j)).iter().map(|v| v + 1.0).collect::<Vec<_>>());
                }
            }
            drop(to_b);
            while let Ok((w, j)) = from_b.recv_timeout(Duration::from_secs(1)) {
                assert_eq!(w.to_vec().unwrap(), broadcast_sum(&row_data(j)).iter().map(|v| v + 1.0).collect::<Vec<_>>());
            }
            a_done.send(()).unwrap();
        });
        scope.spawn(move || {
            while let Ok((y, k)) = from_a.recv_timeout(Duration::from_secs(20)) {
                let s = Array::from_slice_on(ctx, &row_data(k), [COLS, 1]).unwrap();
                let w = &y + 1.0;
                let u = &s + 1.0;
                w.eval().unwrap();
                u.eval().unwrap();
                let _ = to_a.send((w, k));
                assert_eq!(u.to_vec().unwrap(), row_data(k).iter().map(|v| v + 1.0).collect::<Vec<_>>());
            }
            done_tx.send(()).unwrap();
        });
        for _ in 0..2 {
            done_rx.recv_timeout(Duration::from_secs(120)).expect("the threads finished (no deadlock)");
        }
    });
}

/// A homed array's transients are never taken from a file segment, and a chain over a tiny file array does not size them at a full
/// segment (the budget below holds two segments and the chain needs more than one colour).
#[test]
fn a_chain_over_a_tiny_file_array_runs_in_a_small_budget() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(4096).memory_budget(12 * 1024).split(SplitMode::Auto)).expect("a device is available");
    let off = Context::with_options(ContextOptions::from_env().chunk_bytes(4096).memory_budget(12 * 1024).split(SplitMode::Off)).expect("a device is available");
    let path = std::env::temp_dir().join(format!("luma-split-test-{}-tiny", std::process::id()));
    let xs: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let run = |ctx: &Context, suffix: &str| {
        let path = path.with_extension(suffix);
        let mut file = Array::<f32>::create_on(ctx, &path, 100, luma::FileOptions::raw()).unwrap();
        file.write(&xs).unwrap();
        let mut y = file.clone();
        for k in 0..300 {
            y = (&y * 1.0009 + (k % 5) as f32 * 0.25).abs() - (&y * 0.001);
        }
        let got = y.to_vec().unwrap();
        drop((y, file));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.luma", path.display()));
        got
    };
    assert_eq!(run(&ctx, "on"), run(&off, "off"));
}

#[test]
fn reopening_a_file_array_with_a_store_pending_sees_the_store() {
    let ctx = deferring();
    let x = Array::from_slice_on(&ctx, &data(), N).unwrap();
    let path = std::env::temp_dir().join(format!("luma-split-test-{}-store-reopen", std::process::id()));
    let mut target = Array::<f32>::create_on(&ctx, &path, N, luma::FileOptions::raw()).unwrap();
    (&x * 2.0 + 1.0).store(&mut target).unwrap();
    let want: Vec<f32> = data().iter().map(|v| v * 2.0 + 1.0).collect();
    // While the array lives, a second open shares it.
    let again = Array::<f32>::open_on(&ctx, &path, luma::FileOptions::raw().recover().writable()).unwrap();
    assert_eq!(again.to_vec().unwrap(), want);
    drop(again);
    // After it is dropped, a new open reads the file the dropped array wrote.
    (&x * 4.0).store(&mut target).unwrap();
    drop(target);
    let reopened = Array::<f32>::open_on(&ctx, &path, luma::FileOptions::raw().recover()).unwrap();
    assert_eq!(reopened.to_vec().unwrap(), data().iter().map(|v| v * 4.0).collect::<Vec<_>>());
    drop(reopened);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.luma", path.display()));
}

/// The same through a store into a multi-segment array, which defers without committing by itself.
#[test]
fn an_in_place_takeover_waits_for_a_deferred_store_that_reads_the_buffer() {
    for mode in [SplitMode::Auto, SplitMode::Segment] {
        let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(4096).split(mode)).expect("a device is available");
        let big = Array::from_slice_on(&ctx, &big_data(), [COLS, ROWS]).unwrap();
        let s = Array::from_slice_on(&ctx, &row_data(1), [COLS, 1]).unwrap();
        let mut target = Array::<f32>::zeros_on(&ctx, [COLS, ROWS]).unwrap();
        (&big + &s).store(&mut target).unwrap();
        let t = s * 2.0;
        assert_eq!(t.to_vec().unwrap(), row_data(1).iter().map(|v| v * 2.0).collect::<Vec<_>>());
        assert_eq!(target.to_vec().unwrap(), broadcast_sum(&row_data(1)));
    }
}
