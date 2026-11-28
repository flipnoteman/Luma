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
