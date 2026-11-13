//! Several threads sharing one context (the default one, as the tests of a crate do) get the results
//! they would get alone: tracing and replaying loops, recording launches, following `eval` loops
//! and readbacks, and sharing the pending work of the graph must not mix up what threads compute.

// Proving `Context: Sync` for the closures below goes deeper than the default through wgpu-core.
#![recursion_limit = "256"]

mod common;

use std::sync::Barrier;

use common::{assert_close, values};
use luma::*;

const N: usize = 256;

fn on(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

/// `u` of one step of the diffusion stencil, circular.
fn diffuse(u: &[f32]) -> Vec<f32> {
    let n = u.len();
    (0..n).map(|i| 0.5 * u[i] + 0.25 * (u[(i + n - 1) % n] + u[(i + 1) % n])).collect()
}

/// The solver-like step: a padded copy with a ghost written by `assign`, a staged `cache()`, ...
fn padded(u: &[f32]) -> Vec<f32> {
    let n = u.len();
    let w: Vec<f32> = std::iter::once(0.5).chain(u.iter().copied()).chain(std::iter::once(u[n - 1])).collect();
    (0..n).map(|i| (0.5 * w[i + 1] + 0.25 * w[i] + 0.25 * w[i + 2]) * 0.99 + 0.001).collect()
}

fn padded_step(u: &Array<f32>) -> Array<f32> {
    let n = u.len();
    let mut w = u.pad([(1, 1)], PadMode::Edge);
    w.slice_mut(0, ..1).assign(0.5).unwrap();
    let c = (w.slice(0, 1..n + 1) * 0.5 + w.slice(0, ..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25).cache();
    c * 0.99 + 0.001
}

/// Runs `work(thread index, round)` on `threads` threads, every round of every thread starting
/// together with the others.
fn together(threads: usize, rounds: usize, work: impl Fn(usize, usize) + Sync) {
    let barrier = Barrier::new(threads);
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (barrier, work) = (&barrier, &work);
            scope.spawn(move || {
                let mut failures = Vec::new();
                for round in 0..rounds {
                    barrier.wait();
                    // A thread that stopped would leave the others at the barrier.
                    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(t, round))) {
                        let message = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|m| m.to_string()));
                        failures.push(format!("thread {t} round {round}: {}", message.unwrap_or_default()));
                    }
                }
                assert!(failures.is_empty(), "{} of {rounds} rounds failed:\n{}", failures.len(), failures.join("\n"));
            });
        }
    });
}

fn data(t: usize, round: usize) -> Vec<f32> {
    values(N, 1 + (t * 1000 + round) as u64, 0.0, 1.0)
}

#[test]
fn iterate_all_of_stencil_steps_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let (u0, v0) = (data(t, round), data(t + 100, round));
        let (u, v) = (on(&ctx, &u0), on(&ctx, &v0));
        let [u, v] = iterate_all([&u, &v], 11, |[u, v]| [(u.shift(1) + u.shift(-1)) * 0.25 + u * 0.5, v * 0.9 + u * 0.1]);
        let (mut wu, mut wv) = (u0, v0);
        for _ in 0..11 {
            let nu = diffuse(&wu);
            wv = wv.iter().zip(&wu).map(|(v, u)| v * 0.9 + u * 0.1).collect();
            wu = nu;
        }
        assert_close(&u.to_vec().unwrap(), &wu, 1e-4, 1e-5);
        assert_close(&v.to_vec().unwrap(), &wv, 1e-4, 1e-5);
    });
}

#[test]
fn iterate_all_of_padded_assigned_cached_steps_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let u0 = data(t, round);
        let u = on(&ctx, &u0);
        let [u] = iterate_all([&u], 13, |[u]| [padded_step(u)]);
        let mut w = u0;
        for _ in 0..13 {
            w = padded(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
    });
}

#[test]
fn eval_loops_with_readbacks_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let u0 = data(t, round);
        let mut u = on(&ctx, &u0);
        let mut w = u0;
        for step in 0..24 {
            let next = (u.shift(1) + u.shift(-1)) * 0.25 + &u * 0.5;
            let change = (&next - &u).abs().max().item().unwrap();
            u = next;
            u.eval().unwrap();
            w = diffuse(&w);
            let want = w.iter().cloned().fold(f32::MIN, f32::max);
            assert!(change.is_finite(), "thread {t} step {step}");
            if step % 8 == 7 {
                assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
                assert_close(&[u.max().item().unwrap()], &[want], 1e-4, 1e-5);
            }
        }
    });
}

#[test]
fn per_step_loops_of_padded_steps_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let u0 = data(t, round);
        let mut u = on(&ctx, &u0);
        let mut w = u0;
        for _ in 0..20 {
            u = padded_step(&u);
            u.eval().unwrap();
            w = padded(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
    });
}

#[test]
fn iterate_until_all_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let (u0, v0) = (data(t, round), data(t + 100, round));
        let (u, v) = (on(&ctx, &u0), on(&ctx, &v0));
        let run = iterate_until_all(
            (u, v),
            Until::new(40).record("max", |s: &(Array<f32>, Array<f32>)| s.0.max()).check_every(8),
            |s, _| ((s.0.shift(1) + s.0.shift(-1)) * 0.25 + &s.0 * 0.5, &s.1 * 0.9 + &s.0 * 0.1),
        )
        .unwrap();
        let (mut wu, mut wv) = (u0, v0);
        let mut history = Vec::new();
        for _ in 0..40 {
            let nu = diffuse(&wu);
            wv = wv.iter().zip(&wu).map(|(v, u)| v * 0.9 + u * 0.1).collect();
            wu = nu;
            history.push(wu.iter().cloned().fold(f32::MIN, f32::max));
        }
        assert_eq!(run.steps, 40, "thread {t}");
        assert_close(&run.state.0.to_vec().unwrap(), &wu, 1e-4, 1e-5);
        assert_close(&run.state.1.to_vec().unwrap(), &wv, 1e-4, 1e-5);
        assert_close(run.history("max").unwrap(), &history, 1e-4, 1e-5);
    });
}

#[test]
fn every_kind_of_loop_at_once_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(8, 12, |t, round| match t % 4 {
        0 => {
            let u0 = data(t, round);
            let u = on(&ctx, &u0);
            let [u] = iterate_all([&u], 9, |[u]| [padded_step(u)]);
            let mut w = u0;
            for _ in 0..9 {
                w = padded(&w);
            }
            assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        }
        1 => {
            let u0 = data(t, round);
            let mut u = on(&ctx, &u0);
            let mut w = u0;
            for _ in 0..16 {
                let next = (u.shift(1) + u.shift(-1)) * 0.25 + &u * 0.5;
                let _ = (&next - &u).abs().max().item().unwrap();
                u = next;
                w = diffuse(&w);
            }
            assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        }
        2 => {
            let u0 = data(t, round);
            let u = on(&ctx, &u0);
            let run = iterate_until_all(
                (u,),
                Until::new(30).record("max", |s: &(Array<f32>,)| s.0.max()).check_every(8),
                |s, _| ((s.0.shift(1) + s.0.shift(-1)) * 0.25 + &s.0 * 0.5,),
            )
            .unwrap();
            let mut w = u0;
            for _ in 0..30 {
                w = diffuse(&w);
            }
            assert_close(&run.state.0.to_vec().unwrap(), &w, 1e-4, 1e-5);
        }
        _ => {
            let u0 = data(t, round);
            let mut u = on(&ctx, &u0);
            let mut w = u0;
            for _ in 0..12 {
                u = padded_step(&u);
                u.eval().unwrap();
                w = padded(&w);
            }
            assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        }
    });
}

/// Threads that keep the context busy with `eval`, `flush` and `sync` while others trace and replay.
#[test]
fn flushing_threads_do_not_disturb_a_recorded_step() {
    let ctx = default_context().unwrap();
    together(8, 15, |t, round| {
        let u0 = data(t, round);
        match t % 4 {
            0 | 1 => {
                let u = on(&ctx, &u0);
                let [u] = iterate_all([&u], 17, |[u]| [padded_step(u)]);
                let mut w = u0;
                for _ in 0..17 {
                    w = padded(&w);
                }
                assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
            }
            2 => {
                let u = on(&ctx, &u0);
                let run = iterate_until_all(
                    (u,),
                    Until::new(20).record("max", |s: &(Array<f32>,)| s.0.max()).check_every(5),
                    |s, _| (padded_step(&s.0),),
                )
                .unwrap();
                let mut w = u0;
                for _ in 0..20 {
                    w = padded(&w);
                }
                assert_close(&run.state.0.to_vec().unwrap(), &w, 1e-4, 1e-5);
            }
            _ => {
                // Lazy arrays everywhere, evaluated and flushed all the time.
                let mut u = on(&ctx, &u0);
                let mut w = u0;
                for step in 0..40 {
                    u = &u * 0.5 + u.shift(1) * 0.25 + 0.01;
                    w = (0..N).map(|i| w[i] * 0.5 + w[(i + N - 1) % N] * 0.25 + 0.01).collect();
                    match step % 3 {
                        0 => u.eval().unwrap(),
                        1 => ctx.flush().unwrap(),
                        _ => ctx.sync().unwrap(),
                    }
                }
                assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
            }
        }
    });
}

/// What `flush`, `sync` and `eval` of other threads computed (their own arrays, and, before they only took
/// those of the calling thread, the arrays a recording was tracing too) must not reach into a recording.
#[test]
fn flush_sync_and_eval_of_other_threads_leave_a_recording_alone() {
    let ctx = default_context().unwrap();
    together(4, 30, |t, round| {
        let u0 = data(t, round);
        if t < 2 {
            let u = on(&ctx, &u0);
            let [u] = iterate_all([&u], 17, |[u]| [padded_step(u)]);
            let mut w = u0;
            for _ in 0..17 {
                w = padded(&w);
            }
            assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        } else {
            let mut u = on(&ctx, &u0);
            for step in 0..60 {
                u = &u * 0.5 + 0.01;
                match step % 3 {
                    0 => u.eval().unwrap(),
                    1 => ctx.flush().unwrap(),
                    _ => ctx.sync().unwrap(),
                }
            }
            u.to_vec().unwrap();
        }
    });
}

/// The step assigns into a lazy padded state and does not stage it: the lazily assigned expression is
/// read by nodes recorded before it while it is being computed (evaluating alongside once picked
/// those, and the thread waited for itself; it needs a few threads to happen).
#[test]
fn iterate_all_of_lazily_assigned_steps_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 10, |t, round| {
        let u0 = data(t, round);
        let u = on(&ctx, &u0);
        let [u] = iterate_all([&u], 13, |[u]| {
            let n = u.len();
            let mut w = u.pad([(1, 1)], PadMode::Edge);
            w.slice_mut(0, ..1).assign(0.5).unwrap();
            [(w.slice(0, 1..n + 1) * 0.5 + w.slice(0, ..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25) * 0.99 + 0.001]
        });
        let mut w = u0;
        for _ in 0..13 {
            w = padded(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
    });
}

/// A step of an `eval` loop that is requested while another thread keeps the GPU busy stays lazy,
/// and the expression that follows it (`held`) computes it inside its own kernel: the array `s` is
/// then lazy, with a buffer it solely owns. A traced loop that reads `s` must find it computed: the
/// recording once evaluated it in place over that buffer, and the replays applied the last step of
/// the `eval` loop again at every step (it needs two threads, one of them busy, to happen).
#[test]
fn a_lazy_array_read_by_a_traced_loop_after_an_eval_loop_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(4, 40, |t, round| {
        let x0 = data(t, round);
        let mut s = on(&ctx, &x0);
        let mut w = x0;
        for _ in 0..3 {
            s = &s * 1.5 + 0.25;
            s.eval().unwrap();
            w = w.iter().map(|v| v * 1.5 + 0.25).collect();
        }
        let held = &s + 1.0;
        let u0 = data(t + 100, round);
        let [u] = iterate_all([&on(&ctx, &u0)], 5, |[u]| [u.cumsum(0) * 0.01 + s.cumsum(0)]);
        let cumsum = |v: &[f32]| -> Vec<f32> { v.iter().scan(0.0f32, |acc, &e| { *acc += e; Some(*acc) }).collect() };
        let mut wu = u0;
        for _ in 0..5 {
            wu = cumsum(&wu).iter().zip(cumsum(&w)).map(|(a, b)| a * 0.01 + b).collect();
        }
        assert_close(&held.to_vec().unwrap(), &w.iter().map(|v| v + 1.0).collect::<Vec<_>>(), 1e-4, 1e-5);
        assert_close(&s.to_vec().unwrap(), &w, 1e-4, 1e-5);
        assert_close(&u.to_vec().unwrap(), &wu, 1e-3, 1e-4);
    });
}

// ---- Risk areas: sweeps racing with drops, recycling under replays, nested flattening, supersede chains.

/// Thread 0 creates lazy arrays and drops them at once or after a few more operations (the node is
/// dying while it records further work) and traces loops; the other threads sweep over every
/// lazy array of the context with `eval`, `flush` and `sync` all the time. A sweep that upgraded the
/// weak reference of a node that is being dropped, and released its buffer, would show as a wrong value
/// or an error in the dropping thread.
#[test]
fn sweeps_of_other_threads_do_not_release_the_buffers_of_arrays_that_are_being_dropped() {
    let ctx = default_context().unwrap();
    together(4, 25, |t, round| {
        let u0 = data(t, round);
        if t == 0 {
            let u = on(&ctx, &u0);
            let mut keep = Vec::new();
            for k in 0..60 {
                let scale = 1.0 + k as f32 * 0.01;
                let a = &u * scale + 0.5;
                let b = a.shift(1) * 0.5 + &a;
                let want: Vec<f32> = {
                    let a: Vec<f32> = u0.iter().map(|v| v * scale + 0.5).collect();
                    (0..N).map(|i| a[(i + N - 1) % N] * 0.5 + a[i]).collect()
                };
                match k % 4 {
                    0 => drop(a),
                    1 => {
                        drop(a);
                        drop(b);
                    }
                    2 => {
                        drop(b);
                        keep.push((a, None));
                    }
                    _ => keep.push((a, Some((b, want)))),
                }
            }
            for (_, b) in &keep {
                if let Some((b, want)) = b {
                    assert_close(&b.to_vec().unwrap(), want, 1e-4, 1e-5);
                }
            }
            let [r] = iterate_all([&u], 9, |[u]| [padded_step(u)]);
            let mut w = u0;
            for _ in 0..9 {
                w = padded(&w);
            }
            assert_close(&r.to_vec().unwrap(), &w, 1e-4, 1e-5);
        } else {
            let mut u = on(&ctx, &u0);
            for step in 0..80 {
                let half = &u * 0.5 + 0.25;
                u = &half + 0.0 * &u;
                match step % 3 {
                    0 => u.eval().unwrap(),
                    1 => ctx.flush().unwrap(),
                    _ => ctx.sync().unwrap(),
                }
            }
            u.to_vec().unwrap();
        }
    });
}

/// A traced loop replays its recorded launches over buffers it holds; other threads allocate and
/// drop arrays of the same size class all the time, so their buffers are recycled through the pool
/// the recording's buffers came from. Every thread checks what it computed.
#[test]
fn recycled_buffers_do_not_reach_into_the_plan_another_thread_replays() {
    let ctx = default_context().unwrap();
    together(5, 20, |t, round| {
        let u0 = data(t, round);
        if t == 0 {
            let u = on(&ctx, &u0);
            let [u] = iterate_all([&u], 31, |[u]| [padded_step(u)]);
            let mut w = u0;
            for _ in 0..31 {
                w = padded(&w);
            }
            assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        } else {
            for k in 0..40 {
                let scale = 1.0 + (k % 5) as f32;
                let a = on(&ctx, &u0);
                let b = (&a * scale).cache();
                let c = &b + 1.0;
                drop(a);
                let got = c.to_vec().unwrap();
                drop(b);
                let want: Vec<f32> = u0.iter().map(|v| v * scale + 1.0).collect();
                assert_close(&got, &want, 1e-5, 1e-6);
                if (k as u32).is_multiple_of(7) {
                    ctx.sync().unwrap();
                }
            }
        }
    });
}

/// A long chain over the node limit: its operands are cut into several kernels, and evaluating one
/// array evaluates the others that share work with it alongside (flattening inside flattening,
/// `COMPUTING` guards inside each other), while the other threads do the same with their own.
#[test]
fn evaluating_alongside_under_a_tight_node_limit_on_a_shared_context() {
    let ctx = default_context().unwrap();
    together(6, 15, |t, round| {
        let u0 = data(t, round);
        let u = on(&ctx, &u0);
        let (a, b, c, d) = luma::fusion_limit(4, || {
            let a = (((&u * 1.5 + 0.25) * 0.5 + 0.125) * 0.75 + 0.0625) * 0.5 + 0.25;
            let b = luma::fusion_limit(3, || ((&a * 2.0 - 0.5) * 0.5 + a.shift(1) * 0.25) * 0.5 + 0.125);
            let c = (a.shift(-1) + &b) * 0.5 + (&b * &a) * 0.125;
            let d = &c * 0.5 + &a * 0.25;
            (a, b, c, d)
        });
        let ha: Vec<f32> = u0.iter().map(|v| (((v * 1.5 + 0.25) * 0.5 + 0.125) * 0.75 + 0.0625) * 0.5 + 0.25).collect();
        let hb: Vec<f32> = (0..N).map(|i| ((ha[i] * 2.0 - 0.5) * 0.5 + ha[(i + N - 1) % N] * 0.25) * 0.5 + 0.125).collect();
        let hc: Vec<f32> = (0..N).map(|i| (ha[(i + 1) % N] + hb[i]) * 0.5 + hb[i] * ha[i] * 0.125).collect();
        let hd: Vec<f32> = (0..N).map(|i| hc[i] * 0.5 + ha[i] * 0.25).collect();
        // Read in an order that makes the first one pull the others in alongside.
        let order = [(t + round) % 4, (t + round + 1) % 4, (t + round + 2) % 4, (t + round + 3) % 4];
        for which in order {
            match which {
                0 => assert_close(&a.to_vec().unwrap(), &ha, 1e-4, 1e-5),
                1 => assert_close(&b.to_vec().unwrap(), &hb, 1e-4, 1e-5),
                2 => assert_close(&c.to_vec().unwrap(), &hc, 1e-4, 1e-5),
                _ => assert_close(&d.to_vec().unwrap(), &hd, 1e-4, 1e-5),
            }
        }
    });
}

/// The padded state is cloned and the clones are assigned through slices: each assignment supersedes
/// the expression the earlier clones hold, in a per-step `eval` loop and in a traced loop, on several threads.
#[test]
fn assigned_clones_of_a_lazy_padded_state_on_a_shared_context() {
    let ctx = default_context().unwrap();
    // One step: `w` padded, two clones assigned at both ends through slices, read through the clones.
    fn step(u: &Array<f32>) -> Array<f32> {
        let n = u.len();
        let w = u.pad([(1, 1)], PadMode::Edge);
        let mut left = w.clone();
        let mut right = w.clone();
        left.slice_mut(0, ..1).assign(0.5).unwrap();
        right.slice_mut(0, n + 1..).assign(0.25).unwrap();
        (w.slice(0, 1..n + 1) * 0.5 + left.slice(0, ..n) * 0.25 + right.slice(0, 2..n + 2) * 0.25) * 0.99 + 0.001
    }
    fn host(u: &[f32]) -> Vec<f32> {
        let n = u.len();
        let w: Vec<f32> = std::iter::once(0.5).chain(u.iter().copied()).chain(std::iter::once(0.25)).collect();
        // `w` is one array: both clones see both writes.
        (0..n).map(|i| (0.5 * w[i + 1] + 0.25 * w[i] + 0.25 * w[i + 2]) * 0.99 + 0.001).collect()
    }
    together(6, 10, |t, round| {
        let u0 = data(t, round);
        let mut u = on(&ctx, &u0);
        let mut w = u0.clone();
        for _ in 0..15 {
            u = step(&u);
            u.eval().unwrap();
            w = host(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
        let u = on(&ctx, &u0);
        let [u] = iterate_all([&u], 11, |[u]| [step(u)]);
        let mut w = u0;
        for _ in 0..11 {
            w = host(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
    });
}

/// Threads that trace a loop step at the same time all replay it: the recordings of the threads of a context do
/// not exclude each other. A thread that could not record ran its loop on the host instead, calling the step
/// closure at every step, which is what the call counts tell apart. Every thread waits inside its step until all
/// of them are tracing.
#[test]
fn threads_that_trace_at_the_same_time_all_replay() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let ctx = default_context().unwrap();
    let threads = 4;
    let (all_tracing, steps) = (Barrier::new(threads), 24);
    let calls: Vec<[AtomicUsize; 2]> = (0..threads).map(|_| [AtomicUsize::new(0), AtomicUsize::new(0)]).collect();
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (ctx, all_tracing, calls) = (&ctx, &all_tracing, &calls);
            scope.spawn(move || {
                let u0 = data(t, 0);
                let u = on(ctx, &u0);
                let [u] = iterate_all([&u], steps, |[u]| {
                    if calls[t][0].fetch_add(1, Ordering::SeqCst) == 0 {
                        all_tracing.wait();
                    }
                    [padded_step(u)]
                });
                let mut w = u0.clone();
                for _ in 0..steps {
                    w = padded(&w);
                }
                assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);

                let u = on(ctx, &u0);
                let run = iterate_until_all((u,), Until::new(steps).record("max", |s: &(Array<f32>,)| s.0.max()).check_every(8), |s, _| {
                    if calls[t][1].fetch_add(1, Ordering::SeqCst) == 0 {
                        all_tracing.wait();
                    }
                    (padded_step(&s.0),)
                })
                .unwrap();
                let mut w = u0;
                for _ in 0..steps {
                    w = padded(&w);
                }
                assert_close(&run.state.0.to_vec().unwrap(), &w, 1e-4, 1e-5);
            });
        }
    });
    for (t, [all, until]) in calls.iter().enumerate() {
        assert_eq!(all.load(Ordering::SeqCst), 1, "thread {t}: iterate_all called its step again instead of replaying it");
        assert_eq!(until.load(Ordering::SeqCst), 1, "thread {t}: iterate_until_all called its step again instead of replaying it");
    }
}
