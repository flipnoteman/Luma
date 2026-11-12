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
    let w = u.pad([(1, 1)], PadMode::Edge);
    w.slice(0, ..1).assign(0.5).unwrap();
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
                    u = &u * 0.5 + (&u).shift(1) * 0.25 + 0.01;
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
            let w = u.pad([(1, 1)], PadMode::Edge);
            w.slice(0, ..1).assign(0.5).unwrap();
            [(w.slice(0, 1..n + 1) * 0.5 + w.slice(0, ..n) * 0.25 + w.slice(0, 2..n + 2) * 0.25) * 0.99 + 0.001]
        });
        let mut w = u0;
        for _ in 0..13 {
            w = padded(&w);
        }
        assert_close(&u.to_vec().unwrap(), &w, 1e-4, 1e-5);
    });
}
