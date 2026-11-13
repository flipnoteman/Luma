//! Threads that share one context against threads with a context each.
//!
//! `cargo run --release --example bench_threads -- [--threads N] [--steps N] [--size N] [--rounds N] [--df32] [--only <workload>]`
//!
//! Every thread runs the same loop of a 5-point stencil step (a 24-operation expression of the state, padded and read
//! at four offsets, so each step is a handful of fused kernels with views) in three forms:
//!
//! - `until`: `iterate_until_all` with a recorded residual (traced and replayed),
//! - `all`: `iterate_all` (traced and replayed),
//! - `eval`: a host loop that evaluates the state at every step and reads the maximum every 16 steps.
//!
//! The threads start together (a barrier) and a round ends when the slowest one is done. Round 0 compiles the kernels
//! (a context compiles each kernel once, the threads of a shared context share them); later rounds run them from the
//! caches. The reference is one thread alone on a context of its own: threads that do not disturb each other take
//! about as long as it does, as long as the GPU is not the bottleneck. Every thread's final state is compared with the
//! single-thread result (kernels fused differently may round differently, so not bit for bit).

#![recursion_limit = "512"]

use luma::*;
use std::sync::Barrier;
use std::time::Instant;

/// A 24-operation elementwise expression: four rounds of `max(|sqrt(e * 0.9 + 0.2)|, e * 0.5)`.
fn heavy(x: &Array<f32>) -> Array<f32> {
    let mut e = x.clone();
    for _ in 0..4 {
        e = (&e * 0.9 + 0.2).sqrt().abs().maximum(&e * 0.5);
    }
    e
}

/// [`heavy`] on `Df32`.
fn heavy_df32(x: &Array<Df32>) -> Array<Df32> {
    let (a, b, c) = (Df32::from(0.9), Df32::from(0.2), Df32::from(0.5));
    let mut e = x.clone();
    for _ in 0..4 {
        e = (&e * a + b).sqrt().abs().maximum(&e * c);
    }
    e
}

/// One step: `u + 0.05 * laplacian(heavy(u))` with the edge values repeated outside.
fn step(u: &Array<f32>, w: usize, h: usize, df32: bool) -> Array<f32> {
    let e = if df32 { heavy_df32(&u.to_df32()).to_f32() } else { heavy(u) };
    let p = e.pad([(1, 1), (1, 1)], PadMode::Edge);
    let centre = p.slice(0, 1..w + 1).slice(1, 1..h + 1);
    let around = p.slice(0, 0..w).slice(1, 1..h + 1)
        + p.slice(0, 2..w + 2).slice(1, 1..h + 1)
        + p.slice(0, 1..w + 1).slice(1, 0..h)
        + p.slice(0, 1..w + 1).slice(1, 2..h + 2);
    u + (around - centre * 4.0) * 0.05
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Work {
    Until,
    All,
    Eval,
}

impl Work {
    fn name(self) -> &'static str {
        match self {
            Work::Until => "until",
            Work::All => "all",
            Work::Eval => "eval",
        }
    }
}

struct Setup {
    w: usize,
    h: usize,
    steps: usize,
    df32: bool,
}

/// Runs one workload on `ctx` and returns the final state and the checksum of what was read back.
fn run(ctx: &Context, work: Work, s: &Setup) -> (Vec<f32>, f32) {
    let data: Vec<f32> = (0..s.w * s.h).map(|i| 0.2 + 0.8 * ((i * 7919) % 1000) as f32 / 1000.0).collect();
    let u0 = Array::from_slice_on(ctx, &data, [s.w, s.h]).unwrap();
    let (w, h, df32) = (s.w, s.h, s.df32);
    match work {
        Work::Until => {
            let run = iterate_until_all(
                (u0,),
                Until::new(s.steps).check_every(16).record("max", |state: &(Array<f32>,)| state.0.max()),
                |state, _| (step(&state.0, w, h, df32),),
            )
            .unwrap();
            let sum = run.history("max").unwrap_or_default().iter().sum();
            (run.state.0.to_vec().unwrap(), sum)
        }
        Work::All => {
            let [u] = iterate_all([&u0], s.steps, |state| [step(&state[0], w, h, df32)]);
            (u.to_vec().unwrap(), 0.0)
        }
        Work::Eval => {
            let mut u = u0;
            let mut sum = 0.0;
            for t in 0..s.steps {
                u = step(&u, w, h, df32);
                u.eval().unwrap();
                if t % 16 == 15 {
                    sum += u.max().item().unwrap();
                }
            }
            (u.to_vec().unwrap(), sum)
        }
    }
}

/// One round: every thread runs `work` on its context once, started together. Returns the wall time of the round
/// (until the last thread is done) and the times of the threads in milliseconds, and checks the results.
fn round(ctxs: &[Context], work: Work, s: &Setup, want: &(Vec<f32>, f32)) -> (f64, Vec<f64>) {
    let barrier = Barrier::new(ctxs.len() + 1);
    let mut times = vec![0.0; ctxs.len()];
    let wall = std::thread::scope(|scope| {
        let threads: Vec<_> = ctxs
            .iter()
            .map(|ctx| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let start = Instant::now();
                    let got = run(ctx, work, s);
                    (start.elapsed().as_secs_f64() * 1e3, got)
                })
            })
            .collect();
        barrier.wait();
        let start = Instant::now();
        for (time, thread) in times.iter_mut().zip(threads) {
            let (elapsed, got) = thread.join().unwrap();
            let worst = got.0.iter().zip(&want.0).fold((got.1 - want.1).abs() / (1.0 + want.1.abs()), |m, (a, b)| m.max((a - b).abs() / (1.0 + b.abs())));
            assert!(worst < 1e-4, "{}: a thread's result differs from the single-thread run by {worst:e}", work.name());
            *time = elapsed;
        }
        start.elapsed().as_secs_f64() * 1e3
    });
    (wall, times)
}

fn main() {
    let (mut threads, mut steps, mut size, mut rounds, mut df32, mut only) = (4usize, 64usize, 64usize, 3usize, false, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = |what: &str| args.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| panic!("{what} takes a number"));
        match arg.as_str() {
            "--threads" => threads = number("--threads").max(1),
            "--steps" => steps = number("--steps").max(1),
            "--size" => size = number("--size").max(4),
            "--rounds" => rounds = number("--rounds").max(1),
            "--df32" => df32 = true,
            "--only" => only = args.next(),
            "--bench" => {}
            other => panic!("unknown argument {other}; usage: bench_threads [--threads N] [--steps N] [--size N] [--rounds N] [--df32] [--only <workload>]"),
        }
    }
    let setup = Setup { w: size, h: size, steps, df32 };
    println!(
        "bench_threads: {threads} threads, {size}x{size} {}, {steps} steps, {rounds} rounds (round 0 compiles)",
        if df32 { "Df32 arithmetic" } else { "f32" }
    );
    println!("{:<6} {:<8} {:>5} {:>10} {:>10} {:>10}", "work", "contexts", "round", "wall ms", "fastest", "slowest");
    for work in [Work::Until, Work::All, Work::Eval] {
        if only.as_deref().is_some_and(|name| name != work.name()) {
            continue;
        }
        // The reference: one thread alone on a context of its own.
        let alone = vec![Context::new().unwrap()];
        let want = run(&alone[0], work, &setup);
        let mut best_alone = f64::INFINITY;
        for r in 0..rounds {
            let (wall, _) = round(&alone, work, &setup, &want);
            if r > 0 {
                best_alone = best_alone.min(wall);
            }
            println!("{:<6} {:<8} {:>5} {:>10.1} {:>10} {:>10}", work.name(), "alone", r, wall, "", "");
        }
        let mut best = [f64::INFINITY; 2];
        for (m, mode) in ["shared", "private"].into_iter().enumerate() {
            let ctxs: Vec<Context> = (0..threads)
                .map(|_| if m == 0 { default_context().unwrap() } else { Context::new().unwrap() })
                .collect();
            for r in 0..rounds {
                let (wall, times) = round(&ctxs, work, &setup, &want);
                if r > 0 {
                    best[m] = best[m].min(wall);
                }
                let (fastest, slowest) = times.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &t| (lo.min(t), hi.max(t)));
                println!("{:<6} {:<8} {:>5} {:>10.1} {:>10.1} {:>10.1}", work.name(), mode, r, wall, fastest, slowest);
            }
        }
        if rounds > 1 {
            println!(
                "{:<6} warm best: alone {best_alone:.1} ms, {threads} threads shared {:.1} ms ({:.2}x alone), private {:.1} ms ({:.2}x alone), shared/private {:.2}",
                work.name(),
                best[0],
                best[0] / best_alone,
                best[1],
                best[1] / best_alone,
                best[0] / best[1]
            );
        }
    }
}
