//! First-call cost of a recorded step with many distinct kernels: the step of `iterate_all`
//! (cached stages and shifts, so it is traced once and replayed) has [`K`] states with pairwise
//! different expressions, so tracing it builds `K` or more kernels. Run with an empty `LUMA_CACHE_DIR` (and
//! `CUDA_CACHE_DISABLE=1`) for a cold kernel cache; `LUMA_COMPILE_THREADS=0` compiles on the calling
//! thread (the behavior before kernels compiled in the background).
//!
//! `PHASES=1` prints, per kernel on CUDA, the time of luma's code generation, NVVM and the module load. The
//! cold first calls of single elementwise and reduction kernels are timed too.
//!
//! `cargo run --release --example bench_coldtrace [-- steps]`

use luma::{iterate_all, Array, PadMode};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// States of the step, each with an expression of its own.
const K: usize = 24;

static COMPILED: AtomicUsize = AtomicUsize::new(0);

struct Counter;

impl log::Log for Counter {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Debug
    }
    fn log(&self, r: &log::Record) {
        let line = r.args().to_string();
        if line.starts_with("luma: compiled") {
            COMPILED.fetch_add(1, Ordering::Relaxed);
        } else if line.starts_with("luma-cuda:") && std::env::var_os("PHASES").is_some() {
            println!("  {line}");
        }
    }
    fn flush(&self) {}
}

fn stage(k: usize, a: &Array<f32>, b: &Array<f32>) -> Array<f32> {
    // The structure (operations and their order) depends on `k`, so each is a kernel of its own.
    let mut acc = a * 0.5 + b * 0.25;
    for j in 0..(k % 9) + 2 {
        acc = match (k / 9 + j * 3 + k) % 7 {
            0 => (&acc - b).abs() + 0.1,
            1 => &acc * b + 0.01,
            2 => (&acc * &acc + 1.0).sqrt(),
            3 => acc.sin() * 0.5 + a * 0.5,
            4 => &acc / (b.abs() + 1.5),
            5 => acc.maximum(b) * 0.9,
            _ => acc.minimum(a) + 0.2 * b,
        };
        if j % 3 == 2 {
            acc = acc.cache();
        }
    }
    // A shifted read, so the step is not a register loop.
    let n = acc.len();
    let shifted = acc.slice(0, 1..n).pad([(0, 1)], PadMode::Edge);
    (acc + shifted * 0.01).cache() * 0.5
}

/// First-call time of `build` (a new kernel structure per `k`, evaluated and read back).
fn single(name: &str, ctx: &luma::Context, build: impl Fn(&Array<f32>, usize) -> Array<f32>) {
    let x = Array::from_slice_on(ctx, &(0..1 << 20).map(|i| 1.0 + 0.1 * (i as f32 * 0.37).sin()).collect::<Vec<_>>(), [1 << 20]).unwrap();
    ctx.sync().unwrap();
    let mut times = Vec::new();
    for k in 0..6 {
        let t0 = Instant::now();
        build(&x, k).to_vec().unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("{name}: first calls of 6 new kernels (ms): {}", times.iter().map(|t| format!("{t:.1}")).collect::<Vec<_>>().join(" "));
}

fn main() {
    let _ = log::set_logger(&Counter).map(|()| log::set_max_level(log::LevelFilter::Debug));
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ctx = luma::default_context().unwrap();
    let init: [Array<f32>; K] = std::array::from_fn(|i| {
        Array::from_slice_on(&ctx, &(0..4096).map(|x| 1.0 + 0.1 * ((x + 7 * i) as f32 * 0.37).sin()).collect::<Vec<_>>(), [4096]).unwrap()
    });
    ctx.sync().unwrap();
    let refs: [&Array<f32>; K] = std::array::from_fn(|i| &init[i]);
    let run = || {
        let t0 = Instant::now();
        let out = iterate_all(refs, steps, |s| std::array::from_fn(|i| stage(i, &s[i], &s[(i + 1) % K])));
        let traced = t0.elapsed();
        let sums: Vec<f32> = out.iter().map(|a| a.sum().to_vec().unwrap()[0]).collect();
        (traced, t0.elapsed(), sums)
    };
    let (traced, first, sums) = run();
    let compiled = COMPILED.load(Ordering::Relaxed);
    let (_, second, again) = run();
    assert_eq!(sums, again, "results differ between runs");
    let checksum: f64 = sums.iter().map(|&x| f64::from(x)).sum();
    single("elementwise (12 ops)", &ctx, |x, k| {
        let mut acc = x * 1.5;
        for j in 0..12 {
            acc = match (j + k) % 4 {
                0 => acc.sin() + x,
                1 => &acc * x + 0.25,
                2 => (&acc * &acc + 1.0).sqrt(),
                _ => acc.abs() - x * 0.5,
            };
        }
        acc
    });
    single("reduction", &ctx, |x, k| match k {
        0 => x.sum(),
        1 => x.max(),
        2 => x.min(),
        3 => (x * x).sum(),
        4 => (x.sin() * x).max(),
        _ => (x + 1.0).sqrt().min(),
    });
    println!(
        "{} kernels compiled; first call: trace {:.0} ms, with results {:.0} ms; second call {:.0} ms; checksum {checksum:.6}",
        compiled,
        traced.as_secs_f64() * 1e3,
        first.as_secs_f64() * 1e3,
        second.as_secs_f64() * 1e3
    );
}
