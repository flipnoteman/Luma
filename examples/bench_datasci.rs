//! Benchmarks of the data-science primitives against hand-written WGSL kernels.
//!
//! `cargo run --release --example bench_datasci -- [--quick] [--gate] [--only <substring>]`
//!
//! Every case verifies both sides against a host reference, then times luma and the hand-written
//! kernel of `datasci/hand.rs` alike: GPU time from [`Context::profile`] (best of 7 after 2
//! warm-ups) and wall time. One row per case: `case size luma_gpu_ms hand_gpu_ms ratio
//! luma_wall_ms hand_wall_ms rate unit`, `ratio` being luma over hand. `--gate` exits with status 1
//! if a ratio exceeds its threshold, a roofline case misses its minimum fraction of the device
//! bandwidth, or a case is skipped (panic or mismatch). Without GPU timestamps (lavapipe) the
//! gate compares wall times.
//!
//! To add a case: write a `fn(&mut Bench)`, run it through `Bench::guarded`, verify with
//! `check`/`check_exact`, then call `Bench::compare` or `Bench::roofline`.

#[path = "datasci/hand.rs"]
mod hand;

use luma::*;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use std::time::Instant;

// =================================================================================================
// Harness
// =================================================================================================

static PANIC: Mutex<String> = Mutex::new(String::new());

/// Default `--gate` threshold on `luma GPU time / hand GPU time`.
#[allow(dead_code)]
const DEFAULT_GATE: f64 = 1.05;
/// Warm-up runs before timing.
const WARMUPS: usize = 2;
/// Timed runs per measurement; the best is reported.
const RUNS: usize = 7;

/// What a case computes, for its rate column.
#[allow(dead_code)]
enum Work {
    /// Bytes an ideal kernel moves -> GB/s.
    Bytes(f64),
    /// Floating-point operations -> GFLOP/s.
    Flops(f64),
}

/// Best (min) times of one side, in ms. `gpu` is `None` without GPU timestamps.
struct Times {
    gpu: Option<f64>,
    wall: f64,
}

impl Times {
    /// The time the gate and the rate use: GPU when measured, wall otherwise.
    fn gate_ms(&self) -> f64 {
        self.gpu.unwrap_or(self.wall)
    }
}

/// A case the gate looks at: `value` is the ratio (or the roofline fraction), `limit` its bound.
struct Gated {
    name: String,
    size: String,
    value: f64,
    /// `true`: fails when `value > limit` (ratio); `false`: fails when `value < limit` (fraction).
    ratio: bool,
    limit: f64,
}

struct Bench {
    ctx: Context,
    quick: bool,
    gate: bool,
    only: Option<String>,
    gated: Vec<Gated>,
    skipped: Vec<(String, String)>,
}

impl Bench {
    fn want(&self, name: &str) -> bool {
        self.only.as_deref().is_none_or(|f| name.contains(f))
    }

    /// Runs `f` if the case is selected; a panic (library failure or verification mismatch) is
    /// reported as a SKIPPED line.
    fn guarded(&mut self, name: &str, f: impl FnOnce(&mut Bench)) {
        if !self.want(name) {
            return;
        }
        let result = catch_unwind(AssertUnwindSafe(|| f(self)));
        if result.is_err() {
            let msg = PANIC.lock().unwrap().clone().replace('\n', " | ");
            println!("SKIPPED: {name}: {msg}");
            self.skipped.push((name.to_string(), msg));
            let _ = self.ctx.finish_profiling();
            let _ = self.ctx.sync();
        }
    }

    /// Best GPU and wall time of `f`, which must enqueue the whole computation.
    fn measure(&self, f: &mut dyn FnMut()) -> Times {
        let ctx = &self.ctx;
        for _ in 0..WARMUPS {
            f();
            ctx.sync().unwrap();
        }
        let mut wall = f64::INFINITY;
        for _ in 0..RUNS {
            let t = Instant::now();
            f();
            ctx.sync().unwrap();
            wall = wall.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let gpu = ctx.profiling_supported().then(|| {
            (0..RUNS)
                .map(|_| {
                    let ((), p) = ctx.profile(|| {
                        f();
                        ctx.sync().unwrap();
                    })
                    .unwrap();
                    p.total().as_secs_f64() * 1e3
                })
                .fold(f64::INFINITY, f64::min)
        });
        Times { gpu, wall }
    }

    /// Times `luma` against `hand`, prints the row and records the case for the gate (`gate` is
    /// the largest allowed ratio, [`DEFAULT_GATE`] usually; `None` for an informational row).
    /// Each closure must enqueue the full computation, forcing evaluation (`.eval().unwrap()`).
    fn compare(
        &mut self,
        name: &str,
        size: &str,
        work: Work,
        gate: Option<f64>,
        luma: &mut dyn FnMut(),
        hand: &mut dyn FnMut(),
    ) {
        let l = self.measure(luma);
        let h = self.measure(hand);
        let ratio = l.gate_ms() / h.gate_ms();
        let (rate, unit) = match work {
            Work::Bytes(b) => (b / (l.gate_ms() * 1e-3) / 1e9, "GB/s"),
            Work::Flops(f) => (f / (l.gate_ms() * 1e-3) / 1e9, "GFLOP/s"),
        };
        let gpu = |t: &Times| t.gpu.map_or("-".to_string(), |g| format!("{g:.3}"));
        println!(
            "{name:<24} {size:>14} {:>10} {:>10} {ratio:>6.3} {:>10.3} {:>10.3} {rate:>9.1} {unit}",
            gpu(&l),
            gpu(&h),
            l.wall,
            h.wall
        );
        if let Some(limit) = gate {
            self.gated.push(Gated { name: name.to_string(), size: size.to_string(), value: ratio, ratio: true, limit });
        }
    }

    /// Times `f` alone and gates the achieved bandwidth (`bytes` moved by an ideal kernel) on being
    /// at least `min_fraction` of the device's [`CostModel::bandwidth_gbs`].
    #[allow(dead_code)]
    fn roofline(&mut self, name: &str, size: &str, bytes: f64, min_fraction: f64, f: &mut dyn FnMut()) {
        let t = self.measure(f);
        let gbs = bytes / (t.gate_ms() * 1e-3) / 1e9;
        let fraction = gbs / self.ctx.cost_model().bandwidth_gbs;
        let gpu = t.gpu.map_or("-".to_string(), |g| format!("{g:.3}"));
        println!(
            "{name:<24} {size:>14} {gpu:>10} {:>10} {:>6} {:>10.3} {:>10} {gbs:>9.1} GB/s ({:.0}% of bandwidth)",
            "-",
            "-",
            t.wall,
            "-",
            fraction * 100.0
        );
        self.gated.push(Gated { name: name.to_string(), size: size.to_string(), value: fraction, ratio: false, limit: min_fraction });
    }
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(rel * w.abs());
        assert!((g - w).abs() <= tol, "{name}: index {i}: got {g}, want {w} (tol {tol})");
    }
}

#[allow(dead_code)]
fn check_exact<T: PartialEq + Debug>(name: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g == w, "{name}: index {i}: got {g:?}, want {w:?}");
    }
}

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

/// `n` pseudo-random codes in `0..modulo`.
#[allow(dead_code)]
fn codes(n: usize, seed: u64, modulo: u32) -> Vec<u32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % modulo as u64) as u32
        })
        .collect()
}

// =================================================================================================
// Cases
// =================================================================================================

/// Checks the hand-written SGEMM against a host GEMM, so Phase 2 compares against a correct one.
fn sanity_sgemm(b: &mut Bench) {
    b.guarded("hand/sgemm", |b| {
        let ctx = b.ctx.clone();
        let n = 256;
        let ha = values(n * n, 11, -1.0, 1.0);
        let hb = values(n * n, 12, -1.0, 1.0);
        let mut want = vec![0.0f32; n * n];
        for i in 0..n {
            for k in 0..n {
                let a = ha[i * n + k];
                for j in 0..n {
                    want[i * n + j] += a * hb[k * n + j];
                }
            }
        }
        let a = Array::from_slice_on(&ctx, &ha, n * n).unwrap();
        let bm = Array::from_slice_on(&ctx, &hb, n * n).unwrap();
        let mut c = Array::<f32>::zeros_on(&ctx, n * n).unwrap();
        hand::HandSgemm::new(&ctx).unwrap().run(&a, &bm, &mut c, n, n, n);
        check("hand/sgemm", &c.to_vec().unwrap(), &want, 1e-4, 1e-4);
        println!("hand sgemm ok");
    });
}

/// `2 x + y`: the harness smoke test (informational).
fn smoke_axpy(b: &mut Bench) {
    b.guarded("smoke/axpy", |b| {
        let ctx = b.ctx.clone();
        let n = if b.quick { 1 << 16 } else { 1 << 24 };
        let hx = values(n, 1, -1.0, 1.0);
        let hy = values(n, 2, -1.0, 1.0);
        let want: Vec<f32> = hx.iter().zip(&hy).map(|(x, y)| 2.0 * x + y).collect();
        let x = Array::from_slice_on(&ctx, &hx, n).unwrap();
        let y = Array::from_slice_on(&ctx, &hy, n).unwrap();
        let mut out = Array::<f32>::zeros_on(&ctx, n).unwrap();
        let hand = hand::HandSaxpy::new(&ctx).unwrap();

        let r = &x * 2.0 + &y;
        r.eval().unwrap();
        check("smoke/axpy luma", &r.to_vec().unwrap(), &want, 1e-6, 1e-6);
        hand.run(2.0, &x, &y, &mut out, n);
        check("smoke/axpy hand", &out.to_vec().unwrap(), &want, 1e-6, 1e-6);

        b.compare(
            "smoke/axpy",
            &n.to_string(),
            Work::Bytes(12.0 * n as f64),
            None,
            &mut || {
                (&x * 2.0 + &y).eval().unwrap();
            },
            &mut || hand.run(2.0, &x, &y, &mut out, n),
        );
    });
}

fn main() {
    let mut quick = false;
    let mut gate = false;
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => quick = true,
            "--gate" => gate = true,
            "--only" => only = args.next(),
            "--bench" => {}
            other => panic!("unknown argument {other}; usage: bench_datasci [--quick] [--gate] [--only <substring>]"),
        }
    }
    std::panic::set_hook(Box::new(|info| {
        *PANIC.lock().unwrap() = info.to_string();
    }));
    let ctx = default_context().unwrap();
    let mut b = Bench { ctx, quick, gate, only, gated: Vec::new(), skipped: Vec::new() };
    println!("bench_datasci ({}), profiling supported: {}", if quick { "quick" } else { "default sizes" }, b.ctx.profiling_supported());
    if gate && !b.ctx.profiling_supported() {
        println!("wall-clock gate");
    }
    println!(
        "{:<24} {:>14} {:>10} {:>10} {:>6} {:>10} {:>10} {:>9} unit",
        "case", "size", "luma gpu", "hand gpu", "ratio", "luma wall", "hand wall", "rate"
    );
    sanity_sgemm(&mut b);
    smoke_axpy(&mut b);

    if !b.skipped.is_empty() {
        println!("\n{} case(s) SKIPPED:", b.skipped.len());
        for (n, why) in &b.skipped {
            println!("  {n}: {why}");
        }
    }
    if b.gate {
        let mut failed = false;
        for g in &b.gated {
            if (g.ratio && g.value > g.limit) || (!g.ratio && g.value < g.limit) {
                println!("GATE FAIL {} {} {:.3}", g.name, g.size, g.value);
                failed = true;
            }
        }
        for (n, _) in &b.skipped {
            println!("GATE FAIL {n} skipped");
            failed = true;
        }
        if failed {
            std::process::exit(1);
        }
    }
}
