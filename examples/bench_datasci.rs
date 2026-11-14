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

/// Checks the hand-written Gram kernel against a host f64 computation (full and ragged K slices)
/// and reports its speed at `F = 64`, `K = 2^20`.
fn sanity_gram(b: &mut Bench) {
    b.guarded("hand/gram", |b| {
        let ctx = b.ctx.clone();
        let f = 64usize;
        let hand = hand::HandGram::new(&ctx).unwrap();
        let hc = values(f, 31, -0.5, 0.5);
        let hs = values(f, 32, 0.5, 2.0);
        let c = Array::from_slice_on(&ctx, &hc, f).unwrap();
        let s = Array::from_slice_on(&ctx, &hs, f).unwrap();
        for k in [1usize << 16, 5000] {
            let hx = values(f * k, 30 + k as u64, -1.0, 1.0);
            let z: Vec<f64> = (0..f * k).map(|i| ((hx[i] - hc[i % f]) * hs[i % f]) as f64).collect();
            let mut want = vec![0.0f32; f * f];
            for a in 0..f {
                for bb in 0..f {
                    want[a * f + bb] = (0..k).map(|r| z[r * f + a] * z[r * f + bb]).sum::<f64>() as f32;
                }
            }
            let x = Array::from_slice_on(&ctx, &hx, [f, k]).unwrap();
            let mut out = Array::<f32>::zeros_on(&ctx, f * f).unwrap();
            hand.run(&x, &c, &s, &mut out, f as u32, k as u32);
            check("hand/gram", &out.to_vec().unwrap(), &want, 1e-3, 1e-3 * (k as f32).sqrt());
        }
        println!("hand gram ok");

        let k = if b.quick { 1usize << 16 } else { 1 << 20 };
        let x = Array::from_slice_on(&ctx, &values(f * k, 33, -1.0, 1.0), [f, k]).unwrap();
        let mut out = Array::<f32>::zeros_on(&ctx, f * f).unwrap();
        let t = b.measure(&mut || hand.run(&x, &c, &s, &mut out, f as u32, k as u32));
        let ms = t.gate_ms();
        println!(
            "hand/gram F={f} K={k}: gpu {:?} ms, wall {:.3} ms, {:.1} GFLOP/s",
            t.gpu.map(|g| (g * 1e3).round() / 1e3),
            t.wall,
            2.0 * (f * f * k) as f64 / (ms * 1e-3) / 1e9
        );
    });
}

/// Prints the best times of a hand reference and its achieved rate over `bytes`.
fn report_hand(b: &Bench, label: &str, bytes: f64, f: &mut dyn FnMut()) {
    let t = b.measure(f);
    println!(
        "{label}: gpu {:?} ms, wall {:.3} ms, {:.1} GB/s",
        t.gpu.map(|g| (g * 1e3).round() / 1e3),
        t.wall,
        bytes / (t.gate_ms() * 1e-3) / 1e9
    );
}

/// Checks the hand-written strided-axis references (ragged and block-aligned row counts) against
/// host f64 sums, then times them on `[64, 2^22]`.
fn sanity_axis(b: &mut Bench) {
    b.guarded("hand/axis", |b| {
        let ctx = b.ctx.clone();
        let cols = 64usize;
        let sumsq = hand::HandAxisSumSq::new(&ctx).unwrap();
        let var = hand::HandAxisVar::new(&ctx).unwrap();
        let hw = values(cols, 41, 0.5, 1.5);
        let w = Array::from_slice_on(&ctx, &hw, cols).unwrap();
        for rows in [5000usize, 1 << 16] {
            let hx = values(cols * rows, 40 + rows as u64, 2.0, 4.0);
            let x = Array::from_slice_on(&ctx, &hx, [cols, rows]).unwrap();
            let (mut s, mut q, mut m, mut v) = (vec![0.0f64; cols], vec![0.0f64; cols], vec![0.0f64; cols], vec![0.0f64; cols]);
            for r in 0..rows {
                for c in 0..cols {
                    let y = (hx[c + cols * r] * hw[c]) as f64;
                    s[c] += y;
                    q[c] += y * y;
                    m[c] += hx[c + cols * r] as f64;
                }
            }
            m.iter_mut().for_each(|t| *t /= rows as f64);
            for r in 0..rows {
                for c in 0..cols {
                    v[c] += (hx[c + cols * r] as f64 - m[c]).powi(2);
                }
            }
            let want_sq: Vec<f32> = s.iter().chain(&q).map(|&t| t as f32).collect();
            let want_var: Vec<f32> = v.iter().map(|&t| (t / rows as f64) as f32).collect();
            let mut out = Array::<f32>::zeros_on(&ctx, 2 * cols).unwrap();
            sumsq.run(&x, &w, &mut out, cols, rows);
            check("hand/axis_sum_sq", &out.to_vec().unwrap(), &want_sq, 1e-4, 1e-4);
            let mut out = Array::<f32>::zeros_on(&ctx, cols).unwrap();
            var.run(&x, &mut out, cols, rows);
            check("hand/axis_var", &out.to_vec().unwrap(), &want_var, 1e-4, 1e-6);
        }
        println!("hand axis_sum_sq ok");
        println!("hand axis_var ok");

        let rows = if b.quick { 1usize << 18 } else { 1 << 22 };
        let x = Array::from_slice_on(&ctx, &values(cols * rows, 42, 2.0, 4.0), [cols, rows]).unwrap();
        let bytes = 4.0 * (cols * rows) as f64;
        let mut out2 = Array::<f32>::zeros_on(&ctx, 2 * cols).unwrap();
        let mut out1 = Array::<f32>::zeros_on(&ctx, cols).unwrap();
        report_hand(b, &format!("hand/axis_sum_sq [64,{rows}]"), bytes, &mut || sumsq.run(&x, &w, &mut out2, cols, rows));
        report_hand(b, &format!("hand/axis_var [64,{rows}] (2 reads)"), 2.0 * bytes, &mut || var.run(&x, &mut out1, cols, rows));
    });
}

/// `x.histogram(bins, lo, hi)` against the privatised workgroup-atomic hand histogram over `2^26`
/// values (below `lo`, above `hi` and NaN included). Both sides are verified exactly against the
/// host bucket function first.
fn case_hist(b: &mut Bench) {
    for bins in [256usize, 4096] {
        b.guarded(&format!("hist/{bins}"), |b| {
            let ctx = b.ctx.clone();
            let n = if b.quick { 1usize << 20 } else { 1 << 26 };
            let (lo, hi) = (0.0f32, 1.0f32);
            let mut hx = values(n, 50, -0.25, 1.25);
            for i in (0..n).step_by(100_003) {
                hx[i] = f32::NAN;
            }
            let x = Array::from_slice_on(&ctx, &hx, n).unwrap();
            let scale = bins as f32 / (hi - lo);
            let mut want = vec![0u32; bins];
            for &v in &hx {
                let t = (v - lo) * scale;
                let u = if t.is_nan() { 0 } else { t.clamp(0.0, 4294967040.0) as u32 };
                want[u.min(bins as u32 - 1) as usize] += 1;
            }
            let zero = hand::HandZero::new(&ctx).unwrap();
            let hist = hand::HandHist::new(&ctx, bins).unwrap();
            let mut out = Array::<u32>::zeros_on(&ctx, bins).unwrap();

            let h = x.histogram(bins, lo, hi);
            h.eval().unwrap();
            check_exact(&format!("hist/{bins} luma"), &h.to_vec().unwrap(), &want);
            hist.run(&x, &mut out, n, lo, scale);
            check_exact(&format!("hist/{bins} hand"), &out.to_vec().unwrap(), &want);

            b.compare(
                &format!("hist/{bins}"),
                &n.to_string(),
                Work::Bytes(4.0 * n as f64),
                Some(DEFAULT_GATE),
                &mut || x.histogram(bins, lo, hi).eval().unwrap(),
                &mut || {
                    zero.run(&mut out, bins);
                    hist.run(&x, &mut out, n, lo, scale);
                },
            );
        });
    }
}

/// Host f64 scatter-add of `val` by `idx` (indices `>= len` skipped) and the per-bin `sum |v|`.
fn host_scatter(idx: &[u32], val: &[f32], len: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut sum, mut mag) = (vec![0.0f64; len], vec![0.0f64; len]);
    for (&j, &v) in idx.iter().zip(val) {
        if (j as usize) < len {
            sum[j as usize] += v as f64;
            mag[j as usize] += (v as f64).abs();
        }
    }
    (sum, mag)
}

/// `f32` sums within `1e-4` relative, or `1e-4 * sum|v|` absolute, of the f64 host sums.
fn check_sums(name: &str, got: &[f32], sum: &[f64], mag: &[f64]) {
    assert_eq!(got.len(), sum.len(), "{name}: length");
    for (i, g) in got.iter().enumerate() {
        let tol = 1e-4 * sum[i].abs().max(mag[i]);
        assert!((*g as f64 - sum[i]).abs() <= tol, "{name}: bin {i}: got {g}, want {} (tol {tol})", sum[i]);
    }
}

/// `put_add` of `2^24` `f32` values into `2^20` bins (a few indices out of range) against the hand
/// global CAS-loop kernel. Both sides are verified against host f64 sums first.
fn case_scatter_add(b: &mut Bench) {
    b.guarded("scatter_add/f32", |b| {
        let ctx = b.ctx.clone();
        let n = if b.quick { 1usize << 18 } else { 1 << 24 };
        let len = if b.quick { 1usize << 14 } else { 1 << 20 };
        let hi = codes(n, 60, (len + len / 16) as u32);
        let hv = values(n, 61, -1.0, 2.0);
        let (sum, mag) = host_scatter(&hi, &hv, len);
        let idx = Array::from_slice_on(&ctx, &hi, n).unwrap();
        let val = Array::from_slice_on(&ctx, &hv, n).unwrap();
        let zero = hand::HandZero::new(&ctx).unwrap();
        let hand = hand::HandScatterAddF32::new(&ctx).unwrap();
        let mut out = Array::<f32>::zeros_on(&ctx, len).unwrap();

        let mut t = Array::<f32>::zeros_on(&ctx, len).unwrap();
        t.put_add(&idx, &val).unwrap();
        t.eval().unwrap();
        check_sums("scatter_add/f32 luma", &t.to_vec().unwrap(), &sum, &mag);
        hand.run(&idx, &val, &mut out, n, len);
        check_sums("scatter_add/f32 hand", &out.to_vec().unwrap(), &sum, &mag);

        b.compare(
            "scatter_add/f32",
            &format!("{n}->{len}"),
            Work::Bytes(8.0 * n as f64),
            Some(DEFAULT_GATE),
            &mut || {
                let mut t = Array::<f32>::zeros_on(&ctx, len).unwrap();
                t.put_add(&idx, &val).unwrap();
                t.eval().unwrap();
            },
            &mut || {
                zero.run(&mut out, len);
                hand.run(&idx, &val, &mut out, n, len);
            },
        );
    });
}

/// `vals.group_sum(&codes, groups)` over `2^24` rows (a few codes past `groups`) against the hand
/// kernel (privatised for 16 and 1024 groups, global for `2^20`), both zero-filled each run.
/// Both sides are verified against host f64 sums first.
fn case_group_sum(b: &mut Bench) {
    b.guarded("group_sum/f32", |b| {
        let ctx = b.ctx.clone();
        let n = if b.quick { 1usize << 18 } else { 1 << 24 };
        let hv = values(n, 71, -1.0, 2.0);
        let val = Array::from_slice_on(&ctx, &hv, n).unwrap();
        let zero = hand::HandZero::new(&ctx).unwrap();
        for groups in [16usize, 1024, if b.quick { 1 << 14 } else { 1 << 20 }] {
            let hc = codes(n, 70 + groups as u64, (groups + groups / 16 + 1) as u32);
            let (sum, mag) = host_scatter(&hc, &hv, groups);
            let c = Array::from_slice_on(&ctx, &hc, n).unwrap();
            let hand = hand::HandGroupSum::new(&ctx, groups).unwrap();
            let mut out = Array::<f32>::zeros_on(&ctx, groups).unwrap();

            let g = val.group_sum(&c, groups);
            g.eval().unwrap();
            check_sums(&format!("group_sum/{groups} luma"), &g.to_vec().unwrap(), &sum, &mag);
            hand.run(&c, &val, &mut out, n);
            check_sums(&format!("group_sum/{groups} hand"), &out.to_vec().unwrap(), &sum, &mag);

            b.compare(
                &format!("group_sum/f32/{groups}"),
                &format!("{n}->{groups}"),
                Work::Bytes(8.0 * n as f64),
                Some(DEFAULT_GATE),
                &mut || val.group_sum(&c, groups).eval().unwrap(),
                &mut || {
                    zero.run(&mut out, groups);
                    hand.run(&c, &val, &mut out, n);
                },
            );
        }
    });
}

/// Entries `(row, column)` of an `m x n` result to verify: all of them if few, else a
/// pseudo-random sample (a host GEMM at 4096^3 is too slow).
fn sample_entries(m: usize, n: usize, seed: u64) -> Vec<(usize, usize)> {
    if m * n <= 1 << 16 {
        return (0..m).flat_map(|r| (0..n).map(move |c| (r, c))).collect();
    }
    let r = codes(512, seed, m as u32);
    let c = codes(512, seed + 1, n as u32);
    r.into_iter().zip(c).take(256).map(|(r, c)| (r as usize, c as usize)).collect()
}

/// Checks `got` (row-major `m x n`) against `A B` (row-major `m x k` and `k x n`) in f64 on the
/// sampled entries.
fn check_gemm(name: &str, got: &[f32], ha: &[f32], hb: &[f32], m: usize, n: usize, k: usize) {
    let entries = sample_entries(m, n, 77);
    let want: Vec<f32> =
        entries.iter().map(|&(r, c)| (0..k).map(|i| ha[r * k + i] as f64 * hb[i * n + c] as f64).sum::<f64>() as f32).collect();
    let have: Vec<f32> = entries.iter().map(|&(r, c)| got[r * n + c]).collect();
    check(name, &have, &want, 1e-3, 1e-3 * (k as f32).sqrt());
}

/// `matmul/square` (luma against the hand SGEMM at the same size) and `matmul/odd` (luma at an
/// odd size against the hand SGEMM at the next multiple of 128).
fn cmp_matmul(b: &mut Bench) {
    let hand = hand::HandSgemm::new(&b.ctx).unwrap();
    let square: &[usize] = if b.quick { &[256] } else { &[1024, 2048, 4096] };
    let odd: &[(usize, usize)] = if b.quick { &[(200, 256)] } else { &[(1000, 1024), (3000, 3072)] };
    for &n in square {
        if !b.want("matmul/square") {
            continue;
        }
        b.guarded("matmul/square", |b| {
            let ctx = b.ctx.clone();
            let (ha, hb) = (values(n * n, 41, -1.0, 1.0), values(n * n, 42, -1.0, 1.0));
            let a = Array::from_slice_on(&ctx, &ha, [n, n]).unwrap();
            let bm = Array::from_slice_on(&ctx, &hb, [n, n]).unwrap();
            let mut c = Array::<f32>::zeros_on(&ctx, n * n).unwrap();
            let r = a.matmul(&bm);
            check_gemm("matmul/square luma", &r.to_vec().unwrap(), &ha, &hb, n, n, n);
            hand.run(&a, &bm, &mut c, n, n, n);
            check_gemm("matmul/square hand", &c.to_vec().unwrap(), &ha, &hb, n, n, n);
            b.compare(
                "matmul/square",
                &n.to_string(),
                Work::Flops(2.0 * (n * n * n) as f64),
                Some(DEFAULT_GATE),
                &mut || {
                    a.matmul(&bm).eval().unwrap();
                },
                &mut || hand.run(&a, &bm, &mut c, n, n, n),
            );
        });
    }
    for &(n, padded) in odd {
        if !b.want("matmul/odd") {
            continue;
        }
        b.guarded("matmul/odd", |b| {
            let ctx = b.ctx.clone();
            let (ha, hb) = (values(n * n, 43, -1.0, 1.0), values(n * n, 44, -1.0, 1.0));
            let (pa, pb) = (values(padded * padded, 45, -1.0, 1.0), values(padded * padded, 46, -1.0, 1.0));
            let a = Array::from_slice_on(&ctx, &ha, [n, n]).unwrap();
            let bm = Array::from_slice_on(&ctx, &hb, [n, n]).unwrap();
            let ap = Array::from_slice_on(&ctx, &pa, [padded, padded]).unwrap();
            let bp = Array::from_slice_on(&ctx, &pb, [padded, padded]).unwrap();
            let mut c = Array::<f32>::zeros_on(&ctx, padded * padded).unwrap();
            let r = a.matmul(&bm);
            check_gemm("matmul/odd luma", &r.to_vec().unwrap(), &ha, &hb, n, n, n);
            hand.run(&ap, &bp, &mut c, padded, padded, padded);
            check_gemm("matmul/odd hand", &c.to_vec().unwrap(), &pa, &pb, padded, padded, padded);
            b.compare(
                "matmul/odd",
                &format!("{n} vs {padded}"),
                Work::Flops(2.0 * (n * n * n) as f64),
                Some(1.10),
                &mut || {
                    a.matmul(&bm).eval().unwrap();
                },
                &mut || hand.run(&ap, &bp, &mut c, padded, padded, padded),
            );
        });
    }
}

/// `matmul/gram_fused`: `Zᵀ Z` of the standardised `Z = (X - c) * s`, both operands pending.
fn cmp_gram(b: &mut Bench) {
    b.guarded("matmul/gram_fused", |b| {
        let ctx = b.ctx.clone();
        let f = 64usize;
        let k = if b.quick { 1usize << 14 } else { 1 << 20 };
        let hx = values(f * k, 51, -1.0, 1.0);
        let hc = values(f, 52, -0.5, 0.5);
        let hs = values(f, 53, 0.5, 2.0);
        let x = Array::from_slice_on(&ctx, &hx, [f, k]).unwrap();
        let c = Array::from_slice_on(&ctx, &hc, [f, 1]).unwrap();
        let s = Array::from_slice_on(&ctx, &hs, [f, 1]).unwrap();
        let (cv, sv) = (Array::from_slice_on(&ctx, &hc, f).unwrap(), Array::from_slice_on(&ctx, &hs, f).unwrap());
        let mut out = Array::<f32>::zeros_on(&ctx, f * f).unwrap();
        let hand = hand::HandGram::new(&ctx).unwrap();
        let z: Vec<f64> = (0..f * k).map(|i| ((hx[i] - hc[i % f]) * hs[i % f]) as f64).collect();
        let entries = sample_entries(f, f, 78);
        let want: Vec<f32> =
            entries.iter().map(|&(a, bb)| (0..k).map(|r| z[r * f + a] * z[r * f + bb]).sum::<f64>() as f32).collect();
        let pick = |got: &[f32]| entries.iter().map(|&(a, bb)| got[a * f + bb]).collect::<Vec<f32>>();
        let gram = || {
            let z = (&x - &c) * &s;
            z.transpose().matmul(&z)
        };
        let r = gram();
        assert_eq!(r.len(), f * f, "gram shape");
        let tol = 1e-3 * (k as f32).sqrt();
        check("matmul/gram_fused luma", &pick(&r.to_vec().unwrap()), &want, 1e-3, tol);
        hand.run(&x, &cv, &sv, &mut out, f as u32, k as u32);
        check("matmul/gram_fused hand", &pick(&out.to_vec().unwrap()), &want, 1e-3, tol);
        b.compare(
            "matmul/gram_fused",
            &format!("{f}x{k}"),
            Work::Flops(2.0 * (f * f * k) as f64),
            Some(DEFAULT_GATE),
            &mut || {
                gram().eval().unwrap();
            },
            &mut || hand.run(&x, &cv, &sv, &mut out, f as u32, k as u32),
        );
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

/// Radix sort: no hand-written competitor (the library kernels are the hand-written algorithm), so
/// the gate is the achieved bandwidth: bytes per pass are 12 per key (the count pass reads, the
/// scatter pass reads and writes) plus 8 per key with a payload (read and written), 4 passes.
fn case_sort(b: &mut Bench) {
    let n = if b.quick { 1 << 16 } else { 1 << 24 };
    let size = n.to_string();
    let passes = 4.0;
    // The fraction of the bandwidth is only meaningful for arrays large enough to hide the launches.
    let min = if b.quick { 0.0 } else { 0.5 };
    let ctx = b.ctx.clone();
    b.guarded("sort/u32", |b| {
        let host: Vec<u32> = codes(n, 21, u32::MAX).iter().zip(codes(n, 22, 2)).map(|(c, bit)| c ^ (bit << 31)).collect();
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mut want = host.clone();
        want.sort();
        check_exact("sort/u32", &x.sort().to_vec().unwrap(), &want);
        b.roofline("sort/u32", &size, passes * 12.0 * n as f64, min, &mut || {
            x.sort().eval().unwrap();
        });
    });
    b.guarded("sort/u32_payload", |b| {
        let host = codes(n, 23, u32::MAX);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| host[i as usize]);
        check_exact("sort/u32_payload", &x.argsort().to_vec().unwrap(), &order);
        b.roofline("sort/u32_payload", &size, passes * 20.0 * n as f64, min, &mut || {
            x.argsort().eval().unwrap();
        });
    });
    b.guarded("sort/f32", |b| {
        let host = values(n, 24, -1e6, 1e6);
        let x = Array::from_slice_on(&ctx, &host, n).unwrap();
        let mut want = host.clone();
        want.sort_by(f32::total_cmp);
        check_exact("sort/f32", &x.sort().to_vec().unwrap(), &want);
        b.roofline("sort/f32", &size, passes * 12.0 * n as f64, min, &mut || {
            x.sort().eval().unwrap();
        });
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
    sanity_gram(&mut b);
    sanity_axis(&mut b);
    case_hist(&mut b);
    case_scatter_add(&mut b);
    case_group_sum(&mut b);
    case_sort(&mut b);
    cmp_matmul(&mut b);
    cmp_gram(&mut b);
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
