//! The comparison cases. Each builds identical inputs for luma and ArrayFire, verifies the two
//! results against each other (and an `f64` / host reference where cheap) and then hands both
//! sides to the interleaved harness.
//!
//! Layout: luma and ArrayFire both have axis 0 as the fastest one, so an array of shape
//! `[a, b, c]` is the same memory in both. Matrices are row-major on the host: a `rows x cols`
//! matrix is `[cols, rows]` in either library, `luma::matmul(lhs[K, M], rhs[N, K]) -> [N, M]`
//! is the row-major `M x K` times `K x N`, and the same product in ArrayFire is
//! `matmul(rhs_af[N, K], lhs_af[K, M])`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use arrayfire as af;
use luma::Array as LA;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

pub type L = LA<f32>;
pub type A = af::Array<f32>;

pub fn human(n: usize) -> String {
    if n >= 1_000_000 && n % 100_000 == 0 {
        let m = n as f64 / 1e6;
        if m.fract() == 0.0 { format!("{}M", m as u64) } else { format!("{m:.1}M") }
    } else if n >= 1024 && n % 1024 == 0 && n.is_power_of_two() {
        format!("{}", n)
    } else {
        format!("{n}")
    }
}

/// Runs `f`, returning `None` if it panicked (an ArrayFire error).
pub fn attempt<R>(f: impl FnOnce() -> R) -> Option<R> {
    let r = catch_unwind(AssertUnwindSafe(f));
    if r.is_err() {
        let _ = catch_unwind(|| afx::sync());
    }
    r.ok()
}

// ------------------------------------------------------------------------------------------
// 1, 2. Elementwise chains
// ------------------------------------------------------------------------------------------

/// `n` fused operations on `x`: mul, add, sin, mul, sub, abs, sqrt, add `x`, repeated.
pub fn chain_l(x: &L, n: usize) -> L {
    let mut y = x.clone();
    for i in 0..n {
        y = match i % 8 {
            0 => &y * 1.01f32,
            1 => &y + 0.1f32,
            2 => y.sin(),
            3 => &y * 0.9f32,
            4 => &y - 0.05f32,
            5 => y.abs(),
            6 => y.sqrt(),
            _ => &y + x,
        };
    }
    y
}

pub fn chain_a(x: &A, n: usize) -> A {
    let mut y = x.clone();
    for i in 0..n {
        y = match i % 8 {
            0 => &y * 1.01f32,
            1 => &y + 0.1f32,
            2 => af::sin(&y),
            3 => &y * 0.9f32,
            4 => &y - 0.05f32,
            5 => af::abs(&y),
            6 => af::sqrt(&y),
            _ => &y + x,
        };
    }
    y
}

/// The same chain with every operation evaluated (one launch each).
pub fn chain_each_l(x: &L, n: usize) -> L {
    let mut y = x.clone();
    for i in 0..n {
        y = match i % 8 {
            0 => &y * 1.01f32,
            1 => &y + 0.1f32,
            2 => y.sin(),
            3 => &y * 0.9f32,
            4 => &y - 0.05f32,
            5 => y.abs(),
            6 => y.sqrt(),
            _ => &y + x,
        };
        y.eval().unwrap();
    }
    y
}

pub fn chain_each_a(x: &A, n: usize) -> A {
    let mut y = x.clone();
    for i in 0..n {
        y = match i % 8 {
            0 => &y * 1.01f32,
            1 => &y + 0.1f32,
            2 => af::sin(&y),
            3 => &y * 0.9f32,
            4 => &y - 0.05f32,
            5 => af::abs(&y),
            6 => af::sqrt(&y),
            _ => &y + x,
        };
        y.eval();
    }
    y
}

fn ew_case(b: &mut Bench, name: &str, n: usize, ops: usize, note: &str) {
    let data = uniform(n, 1, 0.0, 1.0);
    let (xl, xa) = (lumax::arr(&b.ctx, &data, n), afx::arr(&data, &[n as u64]));
    let v = compare(&lumax::host(&chain_l(&xl, ops)), &afx::host(&chain_a(&xa, ops)), 1e-4, 1e-5);
    let ctx = b.ctx.clone();
    b.run(
        name,
        &human(n),
        v,
        note,
        &mut || {
            chain_l(&xl, ops).eval().unwrap();
            ctx.sync().unwrap();
        },
        &mut || {
            chain_a(&xa, ops).eval();
            afx::sync();
        },
    );
}

pub fn elementwise(b: &mut Bench) {
    b.guarded("ew", |b| {
        for n in [1_000_000, 16_000_000, 64_000_000] {
            ew_case(b, "1 ew chain, 8 fused ops", n, 8, "");
        }
        ew_case(b, "1 ew chain, 32 fused ops", 16_000_000, 32, "");
    });
}

pub fn small_ops(b: &mut Bench) {
    b.guarded("small", |b| {
        let n = 4096;
        let data = uniform(n, 2, 0.0, 1.0);
        let (xl, xa) = (lumax::arr(&b.ctx, &data, n), afx::arr(&data, &[n as u64]));
        let v = compare(&lumax::host(&chain_l(&xl, 64)), &afx::host(&chain_a(&xa, 64)), 1e-4, 1e-5);
        let ctx = b.ctx.clone();
        b.run(
            "2 launch overhead: 64 tiny ops, one eval",
            "4096",
            v,
            "chain built and evaluated every call; includes host-side graph building",
            &mut || {
                chain_l(&xl, 64).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                chain_a(&xa, 64).eval();
                afx::sync();
            },
        );
        let v = compare(&lumax::host(&chain_each_l(&xl, 64)), &afx::host(&chain_each_a(&xa, 64)), 1e-4, 1e-5);
        b.run(
            "2b launch overhead: 64 tiny ops, eval each op",
            "4096",
            v,
            "extra: one eval per op on both sides; AF launches each, luma postpones and fuses evals while the GPU is busy (documented Array::eval behaviour)",
            &mut || {
                // luma never computes arrays that are dropped before the GPU gets to them (see Array::eval),
                // so the result is held until after the sync, as a program that uses it would.
                let r = chain_each_l(&xl, 64);
                ctx.sync().unwrap();
                drop(r);
            },
            &mut || {
                chain_each_a(&xa, 64);
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 3. Reductions
// ------------------------------------------------------------------------------------------

pub fn reductions(b: &mut Bench) {
    b.guarded("reduce", |b| {
        let ctx = b.ctx.clone();
        for n in [1_000_000usize, 100_000_000] {
            let data = uniform(n, 3, 0.0, 1.0);
            let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
            let sum_ref = data.iter().map(|&v| f64::from(v)).sum::<f64>();
            let max_ref = data.iter().copied().fold(f32::MIN, f32::max);
            drop(data);
            let (ls, asum) = (lumax::host(&xl.sum()), afx::host(&af::sum(&xa, 0)));
            let v = compare(&ls, &asum, 1e-4, 0.0)
                .and(compare_ref("luma", &ls, &[sum_ref], 1e-4, 0.0))
                .and(compare_ref("AF", &asum, &[sum_ref], 1e-4, 0.0));
            b.run(
                "3 reduce sum",
                &human(n),
                v,
                "device-side 1-element result, no host copy",
                &mut || {
                    xl.sum().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::sum(&xa, 0).eval();
                    afx::sync();
                },
            );
            let (lm, am) = (lumax::host(&xl.max()), afx::host(&af::max(&xa, 0)));
            let v = exact(&lm, &am).and(exact(&lm, &[max_ref]));
            b.run(
                "3 reduce max",
                &human(n),
                v,
                "",
                &mut || {
                    xl.max().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::max(&xa, 0).eval();
                    afx::sync();
                },
            );
        }
        let n = 4096usize;
        let data = uniform(n * n, 4, 0.0, 1.0);
        let (xl, xa) = (lumax::arr(&ctx, &data, [n, n]), afx::arr(&data, &[n as u64, n as u64]));
        for axis in 0..2usize {
            let (l, a) = (lumax::host(&xl.sum_axis(axis)), afx::host(&af::sum(&xa, axis as i32)));
            let mut want = vec![0.0f64; n];
            for j in 0..n {
                for i in 0..n {
                    want[if axis == 0 { j } else { i }] += f64::from(data[i + n * j]);
                }
            }
            let v = compare(&l, &a, 1e-4, 0.0).and(compare_ref("luma", &l, &want, 1e-4, 0.0));
            let what = if axis == 0 { "axis 0 (contiguous lines)" } else { "axis 1 (strided)" };
            b.run(
                &format!("3 reduce sum along {what}"),
                "4096 x 4096",
                v,
                "",
                &mut || {
                    xl.sum_axis(axis).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::sum(&xa, axis as i32).eval();
                    afx::sync();
                },
            );
        }
        // describe-like statistics of 32 features over 1M samples each.
        let (rows, cols) = (1_000_000usize, 32usize);
        let data = uniform(rows * cols, 5, 0.0, 1.0);
        let (xl, xa) = (lumax::arr(&ctx, &data, [rows, cols]), afx::arr(&data, &[rows as u64, cols as u64]));
        let stats_l = |x: &L| (x.mean_axis(0), x.var_axis(0));
        let stats_a = |x: &A| (af::mean(x, 0), af::var_v2(x, af::VarianceBias::POPULATION, 0));
        let (lm, lv) = stats_l(&xl);
        let (am, av) = stats_a(&xa);
        let v = compare(&lumax::host(&lm), &afx::host(&am), 1e-4, 0.0).and(compare(&lumax::host(&lv), &afx::host(&av), 1e-3, 0.0));
        b.run(
            "3 mean + var (describe-like)",
            "1M x 32 (per feature)",
            v,
            "population variance; both results evaluated together",
            &mut || {
                let (m, v) = stats_l(&xl);
                luma::eval_all(&[&m, &v]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (m, v) = stats_a(&xa);
                af::eval_multiple(vec![&m, &v]);
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 4, 5. Matrix products
// ------------------------------------------------------------------------------------------

/// Entry `(i, j)` of the row-major product of `a` (`m x k`) and `b` (`k x n`), in `f64`.
fn dot_ref(a: &[f32], b: &[f32], k: usize, n: usize, i: usize, j: usize) -> f64 {
    (0..k).map(|t| f64::from(a[i * k + t]) * f64::from(b[t * n + j])).sum()
}

/// A few entries of `out` (row-major `m x n`) against the `f64` product.
fn sample_check(who: &str, out: &[f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, batch: usize, abs: f64) -> Verdict {
    let mut r = Rng::new(99);
    for s in 0..24 {
        let bt = if batch > 1 { (r.next_u64() % batch as u64) as usize } else { 0 };
        let (i, j) = ((r.next_u64() % m as u64) as usize, (r.next_u64() % n as u64) as usize);
        let want = dot_ref(&a[bt * m * k..], &b[bt * k * n..], k, n, i, j);
        let got = f64::from(out[bt * m * n + i * n + j]);
        if (got - want).abs() > abs {
            return Verdict::no(format!("NO: {who} entry {s} ({bt},{i},{j}): {got} vs f64 {want} (tol {abs:.1e})"));
        }
    }
    Verdict::yes(format!("{who} within {abs:.0e} of f64 on 24 samples"))
}

type MmL = fn(&L, &L) -> L;

/// Whether ArrayFire's cuBLAS can be switched to TF32: the math mode that makes `sgemm` differ
/// from the default one by TF32-sized errors.
pub fn probe_af_tf32() -> (Option<std::ffi::c_int>, String) {
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let a = uniform(m * k, 71, -1.0, 1.0);
    let bm = uniform(k * n, 72, -1.0, 1.0);
    let (aa, ba) = (afx::arr(&a, &[k as u64, m as u64]), afx::arr(&bm, &[n as u64, k as u64]));
    let mm = || {
        let c = af::matmul(&ba, &aa, af::MatProp::NONE, af::MatProp::NONE);
        c.eval();
        afx::sync();
        afx::host(&c)
    };
    let _ = afx::set_math_mode(afx::CUBLAS_DEFAULT_MATH);
    let base = mm();
    let r = rms(&base);
    let mut log = String::new();
    let mut found = None;
    for mode in [afx::CUBLAS_TF32_TENSOR_OP_MATH, afx::CUBLAS_TENSOR_OP_MATH] {
        match afx::set_math_mode(mode) {
            Err(e) => log += &format!("mode {mode}: {e}; "),
            Ok(()) => {
                let c = mm();
                let d = (rms(&c.iter().zip(&base).map(|(x, y)| x - y).collect::<Vec<_>>())) / r;
                log += &format!("mode {mode}: rms deviation from default sgemm {d:.2e}; ");
                if d > 5e-5 && found.is_none() {
                    found = Some(mode);
                }
            }
        }
    }
    let _ = afx::set_math_mode(afx::CUBLAS_DEFAULT_MATH);
    (found, log)
}

fn matmul_size(b: &mut Bench, label: &str, m: usize, k: usize, n: usize, af_tf32: Option<std::ffi::c_int>) {
    let ctx = b.ctx.clone();
    let a = uniform(m * k, 11, -1.0, 1.0);
    let bm = uniform(k * n, 12, -1.0, 1.0);
    let (al, bl) = (lumax::arr(&ctx, &a, [k, m]), lumax::arr(&ctx, &bm, [n, k]));
    let (aa, ba) = (afx::arr(&a, &[k as u64, m as u64]), afx::arr(&bm, &[n as u64, k as u64]));
    let size = format!("{m} x {k} x {n}");
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let af_mm = |ba: &A, aa: &A| af::matmul(ba, aa, af::MatProp::NONE, af::MatProp::NONE);
    for tf32 in [false, true] {
        let (lf, name): (MmL, &str) = if tf32 { (luma::matmul_tf32, "4 matmul TF32") } else { (luma::matmul_f32, "4 matmul f32") };
        let mode = if tf32 { af_tf32.unwrap_or(afx::CUBLAS_DEFAULT_MATH) } else { afx::CUBLAS_DEFAULT_MATH };
        afx::set_math_mode(mode).expect("set cuBLAS math mode");
        let lout = lumax::host(&lf(&al, &bl));
        let aout = afx::host(&af_mm(&ba, &aa));
        let r = rms(&aout);
        let (rel, what) = if tf32 { (3e-3, "TF32") } else { (5e-5, "f32") };
        let mut v = compare(&lout, &aout, 0.0, rel * r);
        v = v.and(sample_check("luma", &lout, &a, &bm, m, k, n, 1, rel * r)).and(sample_check("AF", &aout, &a, &bm, m, k, n, 1, rel * r));
        let note = if tf32 && af_tf32.is_none() {
            "DIFFERENT PRECISION: luma TF32 vs ArrayFire f32 (no TF32 mode took effect in ArrayFire)".to_string()
        } else if tf32 {
            format!("both TF32 (AF cuBLAS math mode {}); {what} tolerance {rel:.0e}*rms", af_tf32.unwrap())
        } else {
            "luma matmul_f32 (SIMT f32) vs cuBLAS SGEMM".to_string()
        };
        let case = format!("{name} [{label}]");
        // Report throughput in the note after timing.
        b.run(
            &case,
            &size,
            v,
            &note,
            &mut || {
                lf(&al, &bl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_mm(&ba, &aa).eval();
                afx::sync();
            },
        );
        let row = b.rows.last_mut().unwrap();
        let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;
        row.note += &format!("; {:.1} TFLOP/s luma, {:.1} AF", tf(row.luma_ms.unwrap()), tf(row.af_ms.unwrap()));
    }
    afx::set_math_mode(afx::CUBLAS_DEFAULT_MATH).expect("reset cuBLAS math mode");
}

pub fn matmuls(b: &mut Bench) {
    b.guarded("matmul", |b| {
        let (tf, log) = probe_af_tf32();
        println!("ArrayFire TF32 probe: {log}-> {}", tf.map_or("no TF32 mode".into(), |m| format!("using cuBLAS math mode {m}")));
        let mut sizes: Vec<(&str, usize, usize, usize)> = vec![
            ("square 1024", 1024, 1024, 1024),
            ("square 4096", 4096, 4096, 4096),
            ("square 8192", 8192, 8192, 8192),
            ("dense", 256, 1024, 1024),
            ("tall", 2048, 512, 128),
            ("attention P.V", 2048, 2048, 64),
            ("attention Q.K^T", 2048, 64, 2048),
        ];
        if b.quick {
            sizes.retain(|s| s.1 <= 4096);
        }
        for (label, m, k, n) in sizes {
            matmul_size(b, label, m, k, n, tf);
        }
    });
}

pub fn batched(b: &mut Bench) {
    b.guarded("bmm", |b| {
        let ctx = b.ctx.clone();
        let (bt, m, k, n) = (4096usize, 32usize, 32usize, 32usize);
        let a = uniform(bt * m * k, 21, -1.0, 1.0);
        let bm = uniform(bt * k * n, 22, -1.0, 1.0);
        let (al, bl) = (lumax::arr(&ctx, &a, [k, m, bt]), lumax::arr(&ctx, &bm, [n, k, bt]));
        let (aa, ba) = (afx::arr(&a, &[k as u64, m as u64, bt as u64]), afx::arr(&bm, &[n as u64, k as u64, bt as u64]));
        let af_mm = |ba: &A, aa: &A| af::matmul(ba, aa, af::MatProp::NONE, af::MatProp::NONE);
        let lout = lumax::host(&luma::matmul_batched(&al, &bl));
        let aout = afx::host(&af_mm(&ba, &aa));
        let tol = 5e-5 * rms(&aout);
        let v = compare(&lout, &aout, 0.0, tol).and(sample_check("luma", &lout, &a, &bm, m, k, n, bt, tol)).and(sample_check("AF", &aout, &a, &bm, m, k, n, bt, tol));
        b.run(
            "5 batched small matmul f32",
            "4096 x (32x32 @ 32x32)",
            v,
            "luma matmul_batched (always f32); AF matmul with a batch dimension (cublasSgemmBatched)",
            &mut || {
                luma::matmul_batched(&al, &bl).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af_mm(&ba, &aa).eval();
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 6. FFT
// ------------------------------------------------------------------------------------------

pub fn fft(b: &mut Bench) {
    b.guarded("fft", |b| {
        let ctx = b.ctx.clone();
        // FFT convolution (`conv1d_fft`, built on luma's convolution FFT passes):
        // compare that with ArrayFire's `fft_convolve1` (cuFFT), full linear convolution in both.
        for (n, k) in [(1_000_000usize, 4096usize), (16_000_000, 4096), (1_000_000, 1_000_000)] {
            let x = uniform(n, 31, -1.0, 1.0);
            let kern = uniform(k, 32, -1.0, 1.0);
            let (xl, kl) = (lumax::arr(&ctx, &x, n), lumax::arr(&ctx, &kern, k));
            let (xa, ka) = (afx::arr(&x, &[n as u64]), afx::arr(&kern, &[k as u64]));
            let lout = lumax::host(&xl.conv1d_fft(&kl));
            let aout = afx::host(&af::fft_convolve1(&xa, &ka, af::ConvMode::EXPAND));
            let tol = 1e-3 * rms(&aout);
            let mut v = compare(&lout, &aout, 0.0, tol);
            // A few output samples against the direct sum.
            let want: Vec<f64> = [0usize, 1, k / 2, k - 1, n / 3, n - 1, n + k - 2]
                .iter()
                .map(|&j| (0..k).filter(|&t| j >= t && j - t < n).map(|t| f64::from(kern[t]) * f64::from(x[j - t])).sum())
                .collect();
            let idx = [0usize, 1, k / 2, k - 1, n / 3, n - 1, n + k - 2];
            let pick = |o: &[f32]| idx.iter().map(|&i| o[i]).collect::<Vec<_>>();
            v = v.and(compare_ref("luma", &pick(&lout), &want, 0.0, tol)).and(compare_ref("AF", &pick(&aout), &want, 0.0, tol));
            b.run(
                "6 FFT convolution (real)",
                &format!("signal {} (x) kernel {}", human(n), human(k)),
                v,
                "luma conv1d_fft vs AF fft_convolve1 (cuFFT); luma has no standalone FFT API",
                &mut || {
                    xl.conv1d_fft(&kl).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::fft_convolve1(&xa, &ka, af::ConvMode::EXPAND).eval();
                    afx::sync();
                },
            );
        }
        // Complex and real transforms: luma's own FFT (`fft`, `ifft`, `rfft`, `fft2`, `fft3`) against cuFFT through ArrayFire.
        let cplx = |n: usize, seed: u64| -> Vec<(f32, f32)> {
            let (re, im) = (uniform(n, seed, -1.0, 1.0), uniform(n, seed + 1, -1.0, 1.0));
            re.iter().zip(&im).map(|(&r, &i)| (r, i)).collect()
        };
        let flat_l = |v: &[luma::C32]| -> Vec<f32> { v.iter().flat_map(|c| [c.re, c.im]).collect() };
        let flat_a = |v: &[af::c32]| -> Vec<f32> { v.iter().flat_map(|c| [c.re, c.im]).collect() };
        let agree = |l: &[luma::C32], a: &[af::c32]| {
            let (l, a) = (flat_l(l), flat_a(a));
            compare(&l, &a, 0.0, 5e-5 * rms(&a))
        };
        for n in [1_000_000usize, 16_000_000, 1_000_003, 1 << 20, 1 << 24] {
            let c = cplx(n, 33);
            let lc: Vec<luma::C32> = c.iter().map(|&(r, i)| luma::C32::new(r, i)).collect();
            let ac: Vec<af::c32> = c.iter().map(|&(r, i)| af::c32::new(r, i)).collect();
            let (xl, xa) = (lumax::arr(&ctx, &lc, n), afx::arr(&ac, &[n as u64]));
            let v = agree(&lumax::host(&xl.fft()), &afx::host(&af::fft(&xa, 1.0, 0)));
            b.run(
                "6 FFT 1D complex c2c (forward)",
                &human(n),
                v,
                "luma fft vs AF fft (cuFFT)",
                &mut || {
                    xl.fft().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::fft(&xa, 1.0, 0).eval();
                    afx::sync();
                },
            );
            let v = agree(&lumax::host(&xl.ifft()), &afx::host(&af::ifft(&xa, 1.0 / n as f64, 0)));
            b.run(
                "6 FFT 1D complex c2c (inverse)",
                &human(n),
                v,
                "luma ifft vs AF ifft (cuFFT)",
                &mut || {
                    xl.ifft().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::ifft(&xa, 1.0 / n as f64, 0).eval();
                    afx::sync();
                },
            );
        }
        for n in [1_000_000usize, 16_000_000, 1_000_003, 1 << 20, 1 << 24] {
            let r = uniform(n, 37, -1.0, 1.0);
            let (xl, xa) = (lumax::arr(&ctx, &r, n), afx::arr(&r, &[n as u64]));
            let v = agree(&lumax::host(&xl.rfft()), &afx::host(&af::fft_r2c(&xa, 1.0, 0)));
            b.run(
                "6 FFT 1D real-to-complex",
                &human(n),
                v,
                "luma rfft vs AF fft_r2c (cuFFT R2C)",
                &mut || {
                    xl.rfft().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::fft_r2c(&xa, 1.0, 0).eval();
                    afx::sync();
                },
            );
        }
        let n = 2048usize;
        let c = cplx(n * n, 35);
        let lc: Vec<luma::C32> = c.iter().map(|&(r, i)| luma::C32::new(r, i)).collect();
        let ac: Vec<af::c32> = c.iter().map(|&(r, i)| af::c32::new(r, i)).collect();
        let (xl, xa) = (lumax::arr(&ctx, &lc, [n, n]), afx::arr(&ac, &[n as u64, n as u64]));
        let v = agree(&lumax::host(&xl.fft2()), &afx::host(&af::fft2(&xa, 1.0, 0, 0)));
        b.run("6 FFT 2D complex c2c (forward)", "2048 x 2048", v, "luma fft2 vs AF fft2", &mut || { xl.fft2().eval().unwrap(); ctx.sync().unwrap(); }, &mut || { af::fft2(&xa, 1.0, 0, 0).eval(); afx::sync(); });
        let v = agree(&lumax::host(&xl.ifft2()), &afx::host(&af::ifft2(&xa, 1.0 / (n * n) as f64, 0, 0)));
        b.run("6 FFT 2D complex c2c (inverse)", "2048 x 2048", v, "luma ifft2 vs AF ifft2", &mut || { xl.ifft2().eval().unwrap(); ctx.sync().unwrap(); }, &mut || { af::ifft2(&xa, 1.0 / (n * n) as f64, 0, 0).eval(); afx::sync(); });
        let n = 256usize;
        let c = cplx(n * n * n, 39);
        let lc: Vec<luma::C32> = c.iter().map(|&(r, i)| luma::C32::new(r, i)).collect();
        let ac: Vec<af::c32> = c.iter().map(|&(r, i)| af::c32::new(r, i)).collect();
        let (xl, xa) = (lumax::arr(&ctx, &lc, [n, n, n]), afx::arr(&ac, &[n as u64, n as u64, n as u64]));
        let v = agree(&lumax::host(&xl.fft3()), &afx::host(&af::fft3(&xa, 1.0, 0, 0, 0)));
        b.run("6 FFT 3D complex c2c (forward)", "256^3", v, "luma fft3 vs AF fft3", &mut || { xl.fft3().eval().unwrap(); ctx.sync().unwrap(); }, &mut || { af::fft3(&xa, 1.0, 0, 0, 0).eval(); afx::sync(); });
        // FFT convolution in two and three dimensions (same-size output, odd kernels).
        for (sd, kd) in [([2048usize, 2048, 1], [15usize, 15, 1]), ([4096, 4096, 1], [31, 31, 1]), ([256, 256, 256], [9, 9, 9])] {
            let three = sd[2] > 1;
            let (sl, kl) = (sd.iter().product::<usize>(), kd.iter().product::<usize>());
            let x = uniform(sl, 51, -1.0, 1.0);
            let kern = uniform(kl, 52, -1.0, 1.0);
            let ldims: Vec<usize> = if three { sd.to_vec() } else { sd[..2].to_vec() };
            let kdims: Vec<usize> = if three { kd.to_vec() } else { kd[..2].to_vec() };
            let shape = |d: &[usize]| luma::Shape::from([d[0], d[1], *d.get(2).unwrap_or(&1), 1]);
            let (xl, kl_) = (lumax::arr(&ctx, &x, shape(&ldims)), lumax::arr(&ctx, &kern, shape(&kdims)));
            let adims = |d: &[usize]| d.iter().map(|&v| v as u64).collect::<Vec<_>>();
            let (xa, ka) = (afx::arr(&x, &adims(&ldims)), afx::arr(&kern, &adims(&kdims)));
            let luma_conv = |x: &luma::Array<f32>, k: &luma::Array<f32>| if three { x.fft_convolve3(k, luma::ConvMode::Same) } else { x.fft_convolve2(k, luma::ConvMode::Same) };
            let af_conv = |x: &af::Array<f32>, k: &af::Array<f32>| if three { af::fft_convolve3(x, k, af::ConvMode::DEFAULT) } else { af::fft_convolve2(x, k, af::ConvMode::DEFAULT) };
            let (lo, ao) = (lumax::host(&luma_conv(&xl, &kl_)), afx::host(&af_conv(&xa, &ka)));
            let v = compare(&lo, &ao, 0.0, 2e-4 * rms(&ao));
            let size = format!("{} (x) {}", ldims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"), kdims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"));
            b.run(
                if three { "6 FFT convolution 3D (real)" } else { "6 FFT convolution 2D (real)" },
                &size,
                v,
                "luma fft_convolve (same) vs AF fft_convolve (cuFFT)",
                &mut || {
                    luma_conv(&xl, &kl_).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af_conv(&xa, &ka).eval();
                    afx::sync();
                },
            );
        }
    });
}

// ------------------------------------------------------------------------------------------
// 7. Sort
// ------------------------------------------------------------------------------------------

/// `idx` is a permutation of `0..n` that puts `data` in non-decreasing order, giving `sorted`.
fn argsort_ok(who: &str, data: &[f32], idx: &[u32], sorted: Option<&[f32]>) -> Verdict {
    let n = data.len();
    if idx.len() != n {
        return Verdict::no(format!("NO: {who} argsort length {} vs {n}", idx.len()));
    }
    let mut seen = vec![false; n];
    let mut prev = f32::NEG_INFINITY;
    for (p, &i) in idx.iter().enumerate() {
        let i = i as usize;
        if i >= n || std::mem::replace(&mut seen[i], true) {
            return Verdict::no(format!("NO: {who} argsort is not a permutation (position {p}, index {i})"));
        }
        let v = data[i];
        if v < prev || sorted.is_some_and(|s| s[p] != v) {
            return Verdict::no(format!("NO: {who} argsort does not order the data at position {p}"));
        }
        prev = v;
    }
    Verdict::yes(format!("{who} valid permutation"))
}

pub fn sorts(b: &mut Bench) {
    b.guarded("sort", |b| {
        let ctx = b.ctx.clone();
        for n in [10_000_000usize, 100_000_000] {
            let data = uniform(n, 41, 0.0, 1.0);
            let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
            let (lo, ao) = (lumax::host(&xl.sort()), afx::host(&af::sort(&xa, 0, true)));
            let mut v = exact(&lo, &ao);
            if n <= 10_000_000 {
                let mut want = data.clone();
                want.sort_by(f32::total_cmp);
                v = v.and(exact(&lo, &want));
            } else {
                v = v.and(if is_sorted(&lo) { Verdict::yes("sorted") } else { Verdict::no("NO: luma result not sorted") });
            }
            drop((lo, ao));
            b.run(
                "7 sort f32",
                &human(n),
                v,
                "",
                &mut || {
                    xl.sort().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::sort(&xa, 0, true).eval();
                    afx::sync();
                },
            );
            if n == 10_000_000 {
                // argsort: luma returns the indices only, ArrayFire's sort_index also the sorted values.
                let (li, (av, ai)) = (lumax::host(&xl.argsort()), af::sort_index(&xa, 0, true));
                let (av, ai) = (afx::host(&av), afx::host(&ai));
                let v = argsort_ok("luma", &data, &li, None).and(argsort_ok("AF", &data, &ai, Some(&av)));
                b.run(
                    "7 argsort f32",
                    &human(n),
                    v,
                    "luma argsort returns indices; AF sort_index also materialises the sorted values",
                    &mut || {
                        xl.argsort().eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let (v, i) = af::sort_index(&xa, 0, true);
                        v.eval();
                        i.eval();
                        afx::sync();
                    },
                );
                // sort_by_key: values are the original positions (exact in f32 below 2^24).
                let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let (vl, va) = (lumax::arr(&ctx, &vals, n), afx::arr(&vals, &[n as u64]));
                let by_key_l = || {
                    let idx = xl.argsort();
                    (xl.take(&idx), vl.take(&idx))
                };
                let (lk, lv) = by_key_l();
                let (ak, av) = af::sort_by_key(&xa, &va, 0, true);
                let (lk, lv, ak, av) = (lumax::host(&lk), lumax::host(&lv), afx::host(&ak), afx::host(&av));
                let check = |who: &str, k: &[f32], v: &[f32]| {
                    if v.iter().enumerate().all(|(p, &o)| data[o as usize] == k[p]) { Verdict::yes(format!("{who} values follow keys")) } else { Verdict::no(format!("NO: {who} values do not follow keys")) }
                };
                let v = exact(&lk, &ak).and(check("luma", &lk, &lv)).and(check("AF", &ak, &av));
                b.run(
                    "7 sort_by_key f32 keys, f32 values",
                    &human(n),
                    v,
                    "luma has no sort_by_key: argsort + two takes (gathers); AF sort_by_key",
                    &mut || {
                        let (k, v) = by_key_l();
                        luma::eval_all(&[&k, &v]).unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let (k, v) = af::sort_by_key(&xa, &va, 0, true);
                        k.eval();
                        v.eval();
                        afx::sync();
                    },
                );
            }
        }
        // 1M rows of 64: sort every row (axis 0 is the contiguous one in both libraries).
        let (len, rows) = (64usize, 1_000_000usize);
        let data = uniform(len * rows, 42, 0.0, 1.0);
        let (xl, xa) = (lumax::arr(&ctx, &data, [len, rows]), afx::arr(&data, &[len as u64, rows as u64]));
        let (lo, ao) = (lumax::host(&xl.sort_axis(0)), afx::host(&af::sort(&xa, 0, true)));
        let mut want = data.clone();
        want.chunks_mut(len).for_each(|c| c.sort_by(f32::total_cmp));
        let v = exact(&lo, &ao).and(exact(&lo, &want));
        drop((lo, ao, want));
        b.run(
            "7 sort along axis (each row)",
            "1M rows x 64",
            v,
            "",
            &mut || {
                xl.sort_axis(0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::sort(&xa, 0, true).eval();
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 8. Scan
// ------------------------------------------------------------------------------------------

pub fn scans(b: &mut Bench) {
    b.guarded("scan", |b| {
        let ctx = b.ctx.clone();
        // +-1 data: every partial sum is an exact f32 integer, whatever the summation order.
        let n = 100_000_000usize;
        let data = signs(n, 51);
        let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
        let (lo, ao) = (lumax::host(&xl.cumsum(0)), afx::host(&af::accum(&xa, 0)));
        let mut acc = 0.0f64;
        let want: Vec<f32> = data.iter().map(|&v| { acc += f64::from(v); acc as f32 }).collect();
        let v = exact(&lo, &ao).and(exact(&lo, &want));
        drop((lo, ao, want, data));
        b.run(
            "8 cumsum f32",
            &human(n),
            v,
            "+-1 input (exact sums)",
            &mut || {
                xl.cumsum(0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::accum(&xa, 0).eval();
                afx::sync();
            },
        );
        drop((xl, xa));
        let n = 4096usize;
        let data = signs(n * n, 52);
        let (xl, xa) = (lumax::arr(&ctx, &data, [n, n]), afx::arr(&data, &[n as u64, n as u64]));
        for axis in 0..2usize {
            let (lo, ao) = (lumax::host(&xl.cumsum(axis)), afx::host(&af::accum(&xa, axis as i32)));
            let mut want = data.clone();
            for j in 0..n {
                for i in 0..n {
                    let (idx, prev) = if axis == 0 { (i + n * j, if i > 0 { i - 1 + n * j } else { usize::MAX }) } else { (i + n * j, if j > 0 { i + n * (j - 1) } else { usize::MAX }) };
                    if prev != usize::MAX {
                        want[idx] += want[prev];
                    }
                }
            }
            let v = exact(&lo, &ao).and(exact(&lo, &want));
            let what = if axis == 0 { "axis 0 (contiguous lines)" } else { "axis 1 (strided)" };
            b.run(
                &format!("8 cumsum along {what}"),
                "4096 x 4096",
                v,
                "+-1 input (exact sums)",
                &mut || {
                    xl.cumsum(axis).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::accum(&xa, axis as i32).eval();
                    afx::sync();
                },
            );
        }
    });
}

// ------------------------------------------------------------------------------------------
// 9, 10. Histogram, gather, scatter
// ------------------------------------------------------------------------------------------

fn counts_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&c| c as f32).collect()
}

pub fn histograms(b: &mut Bench) {
    b.guarded("hist", |b| {
        let ctx = b.ctx.clone();
        let n = 10_000_000usize;
        let data = uniform(n, 61, 0.0, 1.0);
        let (xl, xa) = (lumax::arr(&ctx, &data, n), afx::arr(&data, &[n as u64]));
        let lh = counts_f32(&lumax::host(&xl.histogram(256, 0.0, 1.0)));
        let ah = counts_f32(&afx::host(&af::histogram(&xa, 256, 0.0, 1.0)));
        let mut want = vec![0.0f64; 256];
        for &x in &data {
            want[((f64::from(x) * 256.0) as usize).min(255)] += 1.0;
        }
        // Bin edges are rounded differently in the two (float arithmetic on `(x - lo) / width`): a
        // value within an ulp of an edge may land in the neighbouring bin. Allow 64 of ~39000 per bin.
        let v = compare(&lh, &ah, 0.0, 64.0).and(compare_ref("luma", &lh, &want, 0.0, 64.0)).and(compare_ref("AF", &ah, &want, 0.0, 64.0));
        let total = |h: &[f32]| h.iter().map(|&c| f64::from(c)).sum::<f64>() == n as f64;
        let v = if total(&lh) && total(&ah) { v } else { Verdict::no("NO: histogram totals differ from n") };
        b.run(
            "9 histogram 256 bins",
            "10M f32 in [0,1)",
            v,
            "edge rounding may move a few values between neighbouring bins",
            &mut || {
                xl.histogram(256, 0.0, 1.0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::histogram(&xa, 256, 0.0, 1.0).eval();
                afx::sync();
            },
        );
    });
}

pub fn gathers(b: &mut Bench) {
    b.guarded("gather", |b| {
        let ctx = b.ctx.clone();
        let n = 10_000_000usize;
        let src = uniform(n, 81, 0.0, 1.0);
        let idx = indices(n, 82, n as u32);
        let (sl, sa) = (lumax::arr(&ctx, &src, n), afx::arr(&src, &[n as u64]));
        let (il, ia) = (lumax::arr(&ctx, &idx, n), afx::arr(&idx, &[n as u64]));
        let lo = lumax::host(&sl.take(&il));
        let ao = afx::host(&af::lookup(&sa, &ia, 0));
        let want: Vec<f32> = idx.iter().map(|&i| src[i as usize]).collect();
        let v = exact(&lo, &ao).and(exact(&lo, &want));
        drop((lo, ao, want));
        b.run(
            "10 gather (take / lookup)",
            "10M random indices into 10M",
            v,
            "",
            &mut || {
                sl.take(&il).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::lookup(&sa, &ia, 0).eval();
                afx::sync();
            },
        );
        drop((sl, sa, il, ia, src, idx));

        let (n, bins) = (64_000_000usize, 65_536usize);
        let idx = indices(n, 83, bins as u32);
        let il = lumax::arr(&ctx, &idx, n);
        let ia = afx::arr(&idx, &[n as u64]);
        let mut want = vec![0.0f64; bins];
        for &i in &idx {
            want[i as usize] += 1.0;
        }
        let lc = counts_f32(&lumax::host(&luma::bincount(&il, bins)));
        let v = compare_ref("luma", &lc, &want, 0.0, 0.0);
        let lc_run = &mut || {
            luma::bincount(&il, bins).eval().unwrap();
            ctx.sync().unwrap();
        };
        // ArrayFire has no scatter-add; `histogram` over the integer keys is its counting equivalent
        // (unit weights only). Its CUDA kernel keeps the bins in shared memory, so a large bin count may be refused.
        match attempt(|| counts_f32(&afx::host(&af::histogram(&ia, bins as u32, 0.0, bins as f64)))) {
            Some(ac) => {
                let v = v.and(compare_ref("AF", &ac, &want, 0.0, 0.0));
                b.run(
                    "10 bincount (count into 64K bins)",
                    "64M u32 keys",
                    v,
                    "luma bincount (atomic scatter-add of 1) vs AF histogram over u32 keys, 65536 bins: the only AF equivalent (unit weights)",
                    lc_run,
                    &mut || {
                        af::histogram(&ia, bins as u32, 0.0, bins as f64).eval();
                        afx::sync();
                    },
                );
            }
            None => {
                println!("ArrayFire histogram with {bins} bins failed (see message above); luma only");
                b.run_one("10 bincount (count into 64K bins)", "64M u32 keys", Some(lc_run), None, "ArrayFire histogram refused 65536 bins: no AF row");
            }
        }
        // Weighted scatter-add: ArrayFire has no scatter-add. The idiomatic AF route is sort_by_key + sum_by_key
        // (a group-by); every bin is occupied here, so its output is the dense result.
        let vals = uniform(n, 84, 0.0, 1.0);
        let (vl, va) = (lumax::arr(&ctx, &vals, n), afx::arr(&vals, &[n as u64]));
        let scatter_l = || {
            let mut bins_arr = LA::<f32>::zeros_on(&ctx, bins).unwrap();
            bins_arr.put_add(&il, &vl).unwrap();
            bins_arr
        };
        let scatter_a = || {
            let (k, v) = af::sort_by_key(&ia, &va, 0, true);
            af::sum_by_key(&k, &v, 0)
        };
        let mut wsum = vec![0.0f64; bins];
        for (&i, &v) in idx.iter().zip(&vals) {
            wsum[i as usize] += f64::from(v);
        }
        let lw = lumax::host(&scatter_l());
        let (ak, aw) = scatter_a();
        let (ak, aw) = (afx::host(&ak), afx::host(&aw));
        let dense = ak.len() == bins && ak.iter().enumerate().all(|(i, &k)| k as usize == i);
        let v = compare_ref("luma", &lw, &wsum, 1e-4, 0.0).and(compare_ref("AF", &aw, &wsum, 1e-4, 0.0));
        let v = if dense { v } else { Verdict::no("NO: AF sum_by_key keys are not the dense 0..65535") };
        b.run(
            "10 scatter-add f32 (weighted) into 64K bins",
            "64M",
            v,
            "AF has no scatter-add: AF side is sort_by_key + sum_by_key (a different algorithm; valid because every bin is occupied)",
            &mut || {
                scatter_l().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, v) = scatter_a();
                k.eval();
                v.eval();
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 11. Stencil loop
// ------------------------------------------------------------------------------------------

/// One 5-point Jacobi sweep with a right-hand side, periodic: `(n + s + e + w + b) / 4`.
pub fn jacobi_l(u: &L, rhs: &L) -> L {
    let ns = &u.shift([1isize, 0]) + &u.shift([-1isize, 0]);
    let ew = &u.shift([0isize, 1]) + &u.shift([0isize, -1]);
    &(&(&ns + &ew) + rhs) * 0.25f32
}

/// The same sweep with zero outside the grid.
pub fn jacobi_zero_l(u: &L, rhs: &L) -> L {
    let ns = &u.shift_fill([1isize, 0], 0.0) + &u.shift_fill([-1isize, 0], 0.0);
    let ew = &u.shift_fill([0isize, 1], 0.0) + &u.shift_fill([0isize, -1], 0.0);
    &(&(&ns + &ew) + rhs) * 0.25f32
}

pub fn jacobi_a(u: &A, rhs: &A) -> A {
    let ns = &af::shift(u, &[1, 0, 0, 0]) + &af::shift(u, &[-1, 0, 0, 0]);
    let ew = &af::shift(u, &[0, 1, 0, 0]) + &af::shift(u, &[0, -1, 0, 0]);
    &(&(&ns + &ew) + rhs) * 0.25f32
}

/// Zero boundary through a 3x3 convolution (ArrayFire zero-pads) plus `rhs / 4`.
pub fn jacobi_conv_a(u: &A, rhs4: &A, filt: &A) -> A {
    &af::convolve2(u, filt, af::ConvMode::DEFAULT, af::ConvDomain::SPATIAL) + rhs4
}

pub fn jacobi_filter() -> Vec<f32> {
    // Symmetric, so the flip of a convolution does not matter. Column-major 3x3.
    vec![0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0]
}

pub const STENCIL_STEPS: usize = 100;

pub fn stencil(b: &mut Bench) {
    b.guarded("stencil", |b| {
        let ctx = b.ctx.clone();
        let n = 2048usize;
        let u0 = uniform(n * n, 91, 0.0, 1.0);
        let rhs: Vec<f32> = uniform(n * n, 92, -0.5, 0.5).iter().map(|v| v * 0.01).collect();
        let (ul, rl) = (lumax::arr(&ctx, &u0, [n, n]), lumax::arr(&ctx, &rhs, [n, n]));
        let (ua, ra) = (afx::arr(&u0, &[n as u64, n as u64]), afx::arr(&rhs, &[n as u64, n as u64]));
        let size = format!("{n} x {n}, {STENCIL_STEPS} iterations");
        let iterate_l = || {
            let [r] = luma::iterate_all([&ul], STENCIL_STEPS, |s| [jacobi_l(&s[0], &rl)]);
            r
        };
        let host_loop_l = || {
            let mut cur = jacobi_l(&ul, &rl);
            cur.eval().unwrap();
            for _ in 1..STENCIL_STEPS {
                let next = jacobi_l(&cur, &rl);
                next.eval().unwrap();
                cur = next;
            }
            cur
        };
        let loop_a = || {
            let mut cur = jacobi_a(&ua, &ra);
            cur.eval();
            for _ in 1..STENCIL_STEPS {
                let next = jacobi_a(&cur, &ra);
                next.eval();
                cur = next;
            }
            cur
        };
        let ao = afx::host(&loop_a());
        let v = compare(&lumax::host(&iterate_l()), &ao, 1e-4, 1e-5);
        b.run(
            "11 stencil: 5-pt Jacobi, luma iterate_all",
            &size,
            v,
            "periodic (shift); luma: recorded once, replayed as a fused loop; AF: shift kernels + fused add, evaluated every step",
            &mut || {
                let r = iterate_l();
                r.eval().unwrap();
                ctx.sync().unwrap();
                drop(r);
            },
            &mut || {
                loop_a();
                afx::sync();
            },
        );
        let v = compare(&lumax::host(&host_loop_l()), &ao, 1e-4, 1e-5);
        b.run(
            "11 stencil: 5-pt Jacobi, luma eval per step",
            &size,
            v,
            "periodic (shift); one eval per step on both sides; AF launches each step (4 shift copies + 1 fused add), luma postpones and fuses evals while the GPU is busy (documented Array::eval behaviour)",
            &mut || {
                let r = host_loop_l();
                ctx.sync().unwrap();
                drop(r);
            },
            &mut || {
                loop_a();
                afx::sync();
            },
        );
        // Zero boundary: luma shift_fill, ArrayFire convolve2 (spatial).
        let filt = afx::arr(&jacobi_filter(), &[3, 3]);
        let rhs4 = &ra * 0.25f32;
        rhs4.eval();
        let conv_loop = || {
            let mut cur = jacobi_conv_a(&ua, &rhs4, &filt);
            cur.eval();
            for _ in 1..STENCIL_STEPS {
                let next = jacobi_conv_a(&cur, &rhs4, &filt);
                next.eval();
                cur = next;
            }
            cur
        };
        let iterate_zero_l = || {
            let [r] = luma::iterate_all([&ul], STENCIL_STEPS, |s| [jacobi_zero_l(&s[0], &rl)]);
            r
        };
        let v = compare(&lumax::host(&iterate_zero_l()), &afx::host(&conv_loop()), 1e-4, 1e-5);
        b.run(
            "11 stencil: 5-pt Jacobi, convolve2 variant",
            &size,
            v,
            "zero boundary: luma iterate_all (shift_fill) vs AF convolve2 3x3 (spatial) + fused add per step",
            &mut || {
                let r = iterate_zero_l();
                r.eval().unwrap();
                ctx.sync().unwrap();
                drop(r);
            },
            &mut || {
                conv_loop();
                afx::sync();
            },
        );
    });
}

// ------------------------------------------------------------------------------------------
// 12. Select / filter
// ------------------------------------------------------------------------------------------

pub fn filters(b: &mut Bench) {
    b.guarded("filter", |b| {
        let ctx = b.ctx.clone();
        let n = 10_000_000usize;
        let x = uniform(n, 101, 0.0, 1.0);
        let y = uniform(n, 102, 0.0, 1.0);
        let z = uniform(n, 103, 0.0, 1.0);
        let (xl, yl, zl) = (lumax::arr(&ctx, &x, n), lumax::arr(&ctx, &y, n), lumax::arr(&ctx, &z, n));
        let (xa, ya, za) = (afx::arr(&x, &[n as u64]), afx::arr(&y, &[n as u64]), afx::arr(&z, &[n as u64]));
        let hits = x.iter().filter(|&&v| v > 0.5).count();
        let size = format!("{} ({:.0}% true)", human(n), 100.0 * hits as f64 / n as f64);

        // Elementwise select by a mask computed from x.
        let want: Vec<f32> = (0..n).map(|i| if x[i] > 0.5 { y[i] } else { z[i] }).collect();
        let sel_l = || luma::select(&xl.gt(0.5f32), &yl, &zl);
        let sel_a = || af::select(&ya, &af::gt(&xa, &0.5f32, false), &za);
        let v = exact(&lumax::host(&sel_l()), &want).and(exact(&afx::host(&sel_a()), &want));
        b.run(
            "12 select(mask, a, b)",
            &size,
            v,
            "mask computed from x inside the timed call",
            &mut || {
                sel_l().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                sel_a().eval();
                afx::sync();
            },
        );

        // Compaction: the elements where the mask holds, in order.
        let want: Vec<f32> = (0..n).filter(|&i| x[i] > 0.5).map(|i| y[i]).collect();
        let filt_l = || yl.filter(&xl.gt(0.5f32)).unwrap().expect("some hits");
        let filt_a = || {
            let idx = af::locate(&af::gt(&xa, &0.5f32, false));
            af::lookup(&ya, &idx, 0)
        };
        let v = exact(&lumax::host(&filt_l()), &want).and(exact(&afx::host(&filt_a()), &want));
        b.run(
            "12 filter (mask compaction)",
            &size,
            v,
            "luma filter vs AF where + lookup (y(mask)); both sizes data-dependent, result stays on device",
            &mut || {
                filt_l().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                filt_a().eval();
                afx::sync();
            },
        );

        // Indices only.
        let wi: Vec<f32> = (0..n).filter(|&i| x[i] > 0.5).map(|i| i as f32).collect();
        let to_f = |v: Vec<u32>| v.into_iter().map(|i| i as f32).collect::<Vec<f32>>();
        let nz_l = || xl.gt(0.5f32).nonzero().unwrap().expect("some hits");
        let nz_a = || af::locate(&af::gt(&xa, &0.5f32, false));
        let v = exact(&to_f(lumax::host(&nz_l())), &wi).and(exact(&to_f(afx::host(&nz_a())), &wi));
        b.run(
            "12 where / nonzero (indices)",
            &size,
            v,
            "",
            &mut || {
                nz_l().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                nz_a().eval();
                afx::sync();
            },
        );
    });
}
