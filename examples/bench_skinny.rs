//! Products with a small output and a long K: class sums `onehot(y)ᵀ X`, Gram matrices, `Xᵀ r`
//! gradients and covariances, plus rows of other shapes to check nothing else got slower.
//!
//! `cargo run --release --example bench_skinny -- [filter]` (the filter is a substring of the case
//! name; `sweep` runs the shape grid that sets the planner's thresholds).

use luma::*;
use std::time::{Duration, Instant};

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// (min, median) wall time in ms of `f` evaluated and synchronised, after warming up.
fn time(ctx: &Context, f: &mut dyn FnMut()) -> (f64, f64) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(300) {
        f();
        ctx.sync().unwrap();
    }
    let mut s: Vec<f64> = (0..25)
        .map(|_| {
            let t = Instant::now();
            f();
            ctx.sync().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    s.sort_by(f64::total_cmp);
    (s[0], s[s.len() / 2])
}

fn report(name: &str, ctx: &Context, work: f64, f: &mut dyn FnMut()) {
    let (min, med) = time(ctx, f);
    println!("{name:<34} min {min:>8.3} ms  median {med:>8.3} ms  {:>7.1} GFLOP/s", work / (min * 1e-3) / 1e9);
}

fn main() -> Result<()> {
    let ctx = default_context()?;
    let filter = std::env::args().nth(1).unwrap_or_default();
    let want = |name: &str| filter.is_empty() || name.contains(&filter);
    println!("{:?}", ctx.device());

    // ---- class sums, as the fit of a Gaussian naive Bayes: onehot(y)ᵀ X, `classes` x d over n samples.
    for (n, d) in [(1_000_000usize, 32usize), (200_000, 256), (10_000, 16)] {
        let name = format!("onehot.T @ X {n} x {d}");
        if !want(&name) && !want("onehot") {
            continue;
        }
        let classes = 10usize;
        let x = Array::from_slice_on(&ctx, &values(n * d, 7), [d, n])?;
        let labels: Vec<u32> = values(n, 8).iter().map(|v| (((v + 1.0) * 0.5 * classes as f32) as u32).min(classes as u32 - 1)).collect();
        let codes = Array::<u32>::from_slice_on(&ctx, &labels, [n])?;
        x.eval()?;
        codes.eval()?;
        let onehot = || codes.reshape([1, n]).broadcast_to([classes, n]).eq(Array::<u32>::range_on(&ctx, [classes, n], 0).unwrap()).to_f32();
        let run = || onehot().transpose().matmul(&x);
        // Evaluated one-hot too: the matrix as a stored operand.
        let hot = onehot();
        hot.eval()?;
        report(&name, &ctx, 2.0 * (classes * n * d) as f64, &mut || run().eval().unwrap());
        report(&format!("{name} (stored one-hot)"), &ctx, 2.0 * (classes * n * d) as f64, &mut || hot.transpose().matmul(&x).eval().unwrap());
    }

    // ---- Gram matrix of the standardised features, both operands pending.
    for (f, k) in [(64usize, 1usize << 20), (32, 1 << 20), (16, 1 << 20), (64, 1 << 16)] {
        let name = format!("gram_fused {f} x {k}");
        if !want(&name) && !want("gram") {
            continue;
        }
        let x = Array::from_slice_on(&ctx, &values(f * k, 51), [f, k])?;
        let c = Array::from_slice_on(&ctx, &values(f, 52), [f, 1])?;
        let s = Array::from_slice_on(&ctx, &values(f, 53), [f, 1])?;
        eval_all(&[&x, &c, &s])?;
        report(&name, &ctx, 2.0 * (f * f * k) as f64, &mut || {
            let z = (&x - &c) * &s;
            z.transpose().matmul(&z).eval().unwrap();
        });
        // The standardised matrix evaluated first, then the product of the stored array.
        report(&format!("{name} (stored z)"), &ctx, 2.0 * (f * f * k) as f64, &mut || {
            let z = (&x - &c) * &s;
            z.eval().unwrap();
            z.transpose().matmul(&z).eval().unwrap();
        });
    }

    // ---- Xᵀ r (gradient of least squares) and Xᵀ X without fusion, `d` features over `n` samples.
    for (n, d) in [(1usize << 20, 64usize), (1 << 22, 64), (1 << 22, 16), (1 << 18, 256)] {
        if !want("grad") {
            continue;
        }
        let x = Array::from_slice_on(&ctx, &values(n * d, 61), [d, n])?;
        let r = Array::from_slice_on(&ctx, &values(n, 62), [1, n])?;
        // Several columns: the gradient of a softmax regression with 20 classes.
        let rc = Array::from_slice_on(&ctx, &values(20 * n, 63), [20, n])?;
        eval_all(&[&x, &r, &rc])?;
        report(&format!("X.T @ r {n} x {d}"), &ctx, 2.0 * (n * d) as f64, &mut || x.transpose().matmul(&r).eval().unwrap());
        report(&format!("r.T @ X {n} x {d}"), &ctx, 2.0 * (n * d) as f64, &mut || r.transpose().matmul(&x).eval().unwrap());
        report(&format!("X.T @ R(c=20) {n} x {d}"), &ctx, 2.0 * (n * d * 20) as f64, &mut || x.transpose().matmul(&rc).eval().unwrap());
        if n * d * d <= 1 << 32 {
            report(&format!("X.T @ X {n} x {d}"), &ctx, 2.0 * (n * d * d) as f64, &mut || x.transpose().matmul(&x).eval().unwrap());
        }
    }

    // ---- Plain row-major products with a small output: A (m x k) @ B (k x n), k long.
    if want("rowmajor") {
        for (m, n, k) in [(1usize, 1usize, 1usize << 22), (4, 4, 1 << 20), (16, 16, 1 << 20), (8, 128, 1 << 18)] {
            let a = Array::from_slice_on(&ctx, &values(m * k, 71), [k, m])?;
            let b = Array::from_slice_on(&ctx, &values(k * n, 72), [n, k])?;
            eval_all(&[&a, &b])?;
            report(&format!("rowmajor {m}x{k} @ {k}x{n}"), &ctx, 2.0 * (m * n * k) as f64, &mut || a.matmul(&b).eval().unwrap());
        }
    }

    // ---- A matrix times one or a few vectors: A (m x k) @ B (k x n), n small, bandwidth bound.
    if want("matvec") {
        for (m, k, n) in [(2048usize, 2048usize, 1usize), (8192, 8192, 1), (4096, 16384, 1), (32768, 2048, 1), (2048, 2048, 4), (8192, 8192, 8), (256, 65536, 1)] {
            let a = Array::from_slice_on(&ctx, &values(m * k, 91), [k, m])?;
            let b = Array::from_slice_on(&ctx, &values(k * n, 92), [n, k])?;
            eval_all(&[&a, &b])?;
            let bytes = 4.0 * (m * k) as f64;
            let (min, med) = time(&ctx, &mut || a.matmul(&b).eval().unwrap());
            println!("{:<34} min {min:>8.3} ms  median {med:>8.3} ms  {:>7.1} GB/s", format!("matvec {m}x{k} @ {k}x{n}"), bytes / (min * 1e-3) / 1e9);
        }
    }

    // ---- Shape grid: square outputs over several K, to find where the skinny kernels stop paying.
    if want("sweep") {
        for k in [4096usize, 16384, 65536, 262144, 1 << 20] {
            for m in [8usize, 16, 32, 64, 96, 128, 192, 256] {
                let n = m;
                if m * n * k > 1 << 33 || m * k > 1 << 27 {
                    continue;
                }
                let a = Array::from_slice_on(&ctx, &values(m * k, 71), [k, m])?;
                let b = Array::from_slice_on(&ctx, &values(k * n, 72), [n, k])?;
                eval_all(&[&a, &b])?;
                report(&format!("sweep {m}x{k} @ {k}x{n}"), &ctx, 2.0 * (m * n * k) as f64, &mut || a.matmul(&b).eval().unwrap());
            }
        }
    }

    // ---- Dense products the skinny kernels must leave alone.
    if want("dense") {
        for (m, n, k) in [(512usize, 512usize, 512usize), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (1000, 1000, 1000), (4096, 64, 4096), (64, 4096, 4096), (256, 256, 16384), (128, 128, 65536), (2048, 128, 2048), (32, 32, 1 << 16), (1 << 12, 8, 256)] {
            let a = Array::from_slice_on(&ctx, &values(m * k, 81), [k, m])?;
            let b = Array::from_slice_on(&ctx, &values(k * n, 82), [n, k])?;
            eval_all(&[&a, &b])?;
            report(&format!("dense {m}x{k} @ {k}x{n}"), &ctx, 2.0 * (m * n * k) as f64, &mut || a.matmul(&b).eval().unwrap());
        }
    }
    Ok(())
}
