//! Sort along an axis and quantiles: wall time, GPU time and the kernels of one run.
//!
//! `cargo run --release --example bench_sortquantile -- [--quick]`
//!
//! A table of 64 columns (1M rows, `[1M, 64]`, axis 0 fastest): `sort_axis` per axis and of the
//! transposed table, three quantiles per column (describe-style) and seven quantiles of one 16M
//! vector. Every case prints best-of-7 wall and GPU milliseconds, then the profile of one run.

use luma::*;
use std::time::Instant;

/// Best wall time (ms) over `RUNS` runs of `f` after warm-ups, and the profile of one more run.
fn measure(ctx: &Context, name: &str, f: &mut dyn FnMut()) {
    const RUNS: usize = 7;
    for _ in 0..3 {
        f();
    }
    let mut wall = f64::INFINITY;
    let mut gpu = f64::INFINITY;
    for _ in 0..RUNS {
        let t = Instant::now();
        let ((), profile) = ctx.profile(&mut *f).unwrap();
        wall = wall.min(t.elapsed().as_secs_f64() * 1e3);
        gpu = gpu.min(profile.total().as_secs_f64() * 1e3);
    }
    println!("{name:<28} wall {wall:8.3} ms   gpu {gpu:8.3} ms");
    if std::env::var_os("SORTQ_PROFILE").is_some() {
        let ((), profile) = ctx.profile(f).unwrap();
        for k in &profile.kernels {
            let label: String = k.label.replace("RadixPass ", "").chars().take(110).collect();
            println!("    {label:<110} x{:<3} {:9.1} us", k.launches, k.total.as_secs_f64() * 1e6);
        }
    }
}

fn main() -> Result<()> {
    let quick = std::env::args().any(|a| a == "--quick");
    let ctx = Context::new()?;
    println!("backend {:?}, device {}", ctx.backend(), ctx.device_name());
    let rows: usize = if quick { 1 << 16 } else { 1 << 20 };
    let cols = 64;
    let data = Array::<f32>::uniform_on(&ctx, [rows, cols], 1)?;
    let tall = data.clone();
    let wide = Array::<f32>::uniform_on(&ctx, [cols, rows], 2)?;
    data.eval()?;
    wide.eval()?;

    measure(&ctx, "sort_axis(0) [rows, 64]", &mut || tall.sort_axis(0).eval().unwrap());
    measure(&ctx, "sort_axis(1) [rows, 64]", &mut || tall.sort_axis(1).eval().unwrap());
    measure(&ctx, "sort_axis(0) [64, rows]", &mut || wide.sort_axis(0).eval().unwrap());

    if std::env::args().any(|a| a == "--sweep") {
        // 16M elements as lines of n: contiguous (axis 0) and strided (axis 1).
        let total = if quick { 1 << 20 } else { 1 << 24 };
        for n in [2usize, 8, 16, 64, 256, 1024, 2048, 4096, 16384] {
            let a = Array::<f32>::uniform_on(&ctx, [n, total / n], 4)?;
            let b = Array::<f32>::uniform_on(&ctx, [total / n, n], 5)?;
            a.eval()?;
            b.eval()?;
            measure(&ctx, &format!("sort_axis(0) [{n}, {}]", total / n), &mut || a.sort_axis(0).eval().unwrap());
            measure(&ctx, &format!("sort_axis(1) [{}, {n}]", total / n), &mut || b.sort_axis(1).eval().unwrap());
        }
        return Ok(());
    }

    let qs = [0.25f32, 0.5, 0.75];
    measure(&ctx, "3 x quantile_axis(0)", &mut || {
        let r: Vec<Array<f32>> = qs.iter().map(|&q| data.quantile_axis(0, q)).collect();
        eval_all(&r.iter().collect::<Vec<_>>()).unwrap();
    });

    measure(&ctx, "quantiles_axis(0, 3 qs)", &mut || data.quantiles_axis(0, &qs).eval().unwrap());

    let n = if quick { 1 << 20 } else { 1 << 24 };
    let long = Array::<f32>::uniform_on(&ctx, n, 3)?;
    long.eval()?;
    let qs7 = [0.01f32, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
    measure(&ctx, "7 x quantile (16M)", &mut || {
        let r: Vec<Array<f32>> = qs7.iter().map(|&q| long.quantile(q)).collect();
        eval_all(&r.iter().collect::<Vec<_>>()).unwrap();
    });
    measure(&ctx, "quantiles(7 qs) (16M)", &mut || long.quantiles(&qs7).eval().unwrap());
    Ok(())
}
