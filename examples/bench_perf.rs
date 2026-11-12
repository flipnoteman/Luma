//! GPU-only throughput of the large-array workloads, for performance work. One size per run:
//! `cargo run --release --example bench_perf -- 67108864 [filter]`.

use luma::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Median wall time in ms after warming up for at least `warm` (lets the GPU clock ramp up).
fn median(mut f: impl FnMut()) -> f64 {
    let warm = Duration::from_millis(300);
    let start = Instant::now();
    while start.elapsed() < warm {
        f();
        default_context().unwrap().sync().unwrap();
    }
    let mut s: Vec<f64> = (0..15)
        .map(|_| {
            let t = Instant::now();
            f();
            default_context().unwrap().sync().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn report(name: &str, ms: f64, work: f64, unit: &str) {
    println!(
        "{name:<28} {ms:>9.3} ms {:>9.1} {unit}",
        work / (ms * 1e-3) / 1e9
    );
}

fn main() -> Result<()> {
    let n: usize = std::env::args()
        .nth(1)
        .map(|a| a.parse().unwrap())
        .unwrap_or(1 << 26);
    let filter = std::env::args().nth(2).unwrap_or_default();
    let want = |name: &str| filter.is_empty() || name.contains(&filter);
    let nf = n as f64;
    let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 + 0.2).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i % 89) as f32 * 0.02 + 0.1).collect();
    let x = Array::from_slice(&xs, n)?;
    let y = Array::from_slice(&ys, n)?;
    x.eval()?;
    println!("N = {n}");

    if want("sqrt") {
        let ms = median(|| ((&x * 2.0 + &y).sqrt() - 1.0).eval().unwrap());
        report("sqrt(2x+y)-1", ms, nf * 12.0, "GB/s");
    }
    if want("dot") {
        let ms = median(|| {
            black_box((&x * &y).sum().item().unwrap());
        });
        report("dot", ms, nf * 8.0, "GB/s");
    }
    if want("sum") {
        let ms = median(|| {
            black_box(x.sum().item().unwrap());
        });
        report("sum", ms, nf * 4.0, "GB/s");
    }
    if want("alu") {
        let ms = median(|| {
            let mut z = &x * 1.0;
            for _ in 0..20 {
                z = &z * &z * 0.5 + 0.25;
            }
            z.eval().unwrap()
        });
        report("20x (y*y*0.5+0.25)", ms, nf * 60.0, "GFLOP/s");
    }
    if want("sin") {
        let ms = median(|| {
            let mut z = &x * 1.0;
            for _ in 0..32 {
                z = z.sin() * 0.99 + 0.01;
            }
            z.eval().unwrap()
        });
        report("32x sin chain", ms, nf * 32.0, "Gsin/s");
    }
    if want("inplace") {
        let mut z = x.reshape(n) * 1.0;
        z.eval()?;
        let ms = median(|| {
            z += 1.0;
            z.eval().unwrap();
        });
        report("z += 1.0", ms, nf * 8.0, "GB/s");
    }
    if want("chain") {
        let mut z = &x * 1.0;
        z.eval()?;
        let ms = median(|| {
            z += &(&x * &y);
            z *= 0.5;
            z -= 0.25;
            z.eval().unwrap();
        });
        report("z += x*y; z *= .5; z -= .25", ms, nf * 16.0, "GB/s(min)");
    }
    if want("view") {
        let mut z = &x * 1.0;
        z.eval()?;
        let view = z.reshape(n);
        let ms = median(|| {
            z += &(&x * &y);
            view.eval().unwrap();
        });
        report("shared z += x*y", ms, nf * 16.0, "GB/s(min)");
    }
    if want("conv") {
        for taps in [64usize, 1024] {
            let k: Vec<f32> = (0..taps).map(|i| 1.0 / (i as f32 + 1.0)).collect();
            let ka = Array::from_slice(&k, taps)?;
            let ms = median(|| x.conv1d(&ka, 1, 0).eval().unwrap());
            report(
                &format!("conv1d {taps} taps"),
                ms,
                (n - taps + 1) as f64 * taps as f64 * 2.0,
                "GFLOP/s",
            );
        }
    }
    if want("fft") {
        let k: Vec<f32> = (0..1024).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let ka = Array::from_slice(&k, 1024)?;
        let ms = median(|| x.conv1d_fft(&ka).eval().unwrap());
        report("conv1d_fft 1024 taps", ms, nf * 4.0, "GB/s(io)");
    }
    if want("stencil") {
        let steps = 50;
        let ms = median(|| {
            let mut u = &x * 1.0;
            for _ in 0..steps {
                u = &u + (u.shift(1) + u.shift(-1) - &u * 2.0) * 0.1;
            }
            u.eval().unwrap()
        });
        report(
            "heat stencil 50 steps",
            ms,
            nf * 8.0 * steps as f64,
            "GB/s(min)",
        );
    }
    if want("upload") {
        let ms = median(|| Array::from_slice(&xs, n).unwrap().eval().unwrap());
        report("from_slice", ms, nf * 4.0, "GB/s");
    }
    if want("tovec") {
        let ms = median(|| {
            black_box(x.to_vec().unwrap());
        });
        report("to_vec", ms, nf * 4.0, "GB/s");
    }
    if want("read_into") {
        let mut host = vec![0.0f32; n];
        let ms = median(|| {
            x.read_into(&mut host).unwrap();
            black_box(&host);
        });
        report("read_into", ms, nf * 4.0, "GB/s");
    }
    Ok(())
}
