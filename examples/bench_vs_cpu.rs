//! Large-array workloads on the GPU against the same work on every CPU core (plain Rust,
//! written so that LLVM vectorizes it). Sizes in elements as arguments, default 2^24 and 2^26:
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench_vs_cpu -- 67108864`.

use luma::*;
use std::hint::black_box;
use std::time::Instant;

fn median<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..2 {
        f();
        default_context().unwrap().sync().unwrap();
    }
    let mut s: Vec<f64> = (0..iters)
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

fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Splits `out` into one contiguous chunk per thread; `f(base, chunk)` fills a chunk.
fn par_chunks(out: &mut [f32], f: impl Fn(usize, &mut [f32]) + Sync) {
    let chunk = out.len().div_ceil(threads()).max(4096);
    std::thread::scope(|s| {
        for (ci, c) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || f(ci * chunk, c));
        }
    });
}

/// `out[i] = f(inp[i])` over per-thread chunks; no indexing, so the loop vectorizes.
fn par_map(inp: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32 + Sync) {
    let chunk = out.len().div_ceil(threads()).max(4096);
    std::thread::scope(|s| {
        for (i, o) in inp.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let f = &f;
            s.spawn(move || {
                for (a, b) in i.iter().zip(o.iter_mut()) {
                    *b = f(*a);
                }
            });
        }
    });
}

/// 16 independent accumulators so LLVM vectorizes without reassociation flags.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..16 {
            acc[l] += x[l] * y[l];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

fn par_dot(a: &[f32], b: &[f32]) -> f32 {
    let chunk = a.len().div_ceil(threads()).max(4096);
    std::thread::scope(|s| {
        let hs: Vec<_> = a
            .chunks(chunk)
            .zip(b.chunks(chunk))
            .map(|(x, y)| s.spawn(move || dot(x, y)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// Branch-free sin: range reduction to [-pi/2, pi/2] plus a degree-9 Taylor polynomial; vectorizes.
#[inline(always)]
fn fast_sin(x: f32) -> f32 {
    const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
    let k = (x * INV_PI).round();
    let r = x - k * std::f32::consts::PI;
    let r2 = r * r;
    let p = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362880.0)))));
    f32::from_bits(p.to_bits() ^ (((k as i32) << 31) as u32))
}

const SIN_ITERS: usize = 32;
const ALU_ITERS: usize = 20;

fn sin_gpu(x: &Array<f32>) -> Array<f32> {
    let mut y = x * 1.0;
    for _ in 0..SIN_ITERS {
        y = y.sin() * 0.99 + 0.01;
    }
    y
}
#[inline(always)]
fn sin_libm(mut v: f32) -> f32 {
    for _ in 0..SIN_ITERS {
        v = v.sin() * 0.99 + 0.01;
    }
    v
}
/// 16 elements in lockstep so the inner lane loop vectorizes despite the long dependent chain.
fn sin_fast_map(inp: &[f32], out: &mut [f32]) {
    let chunk = out.len().div_ceil(threads()).max(4096);
    std::thread::scope(|s| {
        for (i, o) in inp.chunks(chunk).zip(out.chunks_mut(chunk)) {
            s.spawn(move || {
                for (a, b) in i.chunks(16).zip(o.chunks_mut(16)) {
                    let mut v = [0f32; 16];
                    v[..a.len()].copy_from_slice(a);
                    for _ in 0..SIN_ITERS {
                        for l in &mut v {
                            *l = fast_sin(*l) * 0.99 + 0.01;
                        }
                    }
                    b.copy_from_slice(&v[..b.len()]);
                }
            });
        }
    });
}

fn alu_gpu(x: &Array<f32>) -> Array<f32> {
    let mut y = x * 1.0;
    for _ in 0..ALU_ITERS {
        y = &y * &y * 0.5 + 0.25;
    }
    y
}
#[inline(always)]
fn alu_cpu(mut v: f32) -> f32 {
    for _ in 0..ALU_ITERS {
        v = v * v * 0.5 + 0.25;
    }
    v
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f32::max)
}

fn gpu_kernel_ms(f: impl FnOnce()) -> f64 {
    let ctx = default_context().unwrap();
    let ((), p) = ctx.profile(f).unwrap();
    p.total().as_secs_f64() * 1e3
}

struct Row<'a> {
    name: &'a str,
    n: usize,
    gpu: f64,
    kern: f64,
    cpu: f64,
    work: f64,
    unit: &'a str,
    diff: Option<f32>,
}

fn print(r: Row) {
    println!(
        "{:<28} {:>10} {:>10.2} {:>10.2} {:>10.2} {:>7.1}x {:>9.0} {:>9.0} {:<6} {:>8}",
        r.name,
        r.n,
        r.gpu,
        r.kern,
        r.cpu,
        r.cpu / r.gpu,
        r.work / (r.gpu * 1e-3) / 1e9,
        r.work / (r.cpu * 1e-3) / 1e9,
        r.unit,
        r.diff.map(|d| format!("{d:.1e}")).unwrap_or_default(),
    );
}

fn main() -> Result<()> {
    println!(
        "CPU threads = {}; GPU ms = wall (record+submit+wait); kern ms = GPU timestamp total",
        threads()
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9} {:<6} max|GPU-CPU|",
        "workload", "N", "GPU ms", "kern ms", "CPUmt ms", "speedup", "GPU rate", "CPU rate", "unit"
    );
    let mut sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    if sizes.is_empty() {
        sizes = vec![1 << 24, 1 << 26];
    }
    for &n in &sizes {
        let it = if n > 50_000_000 { 5 } else { 9 };
        let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 + 0.2).collect();
        let ys: Vec<f32> = (0..n).map(|i| (i % 89) as f32 * 0.02 + 0.1).collect();
        let x = Array::from_slice(&xs, n)?;
        let y = Array::from_slice(&ys, n)?;
        let mut out = vec![0f32; n];
        let nf = n as f64;

        // memory-bound: 12 B/elem
        let gpu = median(it, || ((&x * 2.0 + &y).sqrt() - 1.0).eval().unwrap());
        let kern = gpu_kernel_ms(|| ((&x * 2.0 + &y).sqrt() - 1.0).eval().unwrap());
        let cpu = median(it, || {
            par_chunks(&mut out, |b, c| {
                for (i, o) in c.iter_mut().enumerate() {
                    *o = (xs[b + i] * 2.0 + ys[b + i]).sqrt() - 1.0;
                }
            });
            black_box(&out);
        });
        let d = max_diff(&((&x * 2.0 + &y).sqrt() - 1.0).to_vec()?, &out);
        print(Row {
            name: "sqrt(2x+y)-1",
            n,
            gpu,
            kern,
            cpu,
            work: nf * 12.0,
            unit: "GB/s",
            diff: Some(d),
        });

        // dot: 8 B/elem
        let gpu = median(it, || {
            black_box((&x * &y).sum().item().unwrap());
        });
        let kern = gpu_kernel_ms(|| {
            black_box((&x * &y).sum().item().unwrap());
        });
        let cpu = median(it, || {
            black_box(par_dot(&xs, &ys));
        });
        let rel = ((&x * &y).sum().item()? - par_dot(&xs, &ys)).abs() / par_dot(&xs, &ys);
        print(Row {
            name: "dot (rel err)",
            n,
            gpu,
            kern,
            cpu,
            work: nf * 8.0,
            unit: "GB/s",
            diff: Some(rel),
        });

        // pure ALU chain: 3 flop/iter
        let gpu = median(it, || alu_gpu(&x).eval().unwrap());
        let kern = gpu_kernel_ms(|| alu_gpu(&x).eval().unwrap());
        let cpu = median(it, || {
            par_map(&xs, &mut out, alu_cpu);
            black_box(&out);
        });
        let d = max_diff(&alu_gpu(&x).to_vec()?, &out);
        let work = nf * 3.0 * ALU_ITERS as f64;
        print(Row {
            name: "20x (y*y*0.5+0.25)",
            n,
            gpu,
            kern,
            cpu,
            work,
            unit: "GFLOP",
            diff: Some(d),
        });

        // transcendental chain vs libm sin and vs vectorizable fast sin
        let gpu = median(it, || sin_gpu(&x).eval().unwrap());
        let kern = gpu_kernel_ms(|| sin_gpu(&x).eval().unwrap());
        let gpu_out = sin_gpu(&x).to_vec()?;
        let work = nf * SIN_ITERS as f64;
        if n <= 16_777_216 {
            let cpu = median(3, || {
                par_map(&xs, &mut out, sin_libm);
                black_box(&out);
            });
            let d = max_diff(&gpu_out, &out);
            print(Row {
                name: "32x sin chain, CPU libm",
                n,
                gpu,
                kern,
                cpu,
                work,
                unit: "Gsin/s",
                diff: Some(d),
            });
        }
        let cpu = median(it, || {
            sin_fast_map(&xs, &mut out);
            black_box(&out);
        });
        let d = max_diff(&gpu_out, &out);
        print(Row {
            name: "32x sin chain, CPU fast sin",
            n,
            gpu,
            kern,
            cpu,
            work,
            unit: "Gsin/s",
            diff: Some(d),
        });

        // conv1d
        for &taps in &[64usize, 1024] {
            if taps > 64 && n > 16_777_216 {
                continue;
            }
            let k: Vec<f32> = (0..taps).map(|i| 1.0 / (i as f32 + 1.0)).collect();
            let ka = Array::from_slice(&k, taps)?;
            let m = n - taps + 1;
            let mut cout = vec![0f32; m];
            let gpu = median(it, || x.conv1d(&ka, 1, 0).eval().unwrap());
            let kern = gpu_kernel_ms(|| x.conv1d(&ka, 1, 0).eval().unwrap());
            let cpu = median(if taps > 64 { 3 } else { it }, || {
                par_chunks(&mut cout, |b, c| {
                    for (i, o) in c.iter_mut().enumerate() {
                        *o = dot(&xs[b + i..b + i + taps], &k);
                    }
                });
                black_box(&cout);
            });
            let g = x.conv1d(&ka, 1, 0).to_vec()?;
            let d = max_diff(&g, &cout);
            let name = format!("conv1d {taps} taps");
            print(Row {
                name: &name,
                n,
                gpu,
                kern,
                cpu,
                work: m as f64 * taps as f64 * 2.0,
                unit: "GFLOP",
                diff: Some(d),
            });
        }

        // 200-step periodic heat stencil, read back once at the end
        if n <= 1 << 26 {
            let steps = 200;
            let run_gpu = || {
                let mut u = &x * 1.0;
                for _ in 0..steps {
                    u = &u + (u.shift(1) + u.shift(-1) - &u * 2.0) * 0.1;
                }
                u
            };
            let gpu = median(3, || run_gpu().eval().unwrap());
            let kern = gpu_kernel_ms(|| run_gpu().eval().unwrap());
            let mut a = xs.clone();
            let mut bbuf = vec![0f32; n];
            let cpu = median(3, || {
                a.copy_from_slice(&xs);
                for _ in 0..steps {
                    let src = &a;
                    par_chunks(&mut bbuf, |base, c| {
                        let at = |j: usize| {
                            src[j] + (src[(j + n - 1) % n] + src[(j + 1) % n] - src[j] * 2.0) * 0.1
                        };
                        let lo = if base == 0 { 1 } else { 0 };
                        let hi = if base + c.len() == n {
                            c.len() - 1
                        } else {
                            c.len()
                        };
                        let (l, m, r) = (
                            &src[base + lo - 1..],
                            &src[base + lo..],
                            &src[base + lo + 1..],
                        );
                        for (i, o) in c[lo..hi].iter_mut().enumerate() {
                            *o = m[i] + (l[i] + r[i] - m[i] * 2.0) * 0.1;
                        }
                        if lo == 1 {
                            c[0] = at(0);
                        }
                        if hi < c.len() {
                            c[hi] = at(n - 1);
                        }
                    });
                    std::mem::swap(&mut a, &mut bbuf);
                }
                black_box(&a);
            });
            let d = max_diff(&run_gpu().to_vec()?, &a);
            let work = nf * steps as f64 * 8.0;
            print(Row {
                name: "heat stencil 200 steps",
                n,
                gpu,
                kern,
                cpu,
                work,
                unit: "GB/s",
                diff: Some(d),
            });
        }
        println!();
    }
    Ok(())
}
