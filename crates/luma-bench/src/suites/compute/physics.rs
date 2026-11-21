//! Simulation-style workloads: all-pairs forces by broadcasting, particle update loops, random
//! numbers and Monte Carlo pricing, fractal iteration, integer hashing and scatter.

use luma::*;

use super::{check64, host, pick, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    nbody(h);
    particles(h);
    rng(h);
    montecarlo(h);
    fractal(h);
    integers(h);
}

// -------------------------------------------------------------------------------------------------
// n-body
// -------------------------------------------------------------------------------------------------

const SOFTENING2: f32 = 1e-3;

/// The accelerations of all-pairs gravity by broadcasting a column (sources `j`, axis 0) against a
/// row (targets `i`, axis 1): the `n x n` interaction matrix is never stored, and each component
/// is a fused reduction along axis 0.
fn forces(p: [&Array<f32>; 3], m: &Array<f32>) -> [Array<f32>; 3] {
    let n = m.len();
    let col = |a: &Array<f32>| a.reshape([n, 1]);
    let row = |a: &Array<f32>| a.reshape([1, n]);
    let d: Vec<Array<f32>> = p.iter().map(|a| col(a) - row(a)).collect();
    let r2 = &d[0] * &d[0] + &d[1] * &d[1] + &d[2] * &d[2] + SOFTENING2;
    let w = col(m) / (&r2 * r2.sqrt());
    [0, 1, 2].map(|k| (&d[k] * &w).sum_axis(0).reshape(n))
}

/// The accelerations of the bodies `at`, in `f64`.
fn forces_host(p: [&[f32]; 3], m: &[f32], at: &[usize]) -> Vec<[f64; 3]> {
    at.iter()
        .map(|&i| {
            let mut a = [0.0f64; 3];
            for j in 0..m.len() {
                let d = [0, 1, 2].map(|k| f64::from(p[k][j]) - f64::from(p[k][i]));
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + f64::from(SOFTENING2);
                let w = f64::from(m[j]) / (r2 * r2.sqrt());
                for k in 0..3 {
                    a[k] += d[k] * w;
                }
            }
            a
        })
        .collect()
}

fn nbody(h: &mut Harness) {
    h.guarded("nbody", |h| {
        let sizes: Vec<usize> = match (h.ctx.is_cpu(), h.quick) {
            (false, false) => vec![1024, 4096, 8192],
            (false, true) => vec![1024, 4096],
            (true, false) => vec![512, 2048, 4096],
            (true, true) => vec![512, 1024],
        };
        for n in sizes {
            let px = [values(n, 61, -1.0, 1.0), values(n, 62, -1.0, 1.0), values(n, 63, -1.0, 1.0)];
            let m = values(n, 64, 0.5, 1.5);
            let p = [upload(h, &px[0], n), upload(h, &px[1], n), upload(h, &px[2], n)];
            let mass = upload(h, &m, n);
            let at: Vec<usize> = ints(96, 65, n as u32).into_iter().map(|i| i as usize).collect();
            let want = forces_host([&px[0], &px[1], &px[2]], &m, &at);
            let scale = want.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs()));
            let verify = |a: &[Array<f32>; 3]| {
                for k in 0..3 {
                    let got = host(&a[k]);
                    let got: Vec<f32> = at.iter().map(|&i| got[i]).collect();
                    let w: Vec<f64> = want.iter().map(|v| v[k]).collect();
                    check64("nbody/forces", &got, &w, 1e-3, 1e-3 * scale);
                }
            };
            verify(&forces([&p[0], &p[1], &p[2]], &mass));
            let flops = Work::Flops(20.0 * (n * n) as f64);
            h.case("nbody/forces", &format!("{n} bodies"), flops, &mut || {
                let a = forces([&p[0], &p[1], &p[2]], &mass);
                eval_all(&[&a[0], &a[1], &a[2]]).unwrap();
            });
        }

        // The three components stacked on a third axis: one reduction, but the pair terms are
        // materialised (the stacked operands are evaluated first).
        let n: usize = if h.ctx.is_cpu() { pick(h, 512, 1024) } else { pick(h, 1024, 2048) };
        let px = [values(n, 61, -1.0, 1.0), values(n, 62, -1.0, 1.0), values(n, 63, -1.0, 1.0)];
        let m = values(n, 64, 0.5, 1.5);
        let p = [upload(h, &px[0], n), upload(h, &px[1], n), upload(h, &px[2], n)];
        let mass = upload(h, &m, n);
        let stacked = |p: &[Array<f32>; 3], m: &Array<f32>| {
            let col = |a: &Array<f32>| a.reshape([n, 1]);
            let row = |a: &Array<f32>| a.reshape([1, n]);
            let d: Vec<Array<f32>> = p.iter().map(|a| col(a) - row(a)).collect();
            let r2 = &d[0] * &d[0] + &d[1] * &d[1] + &d[2] * &d[2] + SOFTENING2;
            let w = (col(m) / (&r2 * r2.sqrt())).reshape([n, n, 1]);
            (&stack(2, [&d[0], &d[1], &d[2]]) * &w).sum_axis(0)
        };
        let at: Vec<usize> = ints(32, 66, n as u32).into_iter().map(|i| i as usize).collect();
        let want = forces_host([&px[0], &px[1], &px[2]], &m, &at);
        let scale = want.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs()));
        let got = host(&stacked(&p, &mass));
        for (q, &i) in at.iter().enumerate() {
            let g: Vec<f32> = (0..3).map(|k| got[i + n * k]).collect();
            check64("nbody/forces_stacked", &g, &want[q], 1e-3, 1e-3 * scale);
        }
        h.case("nbody/forces_stacked", &format!("{n} bodies"), Work::Flops(20.0 * (n * n) as f64), &mut || stacked(&p, &mass).eval().unwrap());
    });
}

// -------------------------------------------------------------------------------------------------
// Particles
// -------------------------------------------------------------------------------------------------

fn particles(h: &mut Harness) {
    h.guarded("particles", |h| {
        // Leapfrog integration with all-pairs forces, a few steps per call.
        let n: usize = if h.ctx.is_cpu() { pick(h, 512, 1024) } else { pick(h, 1024, 2048) };
        let steps = 4;
        let dt = 1e-3f32;
        let x0 = [values(n, 71, -1.0, 1.0), values(n, 72, -1.0, 1.0), values(n, 73, -1.0, 1.0)];
        let v0 = [values(n, 74, -0.1, 0.1), values(n, 75, -0.1, 0.1), values(n, 76, -0.1, 0.1)];
        let m = values(n, 77, 0.5, 1.5);
        let x: Vec<Array<f32>> = x0.iter().map(|d| upload(h, d, n)).collect();
        let v: Vec<Array<f32>> = v0.iter().map(|d| upload(h, d, n)).collect();
        let mass = upload(h, &m, n);
        let leapfrog = || {
            let (mut x, mut v) = (x.clone(), v.clone());
            for _ in 0..steps {
                let a = forces([&x[0], &x[1], &x[2]], &mass);
                v = [0, 1, 2].map(|k| &v[k] + &a[k] * dt).to_vec();
                x = [0, 1, 2].map(|k| &x[k] + &v[k] * dt).to_vec();
            }
            let all: Vec<&Array<f32>> = x.iter().chain(&v).collect();
            eval_all(&all).unwrap();
            v
        };
        // Host reference in f64 on a sample of bodies' velocities (the forces of all bodies are needed each step).
        let mut hx: Vec<[f64; 3]> = (0..n).map(|i| [0, 1, 2].map(|k| f64::from(x0[k][i]))).collect();
        let mut hv: Vec<[f64; 3]> = (0..n).map(|i| [0, 1, 2].map(|k| f64::from(v0[k][i]))).collect();
        for _ in 0..steps {
            let xs: Vec<Vec<f32>> = (0..3).map(|k| hx.iter().map(|p| p[k] as f32).collect()).collect();
            let all: Vec<usize> = (0..n).collect();
            let a = forces_host([&xs[0], &xs[1], &xs[2]], &m, &all);
            for i in 0..n {
                for k in 0..3 {
                    hv[i][k] += a[i][k] * f64::from(dt);
                    hx[i][k] += hv[i][k] * f64::from(dt);
                }
            }
        }
        let got = leapfrog();
        let scale = hv.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs()));
        for k in 0..3 {
            let want: Vec<f64> = hv.iter().map(|v| v[k]).collect();
            check64("particles/leapfrog", &host(&got[k]), &want, 1e-3, 2e-3 * scale);
        }
        h.case("particles/leapfrog", &format!("{n} bodies x {steps} steps"), Work::Flops(20.0 * (n * n * steps) as f64), &mut || {
            leapfrog();
        });

        // Free particles bouncing in a box: a pointwise step, as one register loop and as one
        // evaluation per step.
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 16, 1 << 18) } else { pick(h, 1 << 18, 1 << 20) };
        let init = [values(n, 81, 0.0, 1.0), values(n, 82, 0.0, 1.0), values(n, 83, -1.0, 1.0), values(n, 84, -1.0, 1.0)];
        let state: Vec<Array<f32>> = init.iter().map(|d| upload(h, d, n)).collect();
        let (dt, g) = (0.01f32, 9.8f32);
        let step = |[x, y, vx, vy]: &[Array<f32>; 4]| -> [Array<f32>; 4] {
            let nx = x + vx * dt;
            let ny = y + vy * dt;
            let hit_x = nx.lt(0.0) | nx.gt(1.0);
            let vy1 = vy - g * dt;
            let hit_y = ny.lt(0.0);
            [nx.clamp(0.0, 1.0), ny.maximum(0.0), select(&hit_x, &(-vx), vx), select(&hit_y, &(&vy1 * -0.9), &vy1)]
        };
        let host_run = |steps: usize| -> Vec<Vec<f32>> {
            let mut s = init.clone();
            for _ in 0..steps {
                for i in 0..n {
                    let (x, y, vx, vy) = (s[0][i], s[1][i], s[2][i], s[3][i]);
                    let nx = x + vx * dt;
                    let ny = y + vy * dt;
                    let vy1 = vy - g * dt;
                    s[0][i] = nx.clamp(0.0, 1.0);
                    s[1][i] = ny.max(0.0);
                    s[2][i] = if !(0.0..=1.0).contains(&nx) { -vx } else { vx };
                    s[3][i] = if ny < 0.0 { vy1 * -0.9 } else { vy1 };
                }
            }
            s.to_vec()
        };
        let close = |name: &str, got: &[Array<f32>; 4], want: &[Vec<f32>]| {
            for k in 0..2 {
                let got = host(&got[k]);
                // A bounce decided differently by one ulp sends the particle elsewhere: a handful may differ.
                let bad = got.iter().zip(&want[k]).filter(|(g, w)| (*g - *w).abs() > 1e-2).count();
                assert!(bad * 500 <= n, "{name}: {bad} of {n} particles differ");
            }
        };
        let loop_steps = 100;
        let looped = || {
            let out = iterate_all([&state[0], &state[1], &state[2], &state[3]], loop_steps, step);
            eval_all(&out.each_ref()).unwrap();
            out
        };
        close("particles/integrate_loop", &looped(), &host_run(loop_steps));
        h.case("particles/integrate_loop", &format!("{n} x {loop_steps} steps"), Work::Cells((n * loop_steps) as f64), &mut || {
            looped();
        });
        let eager_steps = 50;
        let eager = || {
            let mut s = [state[0].clone(), state[1].clone(), state[2].clone(), state[3].clone()];
            for _ in 0..eager_steps {
                s = step(&s);
                eval_all(&s.each_ref()).unwrap();
            }
            s
        };
        close("particles/integrate_eager", &eager(), &host_run(eager_steps));
        h.case("particles/integrate_eager", &format!("{n} x {eager_steps} steps"), Work::Cells((n * eager_steps) as f64), &mut || {
            eager();
        });
    });
}

// -------------------------------------------------------------------------------------------------
// Random numbers and Monte Carlo
// -------------------------------------------------------------------------------------------------

fn rng(h: &mut Harness) {
    h.guarded("rng", |h| {
        let ctx = h.ctx.clone();
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 20, 1 << 22) } else { pick(h, 1 << 22, 1 << 24) };
        let nf = n as f64;
        let moments = |v: &[f32]| {
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
            let var = v.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / nf;
            (mean, var)
        };
        let u = Array::<f32>::uniform_on(&ctx, n, 7).unwrap();
        let v = host(&u);
        let (mean, var) = moments(&v);
        assert!(v.iter().all(|&x| (0.0..1.0).contains(&x)), "rng/uniform: a value outside [0, 1)");
        check_scalar("rng/uniform mean", mean, 0.5, 2e-3);
        check_scalar("rng/uniform variance", var, 1.0 / 12.0, 5e-3);
        h.case("rng/uniform", &format!("{n}"), Work::Bytes(4.0 * nf), &mut || Array::<f32>::uniform_on(&ctx, n, 7).unwrap().eval().unwrap());
        let z = Array::<f32>::normal_on(&ctx, n, 8).unwrap();
        let (mean, var) = moments(&host(&z));
        assert!(mean.abs() < 6.0 / nf.sqrt(), "rng/normal: mean {mean}");
        check_scalar("rng/normal variance", var, 1.0, 5e-3);
        h.case("rng/normal", &format!("{n}"), Work::Bytes(4.0 * nf), &mut || Array::<f32>::normal_on(&ctx, n, 8).unwrap().eval().unwrap());
        let bits = Array::<u32>::random_bits_on(&ctx, n, 9).unwrap().to_vec().unwrap();
        let mean = bits.iter().map(|&b| f64::from(b) / 4294967296.0).sum::<f64>() / nf;
        check_scalar("rng/random_bits mean", mean, 0.5, 2e-3);
        h.case("rng/random_bits", &format!("{n}"), Work::Bytes(4.0 * nf), &mut || Array::<u32>::random_bits_on(&ctx, n, 9).unwrap().eval().unwrap());
    });
}

const S0: f32 = 100.0;
const STRIKE: f32 = 100.0;
const RATE: f32 = 0.02;
const VOL: f32 = 0.3;
const MATURITY: f32 = 1.0;

/// Standard normal CDF (Abramowitz & Stegun 26.2.17), `f64` host version.
fn cnd_host(d: f64) -> f64 {
    const A: [f64; 5] = [0.319_381_53, -0.356_563_782, 1.781_477_937, -1.821_255_978, 1.330_274_429];
    let k = 1.0 / (1.0 + 0.231_641_9 * d.abs());
    let poly = k * (A[0] + k * (A[1] + k * (A[2] + k * (A[3] + k * A[4]))));
    let tail = 0.398_942_280_401_432_7 * (-0.5 * d * d).exp() * poly;
    if d > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

fn cnd(d: &Array<f32>) -> Array<f32> {
    const A: [f32; 5] = [0.319_381_54, -0.356_563_78, 1.781_477_9, -1.821_255_9, 1.330_274_5];
    let k = 1.0 / (d.abs() * 0.231_641_9 + 1.0);
    let poly = &k * (A[0] + &k * (A[1] + &k * (A[2] + &k * (A[3] + &k * A[4]))));
    let tail = (d * d * -0.5).exp() * 0.398_942_3 * poly;
    select(&d.gt(0.0), &(1.0 - &tail), &tail)
}

fn call_put(s: &Array<f32>, k: &Array<f32>, t: &Array<f32>) -> (Array<f32>, Array<f32>) {
    let vol_sqrt_t = t.sqrt() * VOL;
    let d1 = ((s / k).ln() + t * (RATE + 0.5 * VOL * VOL)) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;
    let (n1, n2) = (cnd(&d1), cnd(&d2));
    let discounted = k * (t * -RATE).exp();
    (s * &n1 - &discounted * &n2, &discounted * (1.0 - &n2) - s * (1.0 - &n1))
}

fn call_put_host(s: f32, k: f32, t: f32) -> (f64, f64) {
    let (s, k, t) = (f64::from(s), f64::from(k), f64::from(t));
    let (r, v) = (f64::from(RATE), f64::from(VOL));
    let vst = v * t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / vst;
    let (n1, n2) = (cnd_host(d1), cnd_host(d1 - vst));
    let disc = k * (-r * t).exp();
    (s * n1 - disc * n2, disc * (1.0 - n2) - s * (1.0 - n1))
}

/// The European call price by the closed formula, `f64`.
fn black_scholes_exact() -> f64 {
    call_put_host(S0, STRIKE, MATURITY).0
}

fn montecarlo(h: &mut Harness) {
    h.guarded("mc", |h| {
        let ctx = h.ctx.clone();
        let cpu = h.ctx.is_cpu();

        // pi by counting random points in the unit quarter circle: generator and reduction fused.
        let big: usize = match (cpu, h.quick) {
            (false, false) => 1 << 26,
            (false, true) => 1 << 24,
            (true, false) => 1 << 24,
            (true, true) => 1 << 22,
        };
        let mut sizes = vec![1usize << 16, 1 << 22, big];
        sizes.dedup();
        for n in sizes {
            let count = || {
                let x = Array::<f32>::uniform_on(&ctx, n, 1).unwrap();
                let y = Array::<f32>::uniform_on(&ctx, n, 2).unwrap();
                (&x * &x + &y * &y).le(1.0).count().item().unwrap()
            };
            let got = count();
            let est = 4.0 * f64::from(got) / n as f64;
            let sd = (std::f64::consts::PI * (4.0 - std::f64::consts::PI) / n as f64).sqrt();
            assert!((est - std::f64::consts::PI).abs() < 6.0 * sd, "mc/pi: estimate {est} for {n} samples");
            if n <= 1 << 22 {
                let x = host(&Array::<f32>::uniform_on(&ctx, n, 1).unwrap());
                let y = host(&Array::<f32>::uniform_on(&ctx, n, 2).unwrap());
                let want = (0..n).filter(|&i| x[i] * x[i] + y[i] * y[i] <= 1.0).count() as i64;
                assert!((i64::from(got) - want).abs() <= 4 + (n / 1_000_000) as i64, "mc/pi: counted {got}, host counts {want}");
            }
            h.case("mc/pi", &format!("{n} points"), Work::Items(n as f64, "Msamples/s"), &mut || {
                std::hint::black_box(count());
            });
        }

        // Black-Scholes prices of a batch of options (elementwise with ln, exp, sqrt).
        let sizes: Vec<usize> = match (cpu, h.quick) {
            (false, false) => vec![1 << 20, 1 << 24],
            (false, true) => vec![1 << 20],
            (true, false) => vec![1 << 18, 1 << 22],
            (true, true) => vec![1 << 18],
        };
        for n in sizes {
            let (sd, kd, td) = (values(n, 91, 80.0, 120.0), values(n, 92, 80.0, 120.0), values(n, 93, 0.25, 2.0));
            let (s, k, t) = (upload(h, &sd, n), upload(h, &kd, n), upload(h, &td, n));
            let (call, put) = call_put(&s, &k, &t);
            let (call, put) = (host(&call), host(&put));
            let sample = 4099.min(n);
            let want: Vec<(f64, f64)> = (0..sample).map(|i| call_put_host(sd[i], kd[i], td[i])).collect();
            check64("mc/black_scholes call", &call[..sample], &want.iter().map(|w| w.0).collect::<Vec<_>>(), 1e-3, 5e-3);
            check64("mc/black_scholes put", &put[..sample], &want.iter().map(|w| w.1).collect::<Vec<_>>(), 1e-3, 5e-3);
            h.case("mc/black_scholes_call", &format!("{n} options"), Work::Items(n as f64, "Mopt/s"), &mut || {
                let (call, _) = call_put(&s, &k, &t);
                call.eval().unwrap();
            });
            h.case("mc/black_scholes_call_put", &format!("{n} options"), Work::Items(n as f64, "Mopt/s"), &mut || {
                let (call, put) = call_put(&s, &k, &t);
                eval_all(&[&call, &put]).unwrap();
            });
        }

        // European call by simulating the terminal price: normals generated inside the reduction.
        let discount = (-f64::from(RATE) * f64::from(MATURITY)).exp();
        let drift = (RATE - 0.5 * VOL * VOL) * MATURITY;
        let diffusion = VOL * MATURITY.sqrt();
        let exact = black_scholes_exact();
        let sizes: Vec<usize> = if cpu { vec![1 << 20, pick(h, 1 << 20, 1 << 22)] } else { vec![1 << 20, pick(h, 1 << 22, 1 << 24)] };
        let mut sizes = sizes;
        sizes.dedup();
        for n in sizes {
            let price = || {
                let z = Array::<f32>::normal_on(&ctx, n, 3).unwrap();
                let payoff = ((&z * diffusion + drift).exp() * S0 - STRIKE).maximum(0.0);
                f64::from(payoff.mean().item().unwrap()) * discount
            };
            let got = price();
            let zs = host(&Array::<f32>::normal_on(&ctx, n, 3).unwrap());
            let want = zs.iter().map(|&z| (f64::from(S0) * (f64::from(drift) + f64::from(diffusion) * f64::from(z)).exp() - f64::from(STRIKE)).max(0.0)).sum::<f64>() / n as f64 * discount;
            check_scalar("mc/european vs host", got, want, 1e-3);
            assert!((got - exact).abs() < 6.0 * 35.0 / (n as f64).sqrt(), "mc/european: {got} against the exact {exact}");
            h.case("mc/european", &format!("{n} paths"), Work::Items(n as f64, "Mpath/s"), &mut || {
                std::hint::black_box(price());
            });
        }

        // Asian option: normals on a [steps, paths] grid, a running sum along time (contiguous
        // lines or strided lines), exponentiated and averaged along time.
        let steps = 64usize;
        let dt = MATURITY / steps as f32;
        let (a, b) = ((RATE - 0.5 * VOL * VOL) * dt, VOL * dt.sqrt());
        let paths_list: Vec<usize> = match (cpu, h.quick) {
            (false, false) => vec![1 << 18, 1 << 20],
            (false, true) => vec![1 << 16],
            (true, false) => vec![1 << 16],
            (true, true) => vec![1 << 14],
        };
        for paths in paths_list {
            for time_axis in [0usize, 1] {
                if time_axis == 1 && paths > 1 << 18 {
                    continue;
                }
                let dims = if time_axis == 0 { [steps, paths] } else { [paths, steps] };
                let price = || {
                    let z = Array::<f32>::normal_on(&ctx, dims, 4).unwrap();
                    let log_s = (&z * b + a).cumsum(time_axis);
                    let avg = (log_s.exp() * S0).sum_axis(time_axis) / steps as f32;
                    let payoff = (avg - STRIKE).maximum(0.0);
                    f64::from(payoff.mean().item().unwrap()) * discount
                };
                let got = price();
                let zs = host(&Array::<f32>::normal_on(&ctx, dims, 4).unwrap());
                let mut total = 0.0f64;
                for p in 0..paths {
                    let (mut log_s, mut sum) = (0.0f64, 0.0f64);
                    for t in 0..steps {
                        let z = if time_axis == 0 { zs[t + steps * p] } else { zs[p + paths * t] };
                        log_s += f64::from(a) + f64::from(b) * f64::from(z);
                        sum += f64::from(S0) * log_s.exp();
                    }
                    total += (sum / steps as f64 - f64::from(STRIKE)).max(0.0);
                }
                let want = total / paths as f64 * discount;
                check_scalar("mc/asian vs host", got, want, 1e-3);
                assert!(got < exact * 1.05 && got > 0.3 * exact, "mc/asian: price {got} is not that of an Asian call (European {exact})");
                let name = if time_axis == 0 { "mc/asian_time_axis0" } else { "mc/asian_time_axis1" };
                h.case(name, &format!("{paths} paths x {steps} steps"), Work::Items((paths * steps) as f64, "Mstep/s"), &mut || {
                    std::hint::black_box(price());
                });
            }
        }
    });
}

// -------------------------------------------------------------------------------------------------
// Fractals
// -------------------------------------------------------------------------------------------------

/// The escape-time counts at the sampled pixels of a `w x h` grid, from the host loop in `f32`.
fn escape_host(w: usize, h: usize, steps: usize, julia: Option<(f32, f32)>, samples: &[(usize, usize)]) -> Vec<f32> {
    samples
        .iter()
        .map(|&(i, j)| {
            let px = i as f32 * (3.2 / w as f32) - 2.2;
            let py = j as f32 * (2.4 / h as f32) - 1.2;
            let (mut zx, mut zy, c) = match julia {
                Some(c) => (px * 0.5, py * 1.3, c),
                None => (0.0, 0.0, (px, py)),
            };
            let mut count = 0.0f32;
            for _ in 0..steps {
                if zx * zx + zy * zy <= 4.0 {
                    let nx = zx * zx - zy * zy + c.0;
                    zy = zx * zy * 2.0 + c.1;
                    zx = nx;
                    count += 1.0;
                }
            }
            count
        })
        .collect()
}

fn fractal(h: &mut Harness) {
    h.guarded("fractal", |h| {
        let ctx = h.ctx.clone();
        let side: usize = if h.ctx.is_cpu() { pick(h, 512, 1024) } else { pick(h, 1024, 2048) };
        let (w, hh) = (side, side);
        let steps = 256usize;
        let grid = |axis: usize, scale: f32, origin: f32| {
            let a = Array::<f32>::range_on(&ctx, [w, hh], axis).unwrap() * (scale / if axis == 0 { w } else { hh } as f32) + origin;
            a.eval().unwrap();
            a
        };
        let (px, py) = (grid(0, 3.2, -2.2), grid(1, 2.4, -1.2));
        let zero = Array::<f32>::zeros_on(&ctx, [w, hh]).unwrap();
        let samples: Vec<(usize, usize)> = (0..w).step_by(37).flat_map(|i| (0..hh).step_by(41).map(move |j| (i, j))).collect();
        let sampled = |count: &Array<f32>| -> Vec<f32> {
            let all = host(count);
            samples.iter().map(|&(i, j)| all[i + w * j]).collect()
        };
        let close = |name: &str, got: Vec<f32>, want: Vec<f32>| {
            // Orbits near the set's boundary are chaotic, so rounding decides a few escape counts.
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert!(bad * 50 <= got.len(), "{name}: {bad} of {} escape counts differ", got.len());
        };
        let size = format!("{w}x{hh} x {steps} iterations");
        let work = Work::Cells((w * hh * steps) as f64);

        let mandelbrot = |steps: usize| {
            let [_, _, count] = iterate_all([&zero, &zero, &zero], steps, |[zx, zy, count]| {
                let inside = (zx * zx + zy * zy).le(4.0);
                let nzx = zx * zx - zy * zy + &px;
                let nzy = zx * zy * 2.0 + &py;
                [select(&inside, &nzx, zx), select(&inside, &nzy, zy), count + select(&inside, 1.0, 0.0)]
            });
            count.eval().unwrap();
            count
        };
        close("fractal/mandelbrot", sampled(&mandelbrot(steps)), escape_host(w, hh, steps, None, &samples));
        h.case("fractal/mandelbrot", &size, work, &mut || {
            mandelbrot(steps);
        });

        let c = (-0.8f32, 0.156f32);
        let (jx, jy) = (&px * 0.5, &py * 1.3);
        jx.eval().unwrap();
        jy.eval().unwrap();
        let julia = |steps: usize| {
            let [_, _, count] = iterate_all([&jx, &jy, &zero], steps, |[zx, zy, count]| {
                let inside = (zx * zx + zy * zy).le(4.0);
                let nzx = zx * zx - zy * zy + c.0;
                let nzy = zx * zy * 2.0 + c.1;
                [select(&inside, &nzx, zx), select(&inside, &nzy, zy), count + select(&inside, 1.0, 0.0)]
            });
            count.eval().unwrap();
            count
        };
        close("fractal/julia", sampled(&julia(steps)), escape_host(w, hh, steps, Some(c), &samples));
        h.case("fractal/julia", &size, work, &mut || {
            julia(steps);
        });

        // The same iteration as an unrolled Rust loop: one fused expression of `unrolled` steps.
        let unrolled_steps = 24usize;
        let unrolled = || {
            let (mut zx, mut zy, mut count) = (zero.clone(), zero.clone(), zero.clone());
            for _ in 0..unrolled_steps {
                let inside = (&zx * &zx + &zy * &zy).le(4.0);
                let nzx = &zx * &zx - &zy * &zy + &px;
                let nzy = &zx * &zy * 2.0 + &py;
                count = &count + select(&inside, 1.0, 0.0);
                zx = select(&inside, &nzx, &zx);
                zy = select(&inside, &nzy, &zy);
            }
            count.eval().unwrap();
            count
        };
        close("fractal/mandelbrot_unrolled", sampled(&unrolled()), escape_host(w, hh, unrolled_steps, None, &samples));
        h.case("fractal/mandelbrot_unrolled", &format!("{w}x{hh} x {unrolled_steps} iterations"), Work::Cells((w * hh * unrolled_steps) as f64), &mut || {
            unrolled();
        });
    });
}

// -------------------------------------------------------------------------------------------------
// Integer work
// -------------------------------------------------------------------------------------------------

fn integers(h: &mut Harness) {
    h.guarded("int", |h| {
        let ctx = h.ctx.clone();
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 20, 1 << 22) } else { pick(h, 1 << 22, 1 << 24) };
        let seeds = ints(n, 101, u32::MAX);
        let x = Array::<u32>::from_slice_on(&ctx, &seeds, n).unwrap();
        x.eval().unwrap();

        // Eight rounds of xorshift32: shifts and xors, 48 operations fused.
        let rounds = 8;
        let xorshift = |x: &Array<u32>| {
            let mut z = x.clone();
            for _ in 0..rounds {
                z = &z ^ (&z << 13u32);
                z = &z ^ (&z >> 17u32);
                z = &z ^ (&z << 5u32);
            }
            z
        };
        let want: Vec<u32> = seeds
            .iter()
            .map(|&s| {
                let mut z = s;
                for _ in 0..rounds {
                    z ^= z << 13;
                    z ^= z >> 17;
                    z ^= z << 5;
                }
                z
            })
            .collect();
        assert_eq!(xorshift(&x).to_vec().unwrap(), want, "int/xorshift");
        h.case("int/xorshift8", &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || xorshift(&x).eval().unwrap());

        // Modular arithmetic as in lattice cryptography: z = (z * b + c) mod 3329.
        const Q: u32 = 3329;
        let (b_data, c_data) = (ints(n, 102, Q), ints(n, 103, Q));
        let z0: Vec<u32> = seeds.iter().map(|&s| s % Q).collect();
        let (z, b, c) = (
            Array::<u32>::from_slice_on(&ctx, &z0, n).unwrap(),
            Array::<u32>::from_slice_on(&ctx, &b_data, n).unwrap(),
            Array::<u32>::from_slice_on(&ctx, &c_data, n).unwrap(),
        );
        let modmul = |z: &Array<u32>| {
            let mut z = z.clone();
            for _ in 0..8 {
                z = (&z * &b + &c) % Q;
            }
            z
        };
        let want: Vec<u32> = (0..n)
            .map(|i| {
                let mut v = z0[i];
                for _ in 0..8 {
                    v = (v * b_data[i] + c_data[i]) % Q;
                }
                v
            })
            .collect();
        assert_eq!(modmul(&z).to_vec().unwrap(), want, "int/modmul");
        h.case("int/modmul8", &format!("{n}"), Work::Bytes(16.0 * n as f64), &mut || modmul(&z).eval().unwrap());

        // Scatter-add: few bins (heavy contention) and many bins.
        for bins in [256usize, 1 << 16] {
            let keys = ints(n, 104, bins as u32);
            let idx = Array::<u32>::from_slice_on(&ctx, &keys, n).unwrap();
            idx.eval().unwrap();
            let mut want = vec![0u32; bins];
            for &k in &keys {
                want[k as usize] += 1;
            }
            assert_eq!(bincount(&idx, bins).to_vec().unwrap(), want, "int/bincount");
            h.case(&format!("int/bincount_{bins}"), &format!("{n} keys"), Work::Items(n as f64, "Mkey/s"), &mut || bincount(&idx, bins).eval().unwrap());
        }
        let data = values(n, 105, 0.0, 1.0);
        let xs = upload(h, &data, n);
        let mut want = vec![0u32; 256];
        for &v in &data {
            want[((v * 256.0) as usize).min(255)] += 1;
        }
        assert_eq!(xs.histogram(256, 0.0, 1.0).to_vec().unwrap(), want, "int/histogram");
        h.case("int/histogram_f32", &format!("{n} values, 256 bins"), Work::Items(n as f64, "Mval/s"), &mut || xs.histogram(256, 0.0, 1.0).eval().unwrap());
    });
}
