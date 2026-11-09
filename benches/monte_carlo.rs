//! Monte Carlo workloads, Luma against the same work on every CPU core.
//!
//! `cargo bench --bench monte_carlo` (add `-- <filter>` for one group). Build with
//! `RUSTFLAGS="-C target-cpu=native"`. Both sides use the same counter-based generator (the PCG
//! hashes of Jarzynski & Olano, "Hash Functions for GPU Rendering"), so a path or lattice site
//! sees identical random numbers on the GPU and the CPU. The CPU side processes 8 paths at a
//! time in straight-line code with its own polynomial `ln`/`exp`/`sin`/`cos`, because the
//! libm calls would not vectorize. Pricing and integration groups are checked statistically
//! against exact answers and against each other; the Ising group, whose only floating point
//! operations are exact, is checked for identical lattices.

use std::hint::black_box;
use std::ops::Range;
use std::sync::Barrier;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use luma::*;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Sums the `(sum, sum of squares)` of `f` over contiguous runs of `0..units`, one run per thread.
fn par_sum(units: usize, f: impl Fn(Range<usize>) -> (f64, f64) + Sync) -> (f64, f64) {
    let per = units.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..units.div_ceil(per))
            .map(|t| {
                let f = &f;
                s.spawn(move || f(t * per..((t + 1) * per).min(units)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    })
}

// -- Counter-based generator --

const LANES: usize = 8;

const PCG: &str = r#"
fn pcg2d(p: vec2<u32>) -> vec2<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}
fn pcg4d(p: vec4<u32>) -> vec4<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}
// Uniform in (0, 1) from 23 bits, so every value is exact in f32 and never 0 or 1.
fn u01(h: u32) -> f32 { return (f32(h >> 9u) + 0.5) * 1.1920928955078125e-7; }
"#;

const M: u32 = 1664525;
const C: u32 = 1013904223;

#[inline(always)]
fn pcg2d(x: u32, y: u32) -> (u32, u32) {
    let mut x = x.wrapping_mul(M).wrapping_add(C);
    let mut y = y.wrapping_mul(M).wrapping_add(C);
    x = x.wrapping_add(y.wrapping_mul(M));
    y = y.wrapping_add(x.wrapping_mul(M));
    x ^= x >> 16;
    y ^= y >> 16;
    x = x.wrapping_add(y.wrapping_mul(M));
    y = y.wrapping_add(x.wrapping_mul(M));
    (x ^ (x >> 16), y ^ (y >> 16))
}

#[inline(always)]
fn pcg4d(x: u32, y: u32, z: u32, w: u32) -> [u32; 4] {
    let [mut x, mut y, mut z, mut w] = [x, y, z, w].map(|v| v.wrapping_mul(M).wrapping_add(C));
    x = x.wrapping_add(y.wrapping_mul(w));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    w = w.wrapping_add(y.wrapping_mul(z));
    let [mut x, mut y, mut z, mut w] = [x, y, z, w].map(|v| v ^ (v >> 16));
    x = x.wrapping_add(y.wrapping_mul(w));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    w = w.wrapping_add(y.wrapping_mul(z));
    [x, y, z, w]
}

#[inline(always)]
fn u01(h: u32) -> f32 {
    ((h >> 9) as f32 + 0.5) * (1.0 / 8388608.0)
}

// -- Branch-free scalar math (Cephes polynomials) that LLVM vectorizes across lanes --

/// Natural log of a positive normal `x`.
#[inline(always)]
fn ln(x: f32) -> f32 {
    const SQRT_HALF: f32 = 0.707_106_77;
    let bits = x.to_bits();
    let e = ((bits >> 23) as i32) - 126;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let low = m < SQRT_HALF;
    let e = (e - low as i32) as f32;
    let x = if low { m + m - 1.0 } else { m - 1.0 };
    let z = x * x;
    let p = 7.037_683_6e-2f32;
    let p = p.mul_add(x, -1.151_461e-1);
    let p = p.mul_add(x, 1.167_699_9e-1);
    let p = p.mul_add(x, -1.242_014_1e-1);
    let p = p.mul_add(x, 1.424_932_3e-1);
    let p = p.mul_add(x, -1.666_805_8e-1);
    let p = p.mul_add(x, 2.000_071_5e-1);
    let p = p.mul_add(x, -2.499_999_4e-1);
    let p = p.mul_add(x, 3.333_333_2e-1);
    let y = p * x * z + e * -2.121_944_4e-4 - 0.5 * z;
    x + y + e * 0.693_359_4
}

#[inline(always)]
fn exp(x: f32) -> f32 {
    let x = x.max(-87.0).min(87.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = (-n).mul_add(0.693_359_4, x);
    let r = (-n).mul_add(-2.121_944_4e-4, r);
    let p = 1.987_569_1e-4f32;
    let p = p.mul_add(r, 1.398_199_9e-3);
    let p = p.mul_add(r, 8.333_452e-3);
    let p = p.mul_add(r, 4.166_579_6e-2);
    let p = p.mul_add(r, 1.666_666_5e-1);
    let p = p.mul_add(r, 5.000_000_1e-1);
    let y = (p * (r * r)) + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// `(sin, cos)` of `2 pi u` for `u` in `[0, 1)`: reduce to a quarter turn, then the quadrant.
#[inline(always)]
fn sincos_turns(u: f32) -> (f32, f32) {
    let q = u * 4.0;
    let k = (q + 0.5) as i32;
    let r = (q - k as f32) * std::f32::consts::FRAC_PI_2;
    let z = r * r;
    let s = (r * z).mul_add(
        (-1.951_529_6e-4f32.mul_add(z, 8.332_161e-3)).mul_add(z, -1.666_665_5e-1),
        r,
    );
    let c = (z * z).mul_add(
        (2.443_315_7e-5f32.mul_add(z, -1.388_731_6e-3)).mul_add(z, 4.166_664_6e-2),
        (-0.5f32).mul_add(z, 1.0),
    );
    let (a, b) = if k & 1 != 0 { (c, s) } else { (s, c) };
    let sin = if k & 2 != 0 { -a } else { a };
    let cos = if (k + 1) & 2 != 0 { -b } else { b };
    (sin, cos)
}

/// Two independent standard normals from two uniforms in `(0, 1)`.
#[inline(always)]
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * ln(u1)).sqrt();
    let (s, c) = sincos_turns(u2);
    (r * c, r * s)
}

// -- Estimates and the exact answers they are checked against --

struct Estimate {
    mean: f64,
    std_err: f64,
}

fn estimate(sum: f64, sum_sq: f64, n: usize) -> Estimate {
    let n = n as f64;
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    Estimate { mean, std_err: (var / n).sqrt() }
}

/// Both estimates use the same random numbers, so they must agree far more closely than their
/// standard error; the exact value (if known) is checked against that error.
fn check(name: &str, gpu: &Estimate, cpu: &Estimate, exact: Option<f64>) {
    let se = gpu.std_err.max(cpu.std_err);
    eprintln!(
        "{name}: gpu {:.6} cpu {:.6} se {:.2e} exact {}",
        gpu.mean,
        cpu.mean,
        se,
        exact.map_or("n/a".into(), |e| format!("{e:.6}"))
    );
    assert!((gpu.mean - cpu.mean).abs() <= 3.0 * se, "{name}: GPU and CPU estimates differ");
    if let Some(exact) = exact {
        assert!((gpu.mean - exact).abs() <= 4.0 * se, "{name}: GPU estimate is not within 4 SE of {exact}");
        assert!((cpu.mean - exact).abs() <= 4.0 * se, "{name}: CPU estimate is not within 4 SE of {exact}");
    }
}

/// Standard normal CDF (Abramowitz & Stegun 26.2.17, error below 1e-7).
fn norm_cdf(x: f64) -> f64 {
    let k = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
    let tail = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt() * poly;
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

// -- European and Asian call options under geometric Brownian motion --

const S0: f64 = 100.0;
const STRIKE: f64 = 100.0;
const RATE: f64 = 0.05;
const VOL: f64 = 0.2;
const EXPIRY: f64 = 1.0;

fn black_scholes_call() -> f64 {
    let sd = VOL * EXPIRY.sqrt();
    let d1 = ((S0 / STRIKE).ln() + (RATE + 0.5 * VOL * VOL) * EXPIRY) / sd;
    S0 * norm_cdf(d1) - STRIKE * (-RATE * EXPIRY).exp() * norm_cdf(d1 - sd)
}

/// Discretely monitored geometric-average Asian call; the arithmetic one is bounded below by it.
fn geometric_asian_call(steps: usize) -> f64 {
    let n = steps as f64;
    let mean = S0.ln() + (RATE - 0.5 * VOL * VOL) * EXPIRY * (n + 1.0) / (2.0 * n);
    let var = VOL * VOL * EXPIRY * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    let sd = var.sqrt();
    let d1 = (mean + var - STRIKE.ln()) / sd;
    (-RATE * EXPIRY).exp() * ((mean + 0.5 * var).exp() * norm_cdf(d1) - STRIKE * norm_cdf(d1 - sd))
}

/// Per-workgroup partial sums of a value and its square; the host sums the partials.
const REDUCE: &str = r#"
@group(1) @binding(0) var<storage, read_write> sums: array<f32>;
@group(1) @binding(1) var<storage, read_write> squares: array<f32>;
var<workgroup> wa: array<f32, 256>;
var<workgroup> wb: array<f32, 256>;
fn reduce_to(lid: u32, wg: u32, a: f32, b: f32) {
    wa[lid] = a;
    wb[lid] = b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            wa[lid] += wa[lid + s];
            wb[lid] += wb[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        sums[wg] = wa[0];
        squares[wg] = wb[0];
    }
}
const TWO_PI: f32 = 6.2831855;
"#;

const EUROPEAN: &str = r#"
struct Params { s0: f32, strike: f32, drift: f32, vol: f32, discount: f32, seed: u32, per_thread: u32 }
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) g: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < p.per_thread; i = i + 1u) {
        let h = pcg2d(vec2<u32>(g.x * p.per_thread + i, p.seed));
        let z = sqrt(-2.0 * log(u01(h.x))) * cos(TWO_PI * u01(h.y));
        let st = p.s0 * exp(p.drift + p.vol * z);
        let pay = p.discount * max(st - p.strike, 0.0);
        a += pay;
        b += pay * pay;
    }
    reduce_to(lid, wg.x, a, b);
}
"#;

/// Each hash gives two normals, i.e. two time steps.
const ASIAN: &str = r#"
struct Params { s0: f32, strike: f32, drift: f32, vol: f32, discount: f32, seed: u32, pairs: u32, inv_steps: f32, per_thread: u32 }
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) g: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    var a = 0.0;
    var b = 0.0;
    for (var q = 0u; q < p.per_thread; q = q + 1u) {
        let path = g.x * p.per_thread + q;
        var s = p.s0;
        var acc = 0.0;
        for (var j = 0u; j < p.pairs; j = j + 1u) {
            let h = pcg2d(vec2<u32>(path, p.seed + j));
            let r = sqrt(-2.0 * log(u01(h.x)));
            let angle = TWO_PI * u01(h.y);
            s = s * exp(p.drift + p.vol * (r * cos(angle)));
            acc += s;
            s = s * exp(p.drift + p.vol * (r * sin(angle)));
            acc += s;
        }
        let pay = p.discount * max(acc * p.inv_steps - p.strike, 0.0);
        a += pay;
        b += pay * pay;
    }
    reduce_to(lid, wg.x, a, b);
}
"#;

#[derive(encase::ShaderType)]
struct EuropeanParams {
    s0: f32,
    strike: f32,
    drift: f32,
    vol: f32,
    discount: f32,
    seed: u32,
    per_thread: u32,
}

#[derive(encase::ShaderType)]
struct AsianParams {
    s0: f32,
    strike: f32,
    drift: f32,
    vol: f32,
    discount: f32,
    seed: u32,
    pairs: u32,
    inv_steps: f32,
    per_thread: u32,
}

const EUROPEAN_SEED: u32 = 0x9E37_79B9;
const ASIAN_SEED: u32 = 0x85EB_CA6B;
const WORKGROUP: usize = 256;
/// Paths per CPU work unit: 64 groups of 8 lanes, accumulated in f32 lanes and flushed to f64.
const PATH_UNIT: usize = 512;

fn european_cpu(units: Range<usize>, p: &EuropeanParams) -> (f64, f64) {
    let (mut sum, mut sq) = (0.0f64, 0.0f64);
    for unit in units {
        let (mut a, mut b) = ([0.0f32; LANES], [0.0f32; LANES]);
        for group in 0..PATH_UNIT / LANES {
            let base = (unit * PATH_UNIT + group * LANES) as u32;
            for l in 0..LANES {
                let (h0, h1) = pcg2d(base + l as u32, p.seed);
                let z = box_muller(u01(h0), u01(h1)).0;
                let st = p.s0 * exp(p.drift + p.vol * z);
                let pay = p.discount * (st - p.strike).max(0.0);
                a[l] += pay;
                b[l] += pay * pay;
            }
        }
        sum += a.iter().map(|&v| f64::from(v)).sum::<f64>();
        sq += b.iter().map(|&v| f64::from(v)).sum::<f64>();
    }
    (sum, sq)
}

fn asian_cpu(units: Range<usize>, p: &AsianParams) -> (f64, f64) {
    let (mut sum, mut sq) = (0.0f64, 0.0f64);
    for unit in units {
        let (mut a, mut b) = ([0.0f32; LANES], [0.0f32; LANES]);
        for group in 0..PATH_UNIT / LANES {
            let base = (unit * PATH_UNIT + group * LANES) as u32;
            let mut s = [p.s0; LANES];
            let mut acc = [0.0f32; LANES];
            for j in 0..p.pairs {
                for l in 0..LANES {
                    let (h0, h1) = pcg2d(base + l as u32, p.seed + j);
                    let (z1, z0) = box_muller(u01(h0), u01(h1));
                    s[l] *= exp(p.drift + p.vol * z0);
                    acc[l] += s[l];
                    s[l] *= exp(p.drift + p.vol * z1);
                    acc[l] += s[l];
                }
            }
            for l in 0..LANES {
                let pay = p.discount * (acc[l] * p.inv_steps - p.strike).max(0.0);
                a[l] += pay;
                b[l] += pay * pay;
            }
        }
        sum += a.iter().map(|&v| f64::from(v)).sum::<f64>();
        sq += b.iter().map(|&v| f64::from(v)).sum::<f64>();
    }
    (sum, sq)
}

fn gpu_estimate(sums: &Array<f32>, squares: &Array<f32>, n: usize) -> Estimate {
    estimate(f64::from(sums.sum().unwrap()), f64::from(squares.sum().unwrap()), n)
}

fn european(c: &mut Criterion) {
    let paths = 16usize << 20;
    let per_thread = 256u32;
    let threads_gpu = paths / per_thread as usize;
    let p = EuropeanParams {
        s0: S0 as f32,
        strike: STRIKE as f32,
        drift: ((RATE - 0.5 * VOL * VOL) * EXPIRY) as f32,
        vol: (VOL * EXPIRY.sqrt()) as f32,
        discount: (-RATE * EXPIRY).exp() as f32,
        seed: EUROPEAN_SEED,
        per_thread,
    };
    let kernel = Kernel::new(&format!("{PCG}{REDUCE}{EUROPEAN}")).unwrap();
    let mut sums = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut squares = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut run = || {
        kernel
            .launch([threads_gpu as u32], args![&mut sums, &mut squares, uniform(&p)])
            .unwrap();
        gpu_estimate(&sums, &squares, paths)
    };
    let cpu = || {
        let (s, q) = par_sum(paths / PATH_UNIT, |u| european_cpu(u, &p));
        estimate(s, q, paths)
    };
    check("european", &run(), &cpu(), Some(black_scholes_call()));

    let mut group = c.benchmark_group("european_option_16M_paths");
    group.throughput(Throughput::Elements(paths as u64));
    group.bench_function("luma", |b| b.iter(|| black_box(run())));
    group.bench_function("cpu", |b| b.iter(|| black_box(cpu())));
    group.finish();
}

fn asian(c: &mut Criterion) {
    let paths = 1usize << 20;
    let steps = 64usize;
    let dt = EXPIRY / steps as f64;
    let p = AsianParams {
        s0: S0 as f32,
        strike: STRIKE as f32,
        drift: ((RATE - 0.5 * VOL * VOL) * dt) as f32,
        vol: (VOL * dt.sqrt()) as f32,
        discount: (-RATE * EXPIRY).exp() as f32,
        seed: ASIAN_SEED,
        pairs: (steps / 2) as u32,
        inv_steps: (1.0 / steps as f64) as f32,
        per_thread: 16,
    };
    let kernel = Kernel::new(&format!("{PCG}{REDUCE}{ASIAN}")).unwrap();
    let threads_gpu = paths / p.per_thread as usize;
    let mut sums = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut squares = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut run = || {
        kernel
            .launch([threads_gpu as u32], args![&mut sums, &mut squares, uniform(&p)])
            .unwrap();
        gpu_estimate(&sums, &squares, paths)
    };
    let cpu = || {
        let (s, q) = par_sum(paths / PATH_UNIT, |u| asian_cpu(u, &p));
        estimate(s, q, paths)
    };
    let (gpu, cpu_est) = (run(), cpu());
    // No closed form for the arithmetic average: it lies between the geometric Asian and the
    // European call.
    let (lo, hi) = (geometric_asian_call(steps), black_scholes_call());
    let se = gpu.std_err;
    assert!(gpu.mean + 4.0 * se >= lo && gpu.mean - 4.0 * se <= hi, "asian: {} outside [{lo}, {hi}]", gpu.mean);
    check("asian", &gpu, &cpu_est, None);

    let mut group = c.benchmark_group("asian_option_1M_paths_64_steps");
    group.throughput(Throughput::Elements((paths * steps) as u64));
    group.bench_function("luma", |b| b.iter(|| black_box(run())));
    group.bench_function("cpu", |b| b.iter(|| black_box(cpu())));
    group.finish();
}

// -- 4-D integral of exp(-|x|^2) over the unit hypercube --

const INTEGRAL: &str = r#"
struct Params { seed: u32, per_thread: u32 }
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) g: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    var a = 0.0;
    var b = 0.0;
    for (var c = 0u; c < p.per_thread / 32u; c = c + 1u) {
        var ca = 0.0;
        var cb = 0.0;
        for (var k = 0u; k < 32u; k = k + 1u) {
            let n = g.x * p.per_thread + c * 32u + k;
            let h = pcg4d(vec4<u32>(n, p.seed, 0u, 0u));
            let x = u01(h.x);
            let y = u01(h.y);
            let z = u01(h.z);
            let w = u01(h.w);
            let f = exp(-(x * x + y * y + z * z + w * w));
            ca += f;
            cb += f * f;
        }
        a += ca;
        b += cb;
    }
    reduce_to(lid, wg.x, a, b);
}
"#;

#[derive(encase::ShaderType)]
struct IntegralParams {
    seed: u32,
    per_thread: u32,
}

const INTEGRAL_SEED: u32 = 0xC2B2_AE35;
/// Samples per CPU work unit: 32 groups of 8 lanes.
const SAMPLE_UNIT: usize = 256;

fn integral_cpu(units: Range<usize>, seed: u32) -> (f64, f64) {
    let (mut sum, mut sq) = (0.0f64, 0.0f64);
    for unit in units {
        let (mut a, mut b) = ([0.0f32; LANES], [0.0f32; LANES]);
        for group in 0..SAMPLE_UNIT / LANES {
            let base = (unit * SAMPLE_UNIT + group * LANES) as u32;
            for l in 0..LANES {
                let [x, y, z, w] = pcg4d(base + l as u32, seed, 0, 0).map(u01);
                let f = exp(-(x * x + y * y + z * z + w * w));
                a[l] += f;
                b[l] += f * f;
            }
        }
        sum += a.iter().map(|&v| f64::from(v)).sum::<f64>();
        sq += b.iter().map(|&v| f64::from(v)).sum::<f64>();
    }
    (sum, sq)
}

fn integral(c: &mut Criterion) {
    let samples = 1usize << 28;
    let per_thread = 2048u32;
    let threads_gpu = samples / per_thread as usize;
    let p = IntegralParams { seed: INTEGRAL_SEED, per_thread };
    let kernel = Kernel::new(&format!("{PCG}{REDUCE}{INTEGRAL}")).unwrap();
    let mut sums = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut squares = Array::<f32>::zeros(threads_gpu / WORKGROUP).unwrap();
    let mut run = || {
        kernel
            .launch([threads_gpu as u32], args![&mut sums, &mut squares, uniform(&p)])
            .unwrap();
        gpu_estimate(&sums, &squares, samples)
    };
    let cpu = || {
        let (s, q) = par_sum(samples / SAMPLE_UNIT, |u| integral_cpu(u, INTEGRAL_SEED));
        estimate(s, q, samples)
    };
    // The integral factorizes into the fourth power of the 1-D one, int_0^1 exp(-x^2) dx.
    let exact = 0.746_824_132_812_427_1f64.powi(4);
    check("integral_4d", &run(), &cpu(), Some(exact));

    let mut group = c.benchmark_group("integral_4d_256M_samples");
    group.throughput(Throughput::Elements(samples as u64));
    group.bench_function("luma", |b| b.iter(|| black_box(run())));
    group.bench_function("cpu", |b| b.iter(|| black_box(cpu())));
    group.finish();
}

// -- 2-D Ising model, Metropolis with checkerboard updates --

/// The lattice is stored as two planes by checkerboard color; plane `c` row `j` holds the sites
/// `x = 2 i + ((c ^ j) & 1)` at index `i`. A site's vertical neighbors are then at the same `i`
/// in the other plane's adjacent rows and its horizontal neighbors at `i` and `i -+ 1` of the
/// other plane's row, so all accesses are contiguous. Spins are +-1. With J = 1 a flip costs
/// `2 s n` for neighbor sum `n`, and the only energies that need a random number are 4 and 8,
/// whose acceptance probabilities arrive precomputed (`p4`, `p8`): all arithmetic is exact, so
/// both sides evolve identical lattices.
const ISING: &str = r#"
struct Params { color: u32, tag: u32, p4: f32, p8: f32 }
@group(1) @binding(0) var<storage, read_write> mine: array<f32>;
@group(1) @binding(1) var<storage, read> other: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let hw = luma.size.x;
    let h = luma.size.y;
    if (g.x >= hw || g.y >= h) { return; }
    let i = g.x;
    let j = g.y;
    let par = (p.color ^ j) & 1u;
    let side = select((i + hw - 1u) % hw, (i + 1u) % hw, par == 1u);
    let up = (j + h - 1u) % h;
    let down = (j + 1u) % h;
    let n = other[luma_idx2(i, up)] + other[luma_idx2(i, down)]
        + other[luma_idx2(i, j)] + other[luma_idx2(side, j)];
    let idx = luma_idx2(i, j);
    let s = mine[idx];
    let e = s * n;
    let accept = select(select(1.0, p.p4, e > 1.0), p.p8, e > 3.0);
    let site = (2u * i + par) + j * (2u * hw);
    let u = u01(pcg2d(vec2<u32>(site, p.tag)).x);
    mine[idx] = select(s, -s, u < accept);
}
"#;

#[derive(encase::ShaderType)]
struct IsingParams {
    color: u32,
    tag: u32,
    p4: f32,
    p8: f32,
}

/// Temporal blocking. A workgroup loads a `TX x TY` interior plus a halo (`K` rows, one word of 8
/// sites per side) into workgroup memory, runs up to `K` half-sweeps there (each invalidates one
/// more ring of the halo) and writes the interior to the other pair of planes. Spins are one byte
/// (0 or 1) per site, four consecutive sites of a plane row per `u32` that one invocation owns
/// and updates, so tiles are four times smaller than as floats and never share a word between
/// writers. The neighbor counts of four sites come from byte-wise adds. A site flips for sure
/// unless at least 3 of its 4 neighbors agree with it, so only those draw a number: the hash is
/// `pcg2d(site, tag).x` of `PCG` with the constant tag folded, and `u01(h) < p` is the integer
/// compare `h >> 9 < t` against precomputed thresholds (exact: `u01` is exact in f32), so
/// lattices stay bit-identical to `ISING`. Tile origins are multiples of 8 and `K` is even, so a
/// tile-local `(lx, ly)` has the checkerboard color `(lx ^ ly) & 1` of its global site.
///
/// The planes in memory hold four entries per `Cell`, a `vec4<f32>` of +-1 (the layout of
/// `ISING`, so the same lattice feeds both kernels).
const ISING_BLOCKED: &str = r#"
struct Params { hw: u32, h: u32, base: u32, steps: u32, t4: u32, t8: u32 }
@group(1) @binding(0) var<storage, read> src0: array<Cell>;
@group(1) @binding(1) var<storage, read> src1: array<Cell>;
@group(1) @binding(2) var<storage, read_write> dst0: array<Cell>;
@group(1) @binding(3) var<storage, read_write> dst1: array<Cell>;
@group(1) @binding(4) var<uniform> p: Params;

const WP: u32 = TX / 8u + 2u;
const H: u32 = TY + 2u * K;
const PL: u32 = H * WP;
const M: u32 = 1664525u;
const C: u32 = 1013904223u;
// Planes by color, `PL` words each; word `w` of row `ly` holds entries `4 w .. 4 w + 3`.
var<workgroup> t: array<u32, 2u * PL>;

// `v` is below `2 n + 8`.
fn wrap(v: u32, n: u32) -> u32 {
    let r = select(v, v - n, v >= n);
    return select(r, r - n, r >= n);
}

// The flip bit (at bit `sh`) of the site whose neighbors mostly agree with it: those with 3 or 4
// agreeing neighbors (`draws`) flip when the draw is below the threshold, the others always flip
// and are set by the caller. `a` is `site * M + C + y1 * M` and `y1 = tag * M + C` (the first
// `pcg2d` mixing).
fn flips(a: u32, y1: u32, draws: u32, all: u32, sh: u32) -> u32 {
    if (((draws >> sh) & 1u) == 0u) { return 0u; }
    let b = y1 + a * M;
    let a3 = a ^ (a >> 16u);
    let b3 = b ^ (b >> 16u);
    let a4 = a3 + b3 * M;
    let q = (a4 ^ (a4 >> 16u)) >> 9u;
    return select(0u, 1u, q < select(p.t4, p.t8, ((all >> sh) & 1u) == 1u)) << sh;
}

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let hw = p.hw;
    let gw = 2u * hw;
    let hw4 = hw / 4u;
    let x0 = wid.x * TX;
    let y0 = wid.y * TY;

    for (var n = li; n < PL; n += THREADS) {
        let ly = n / WP;
        let w = n - ly * WP;
        let at = (wrap(x0 + 8u * w + gw - 8u, gw) >> 3u) + wrap(y0 + ly + p.h - K, p.h) * hw4;
        t[n] = decode(src0[at]);
        t[PL + n] = decode(src1[at]);
    }
    workgroupBarrier();

    for (var s = 0u; s < p.steps; s++) {
        let r = p.steps - 1u - s;
        let mo = (s & 1u) * PL;
        let oo = PL - mo;
        let y1 = (p.base + s) * M + C;
        let c1 = C + y1 * M;
        let items = (TY + 2u * r) * WP;
        for (var n = li; n < items; n += THREADS) {
            let ry = n / WP;
            let w = n - ry * WP;
            if (r == 0u && (w == 0u || w == WP - 1u)) { continue; }
            let ly = K - r + ry;
            let at = ly * WP + w;
            let par = (s ^ ly) & 1u;
            let mine = t[mo + at];
            let mid = t[oo + at];
            let side = select(
                (mid << 8u) | (t[oo + at - 1u] >> 24u),
                (mid >> 8u) | (t[oo + at + 1u] << 24u),
                par == 1u,
            );
            let count = t[oo + at - WP] + t[oo + at + WP] + mid + side;
            // Per byte, the number of neighbors with the site's own spin: bit 2 set means all 4
            // agree, and 3 or more is bit 2 or (bits 1 and 0).
            let own = mine * 255u;
            let agree = (count & own) | ((0x04040404u - count) & ~own);
            let all = (agree >> 2u) & 0x01010101u;
            let draws = all | ((agree >> 1u) & agree & 0x01010101u);
            let gy = wrap(y0 + ly + p.h - K, p.h);
            let site = wrap(x0 + 8u * w + gw - 8u, gw) + par + gy * gw;
            let a = site * M + c1;
            let flip = (draws ^ 0x01010101u)
                | flips(a, y1, draws, all, 0u)
                | flips(a + 2u * M, y1, draws, all, 8u)
                | flips(a + 4u * M, y1, draws, all, 16u)
                | flips(a + 6u * M, y1, draws, all, 24u);
            t[mo + at] = mine ^ flip;
        }
        workgroupBarrier();
    }

    for (var n = li; n < TY * (WP - 2u); n += THREADS) {
        let ry = n / (WP - 2u);
        let w = n - ry * (WP - 2u) + 1u;
        let at = ((x0 + 8u * w - 8u) >> 3u) + (y0 + ry) * hw4;
        let word = (K + ry) * WP + w;
        dst0[at] = encode(t[word]);
        dst1[at] = encode(t[PL + word]);
    }
}
"#;

const CELL_FLOATS: &str = r#"
alias Cell = vec4<f32>;
fn decode(v: Cell) -> u32 {
    return select(0u, 1u, v.x > 0.0) | (select(0u, 1u, v.y > 0.0) << 8u)
        | (select(0u, 1u, v.z > 0.0) << 16u) | (select(0u, 1u, v.w > 0.0) << 24u);
}
fn encode(v: u32) -> Cell {
    return vec4<f32>(f32(v & 1u), f32((v >> 8u) & 1u), f32((v >> 16u) & 1u), f32(v >> 24u)) * 2.0 - 1.0;
}
"#;

#[derive(encase::ShaderType)]
struct IsingBlockedParams {
    hw: u32,
    h: u32,
    base: u32,
    steps: u32,
    t4: u32,
    t8: u32,
}

/// Number of 23-bit draws `q` with `(q + 0.5) / 2^23 < accept`.
fn draw_threshold(accept: f32) -> u32 {
    (f64::from(accept) * 8388608.0 - 0.5).ceil().max(0.0) as u32
}

/// Interior of a workgroup's tile and its invocation count: the best measured shape whose tile
/// fits workgroup memory, which slows kernels sharply beyond about 8 KiB per workgroup.
const TILE_X: u32 = 128;
const TILE_Y: u32 = 32;
const TILE_THREADS: u32 = 128;

/// Lattice size and generator thresholds shared by the blocked launches.
#[derive(Clone, Copy)]
struct IsingShape {
    hw: u32,
    h: u32,
    half_sweeps: u32,
    t4: u32,
    t8: u32,
}

/// The planes a blocked kernel works on (`T` is a four-spin cell) and the plane pair it writes.
struct BlockedRun<T: GpuType> {
    kernel: Kernel,
    groups: [u32; 3],
    k: u32,
    state: [Array<T>; 2],
    scratch: [Array<T>; 2],
}

impl<T: GpuType> BlockedRun<T> {
    /// `k` half-sweeps per launch.
    fn new(k: u32, shape: IsingShape, state: [Array<T>; 2], scratch: [Array<T>; 2]) -> Self {
        assert!(2 * shape.hw % TILE_X == 0 && shape.h % TILE_Y == 0, "ising: tiles must cover the lattice");
        let cell = CELL_FLOATS;
        let source = format!(
            "const K: u32 = {k}u;\nconst TX: u32 = {TILE_X}u;\nconst TY: u32 = {TILE_Y}u;\n\
             const THREADS: u32 = {TILE_THREADS}u;\n{cell}{ISING_BLOCKED}"
        );
        BlockedRun {
            kernel: Kernel::new(&source).unwrap(),
            groups: [2 * shape.hw / TILE_X, shape.h / TILE_Y, 1],
            k,
            state,
            scratch,
        }
    }

    /// Runs all half-sweeps, `k` per launch. Each launch reads `state` and writes `scratch`, then
    /// they swap, so a tile never reads what a neighbor has already advanced.
    fn run(&mut self, ctx: &Context, shape: IsingShape) {
        let IsingShape { hw, h, half_sweeps, t4, t8 } = shape;
        let mut base = 0;
        while base < half_sweeps {
            let steps = self.k.min(half_sweeps - base);
            let p = IsingBlockedParams { hw, h, base, steps, t4, t8 };
            let [s0, s1] = &self.state;
            let [d0, d1] = &mut self.scratch;
            self.kernel.launch_raw(self.groups, args![s0, s1, d0, d1, uniform(&p)]).unwrap();
            std::mem::swap(&mut self.state, &mut self.scratch);
            base += steps;
        }
        ctx.sync().unwrap();
    }
}

#[inline(always)]
fn flip(s: f32, n: f32, site: u32, tag: u32, p4: f32, p8: f32) -> f32 {
    let e = s * n;
    let accept = if e > 3.0 {
        p8
    } else if e > 1.0 {
        p4
    } else {
        1.0
    };
    if u01(pcg2d(site, tag).0) < accept {
        -s
    } else {
        s
    }
}

/// Updates row `j` of the plane `mine` (color `c`) from the plane `other`.
fn ising_row(mine: &mut [f32], other: &[f32], j: usize, h: usize, c: usize, tag: u32, p4: f32, p8: f32) {
    let hw = mine.len();
    let up = &other[((j + h - 1) % h) * hw..][..hw];
    let down = &other[((j + 1) % h) * hw..][..hw];
    let mid = &other[j * hw..][..hw];
    let base = (j * 2 * hw) as u32;
    if (c ^ j) & 1 == 0 {
        mine[0] = flip(mine[0], up[0] + down[0] + mid[0] + mid[hw - 1], base, tag, p4, p8);
        for i in 1..hw {
            let n = up[i] + down[i] + mid[i] + mid[i - 1];
            mine[i] = flip(mine[i], n, base + 2 * i as u32, tag, p4, p8);
        }
    } else {
        for i in 0..hw - 1 {
            let n = up[i] + down[i] + mid[i] + mid[i + 1];
            mine[i] = flip(mine[i], n, base + 2 * i as u32 + 1, tag, p4, p8);
        }
        let i = hw - 1;
        mine[i] = flip(mine[i], up[i] + down[i] + mid[i] + mid[0], base + 2 * i as u32 + 1, tag, p4, p8);
    }
}

#[derive(Clone, Copy)]
struct Planes([*mut f32; 2]);
// SAFETY: within one half-sweep threads write disjoint rows of one plane and only read the other,
// and a barrier separates half-sweeps.
unsafe impl Send for Planes {}
unsafe impl Sync for Planes {}

impl Planes {
    // A method, so closures capture the whole `Planes` (which is `Send`), not its raw-pointer field.
    fn plane(self, c: usize) -> *mut f32 {
        self.0[c]
    }
}

/// `sweeps` Metropolis sweeps over all cores; each thread owns a band of rows, threads stay alive
/// across half-sweeps and meet at a barrier between them.
fn ising_cpu(planes: &mut [Vec<f32>; 2], hw: usize, h: usize, sweeps: usize, p4: f32, p8: f32) {
    let nt = threads().min(h);
    let per = h.div_ceil(nt);
    let nt = h.div_ceil(per);
    let shared = Planes([planes[0].as_mut_ptr(), planes[1].as_mut_ptr()]);
    let barrier = Barrier::new(nt);
    std::thread::scope(|s| {
        for t in 0..nt {
            let barrier = &barrier;
            s.spawn(move || {
                let rows = t * per..((t + 1) * per).min(h);
                for sweep in 0..sweeps {
                    for c in 0..2 {
                        // SAFETY: see `Planes`; `other` is not written during this half-sweep.
                        let other = unsafe { std::slice::from_raw_parts(shared.plane(1 - c), hw * h) };
                        for j in rows.clone() {
                            let mine = unsafe { std::slice::from_raw_parts_mut(shared.plane(c).add(j * hw), hw) };
                            ising_row(mine, other, j, h, c, (sweep * 2 + c) as u32, p4, p8);
                        }
                        barrier.wait();
                    }
                }
            });
        }
    });
}

/// Magnetization and energy per spin of the lattice stored as `planes`.
fn ising_observables(planes: &[Vec<f32>; 2], hw: usize, h: usize) -> (f64, f64) {
    let w = 2 * hw;
    let mut full = vec![0.0f32; w * h];
    for c in 0..2 {
        for j in 0..h {
            for i in 0..hw {
                full[2 * i + ((c ^ j) & 1) + j * w] = planes[c][i + j * hw];
            }
        }
    }
    let (mut m, mut e) = (0.0f64, 0.0f64);
    for j in 0..h {
        for x in 0..w {
            let s = full[x + j * w];
            m += f64::from(s);
            e -= f64::from(s * (full[(x + 1) % w + j * w] + full[x + (j + 1) % h * w]));
        }
    }
    let n = (w * h) as f64;
    (m / n, e / n)
}

fn ising(c: &mut Criterion) {
    let (hw, h, sweeps) = (1024usize, 2048usize, 50usize);
    let w = 2 * hw;
    // Slightly above the critical coupling 0.4407, so domains grow on every scale.
    let beta = 0.45f32;
    let (p4, p8) = ((-4.0 * beta).exp(), (-8.0 * beta).exp());

    let mut cpu_planes = [vec![0.0f32; hw * h], vec![0.0f32; hw * h]];
    for (color, plane) in cpu_planes.iter_mut().enumerate() {
        for j in 0..h {
            for i in 0..hw {
                let site = (2 * i + ((color ^ j) & 1) + j * w) as u32;
                plane[i + j * hw] = if pcg2d(site, 0xA5A5_A5A5).0 >> 31 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let (m0, e0) = ising_observables(&cpu_planes, hw, h);
    let upload = |planes: &[Vec<f32>; 2]| {
        [
            Array::from_slice(&planes[0], [hw, h]).unwrap(),
            Array::from_slice(&planes[1], [hw, h]).unwrap(),
        ]
    };
    let initial = cpu_planes.clone();
    let ctx = &default_context().unwrap();

    let kernel = Kernel::new(&format!("{PCG}{ISING}")).unwrap();
    let per_half_sweep = |planes: &mut [Array<f32>; 2]| {
        let [a, b] = planes;
        for sweep in 0..sweeps {
            for color in 0..2u32 {
                let p = IsingParams { color, tag: (sweep * 2) as u32 + color, p4, p8 };
                if color == 0 {
                    kernel.launch([hw as u32, h as u32], args![&mut *a, &*b, uniform(&p)]).unwrap();
                } else {
                    kernel.launch([hw as u32, h as u32], args![&mut *b, &*a, uniform(&p)]).unwrap();
                }
            }
        }
        ctx.sync().unwrap();
    };

    // The blocked kernel takes four consecutive entries of a plane row per element.
    let to_floats = |planes: &[Vec<f32>; 2]| {
        planes.each_ref().map(|plane| {
            let quads: Vec<[f32; 4]> = plane.chunks_exact(4).map(|q| [q[0], q[1], q[2], q[3]]).collect();
            Array::from_slice(&quads, [hw / 4, h]).unwrap()
        })
    };
    let from_floats = |planes: &[Array<[f32; 4]>; 2]| {
        planes.each_ref().map(|plane| plane.to_vec().unwrap().into_iter().flatten().collect::<Vec<f32>>())
    };
    let shape = IsingShape {
        hw: hw as u32,
        h: h as u32,
        half_sweeps: sweeps as u32 * 2,
        t4: draw_threshold(p4),
        t8: draw_threshold(p8),
    };

    let mut gpu_planes = upload(&initial);
    per_half_sweep(&mut gpu_planes);
    ising_cpu(&mut cpu_planes, hw, h, sweeps, p4, p8);
    let got = [gpu_planes[0].to_vec().unwrap(), gpu_planes[1].to_vec().unwrap()];
    assert!(got == cpu_planes, "ising: GPU and CPU lattices differ");

    let mut float_run = BlockedRun::new(8, shape, to_floats(&initial), to_floats(&initial));
    float_run.run(ctx, shape);
    assert!(from_floats(&float_run.state) == cpu_planes, "ising: blocked GPU and CPU lattices differ");
    let (m, e) = ising_observables(&cpu_planes, hw, h);
    eprintln!("ising: identical lattices; m {m0:.4} -> {m:.4}, E/N {e0:.4} -> {e:.4}");
    // A random lattice has E/N near 0; after 50 sweeps at beta 0.45 domains have formed.
    assert!(e < -0.8 && e0.abs() < 0.01, "ising: unphysical energy {e}");

    let mut group = c.benchmark_group("ising_2048x2048_50_sweeps");
    group.throughput(Throughput::Elements((w * h * sweeps) as u64));
    group.bench_function("luma_per_half_sweep", |b| b.iter(|| per_half_sweep(&mut gpu_planes)));
    group.bench_function("luma_blocked_k8", |b| b.iter(|| float_run.run(ctx, shape)));
    group.bench_function("cpu", |b| {
        b.iter(|| ising_cpu(black_box(&mut cpu_planes), hw, h, sweeps, p4, p8))
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = european, asian, integral, ising
}
criterion_main!(benches);
