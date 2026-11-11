//! Workloads from real applications, Luma against the same work on every CPU core.
//!
//! `cargo bench --bench real_world` (add `-- <filter>` for one group). Build with
//! `RUSTFLAGS="-C target-cpu=native"` so the CPU side gets the host's vector units. The CPU
//! baselines are parallel over all cores and written so that LLVM vectorizes their inner loops;
//! matrix multiply uses Apple's Accelerate (`cblas_sgemm`) and the FIR filter `rustfft`. Every
//! group first checks that both sides compute the same result.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use luma::*;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Runs `f(first_row, rows)` over contiguous runs of `row_len`-element rows of `out`, one run
/// per thread.
fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len() / row_len;
    let per = rows.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, chunk));
        }
    });
}

/// Deterministic values in `lo..hi`.
fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

/// Largest difference relative to the largest magnitude of `want`.
fn check(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(worst / scale <= tolerance, "{name}: GPU and CPU differ by {worst} (scale {scale})");
}

// -- Black-Scholes call prices --

const RATE: f32 = 0.02;
const VOL: f32 = 0.3;
const A: [f32; 5] = [0.319_381_54, -0.356_563_78, 1.781_477_9, -1.821_255_9, 1.330_274_5];
const INV_SQRT_2PI: f32 = 0.398_942_3;

/// Cumulative normal distribution (Abramowitz & Stegun 26.2.17).
fn cnd_luma(d: &Array<f32>) -> Array<f32> {
    let k = 1.0 / (d.abs() * 0.2316419 + 1.0);
    let poly = &k * (A[0] + &k * (A[1] + &k * (A[2] + &k * (A[3] + &k * A[4]))));
    let tail = (d * d * -0.5).exp() * INV_SQRT_2PI * poly;
    select(&d.gt(0.0), &(1.0 - &tail), &tail)
}

fn black_scholes_luma(s: &Array<f32>, k: &Array<f32>, t: &Array<f32>) -> Array<f32> {
    let vol_sqrt_t = t.sqrt() * VOL;
    let d1 = ((s / k).ln() + t * (RATE + 0.5 * VOL * VOL)) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;
    s * cnd_luma(&d1) - k * (t * -RATE).exp() * cnd_luma(&d2)
}

/// Call and put prices of the same options; they share every intermediate.
fn call_put_luma(s: &Array<f32>, k: &Array<f32>, t: &Array<f32>) -> (Array<f32>, Array<f32>) {
    let vol_sqrt_t = t.sqrt() * VOL;
    let d1 = ((s / k).ln() + t * (RATE + 0.5 * VOL * VOL)) / &vol_sqrt_t;
    let d2 = &d1 - &vol_sqrt_t;
    let (n1, n2) = (cnd_luma(&d1), cnd_luma(&d2));
    let discounted = k * (t * -RATE).exp();
    let call = s * &n1 - &discounted * &n2;
    let put = &discounted * (1.0 - &n2) - s * (1.0 - &n1);
    (call, put)
}

fn call_put_cpu(s: &[f32], k: &[f32], t: &[f32], call: &mut [f32], put: &mut [f32]) {
    std::thread::scope(|scope| {
        let per = s.len().div_ceil(threads());
        for (c, (cs, ps)) in call.chunks_mut(per).zip(put.chunks_mut(per)).enumerate() {
            scope.spawn(move || {
                for (i, (co, po)) in cs.iter_mut().zip(ps.iter_mut()).enumerate() {
                    let j = c * per + i;
                    let (s, k, t) = (s[j], k[j], t[j]);
                    let vol_sqrt_t = VOL * t.sqrt();
                    let d1 = ((s / k).ln() + (RATE + 0.5 * VOL * VOL) * t) / vol_sqrt_t;
                    let (n1, n2) = (cnd_cpu(d1), cnd_cpu(d1 - vol_sqrt_t));
                    let discounted = k * (-RATE * t).exp();
                    *co = s * n1 - discounted * n2;
                    *po = discounted * (1.0 - n2) - s * (1.0 - n1);
                }
            });
        }
    });
}

fn cnd_cpu(d: f32) -> f32 {
    let k = 1.0 / (1.0 + 0.2316419 * d.abs());
    let poly = k * (A[0] + k * (A[1] + k * (A[2] + k * (A[3] + k * A[4]))));
    let tail = INV_SQRT_2PI * (-0.5 * d * d).exp() * poly;
    if d > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

fn black_scholes_cpu(s: &[f32], k: &[f32], t: &[f32], out: &mut [f32]) {
    par_rows(out, 1, |first, chunk| {
        for (i, o) in chunk.iter_mut().enumerate() {
            let (s, k, t) = (s[first + i], k[first + i], t[first + i]);
            let vol_sqrt_t = VOL * t.sqrt();
            let d1 = ((s / k).ln() + (RATE + 0.5 * VOL * VOL) * t) / vol_sqrt_t;
            let d2 = d1 - vol_sqrt_t;
            *o = s * cnd_cpu(d1) - k * (-RATE * t).exp() * cnd_cpu(d2);
        }
    });
}

fn black_scholes(c: &mut Criterion) {
    let n = 1 << 24;
    let (s, k, t) = (values(n, 1, 10.0, 100.0), values(n, 2, 10.0, 100.0), values(n, 3, 0.1, 5.0));
    let (gs, gk, gt) =
        (Array::from_slice(&s, n).unwrap(), Array::from_slice(&k, n).unwrap(), Array::from_slice(&t, n).unwrap());
    let mut out = vec![0.0; n];
    black_scholes_cpu(&s, &k, &t, &mut out);
    check("black-scholes", &black_scholes_luma(&gs, &gk, &gt).to_vec().unwrap(), &out, 1e-4);

    let mut group = c.benchmark_group("black_scholes_16M_options");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("luma", |b| b.iter(|| black_scholes_luma(&gs, &gk, &gt).eval().unwrap()));
    group.bench_function("luma_with_transfers", |b| {
        b.iter(|| {
            let up = |v: &[f32]| Array::from_slice(v, n).unwrap();
            black_box(black_scholes_luma(&up(&s), &up(&k), &up(&t)).to_vec().unwrap())
        })
    });
    group.bench_function("cpu", |b| b.iter(|| black_scholes_cpu(&s, &k, &t, black_box(&mut out))));
    group.finish();

    let (mut call, mut put) = (vec![0.0; n], vec![0.0; n]);
    call_put_cpu(&s, &k, &t, &mut call, &mut put);
    let (gc, gp) = call_put_luma(&gs, &gk, &gt);
    eval_all(&[&gc, &gp]).unwrap();
    check("call", &gc.to_vec().unwrap(), &call, 1e-4);
    check("put", &gp.to_vec().unwrap(), &put, 1e-4);
    let mut group = c.benchmark_group("black_scholes_call_and_put_16M_options");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("luma_eval_all", |b| {
        b.iter(|| {
            let (call, put) = call_put_luma(&gs, &gk, &gt);
            eval_all(&[&call, &put]).unwrap()
        })
    });
    group.bench_function("luma_one_by_one", |b| {
        b.iter(|| {
            let (call, put) = call_put_luma(&gs, &gk, &gt);
            call.eval().unwrap();
            put.eval().unwrap()
        })
    });
    group.bench_function("cpu", |b| b.iter(|| call_put_cpu(&s, &k, &t, black_box(&mut call), black_box(&mut put))));
}

// -- Mandelbrot set --

const MANDELBROT: &str = r#"
struct Params { x0: f32, y0: f32, dx: f32, dy: f32, max_iter: u32 }
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= luma.size.x || g.y >= luma.size.y) { return; }
    let cx = p.x0 + f32(g.x) * p.dx;
    let cy = p.y0 + f32(g.y) * p.dy;
    var x = 0.0;
    var y = 0.0;
    var i = 0u;
    loop {
        if (i >= p.max_iter || x * x + y * y > 4.0) { break; }
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        i = i + 1u;
    }
    out[luma_idx2(g.x, g.y)] = i;
}
"#;

#[derive(encase::ShaderType)]
struct MandelbrotParams {
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
    max_iter: u32,
}

/// Eight pixels in lockstep (branch-free, so the lane loops vectorize) until all have escaped.
fn mandelbrot_cpu(w: usize, p: &MandelbrotParams, out: &mut [u32]) {
    par_rows(out, w, |first, rows| {
        for (r, row) in rows.chunks_mut(w).enumerate() {
            let cy = p.y0 + (first + r) as f32 * p.dy;
            for (c, lanes) in row.chunks_mut(8).enumerate() {
                let mut cx = [0.0f32; 8];
                for (l, v) in cx.iter_mut().enumerate() {
                    *v = p.x0 + (c * 8 + l) as f32 * p.dx;
                }
                let (mut x, mut y, mut count) = ([0.0f32; 8], [0.0f32; 8], [0u32; 8]);
                for _ in 0..p.max_iter {
                    let mut any = false;
                    for l in 0..8 {
                        let inside = x[l] * x[l] + y[l] * y[l] <= 4.0;
                        let xt = x[l] * x[l] - y[l] * y[l] + cx[l];
                        let yt = 2.0 * x[l] * y[l] + cy;
                        x[l] = if inside { xt } else { x[l] };
                        y[l] = if inside { yt } else { y[l] };
                        count[l] += u32::from(inside);
                        any |= inside;
                    }
                    if !any {
                        break;
                    }
                }
                lanes.copy_from_slice(&count[..lanes.len()]);
            }
        }
    });
}

fn mandelbrot(c: &mut Criterion) {
    let (w, h) = (2048usize, 2048usize);
    let p = MandelbrotParams { x0: -2.2, y0: -1.4, dx: 3.2 / w as f32, dy: 2.8 / h as f32, max_iter: 1000 };
    let kernel = Kernel::new(MANDELBROT).unwrap();
    let mut img = Array::<u32>::zeros([w, h]).unwrap();
    let ctx = default_context().unwrap();
    let run = |img: &mut Array<u32>| {
        kernel.launch([w as u32, h as u32], args![&mut *img, uniform(&p)]).unwrap();
        ctx.sync().unwrap();
    };
    run(&mut img);
    let mut out = vec![0u32; w * h];
    mandelbrot_cpu(w, &p, &mut out);
    let got = img.to_vec().unwrap();
    // FMA contraction differs between the two compilers; points on the boundary may escape at
    // another iteration.
    let differ = got.iter().zip(&out).filter(|(a, b)| a != b).count();
    assert!(differ * 100 < out.len(), "mandelbrot: {differ} of {} pixels differ", out.len());

    let mut group = c.benchmark_group("mandelbrot_2048x2048_1000_iter");
    group.throughput(Throughput::Elements((w * h) as u64));
    group.bench_function("luma", |b| b.iter(|| run(&mut img)));
    group.bench_function("cpu", |b| b.iter(|| mandelbrot_cpu(w, &p, black_box(&mut out))));
    group.finish();
}

// -- N-body gravitational accelerations --

/// Bodies held in registers by each invocation, so that one partner load from workgroup memory
/// feeds this many interactions.
const NBODY_PER_INVOCATION: usize = 8;
/// Partners staged in workgroup memory per iteration.
const NBODY_TILE: usize = 512;

/// `(bodies per workgroup, accumulators reduced per pass)`: Apple wants a whole subgroup on one slice
/// and a small workgroup footprint, NVIDIA a single reduction pass.
fn nbody_tuning(profile: TuningProfile) -> (usize, usize) {
    match profile {
        TuningProfile::Nvidia => (64, NBODY_PER_INVOCATION),
        _ => (128, 2),
    }
}

/// Each body is shared by `256 * NBODY_PER_INVOCATION / bodies` invocations that take a slice of
/// every tile of partners; their partial sums are reduced through the tile, `chunk` accumulators at a time.
fn nbody_source(bodies: usize, chunk: usize) -> String {
    let (r, tile) = (NBODY_PER_INVOCATION, NBODY_TILE);
    let (groups, slices) = (bodies / r, 256 * r / bodies);
    let each = |f: &dyn Fn(usize) -> String| (0..r).map(f).collect::<String>();
    let lets = each(&|i| {
        format!(
            r#"
    let me{i} = pos[min(first + {i}u, p.n - 1u)].xyz;
    var a{i} = vec3<f32>(0.0);"#
        )
    });
    let pairs = each(&|i| {
        format!(
            r#"
            let d{i} = o.xyz - me{i};
            let i{i} = inverseSqrt(fma(d{i}.z, d{i}.z, fma(d{i}.y, d{i}.y, fma(d{i}.x, d{i}.x, p.eps2))));
            let s{i} = o.w * i{i} * (i{i} * i{i});
            a{i} = fma(d{i}, vec3<f32>(s{i}), a{i});"#
        )
    });
    let passes: String = (0..r / chunk)
        .map(|q| {
            let (first, active) = (q * chunk, groups * chunk);
            let stores: String = (0..chunk).map(|c| format!("\n    tile[lid + {}u] = vec4<f32>(a{}, 0.0);", c * 256, first + c)).collect();
            format!(
                r#"{stores}
    workgroupBarrier();
    if (lid < {active}u) {{
        let r = lid % {chunk}u;
        let g = lid / {chunk}u;
        var sum = vec4<f32>(0.0);
        for (var s = 0u; s < {slices}u; s = s + 1u) {{
            sum = sum + tile[r * 256u + s * {groups}u + g];
        }}
        let body = first0 + g * {r}u + {first}u + r;
        if (body < p.n) {{
            acc[body] = sum;
        }}
    }}
    workgroupBarrier();
"#
            )
        })
        .collect();
    format!(
        r#"
struct Params {{ n: u32, eps2: f32 }}
@group(1) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> acc: array<vec4<f32>>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> tile: array<vec4<f32>, {size}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let slot = lid % {groups}u;
    let slice = lid / {groups}u;
    let first0 = wid.x * {bodies}u;
    let first = first0 + slot * {r}u;{lets}
    for (var base = 0u; base < p.n; base = base + {tile}u) {{
        for (var t = lid; t < {tile}u; t = t + 256u) {{
            let j = base + t;
            // Missing partners have no mass.
            tile[t] = select(vec4<f32>(0.0), pos[min(j, p.n - 1u)], j < p.n);
        }}
        workgroupBarrier();
        for (var k = slice; k < {tile}u; k = k + {slices}u) {{
            let o = tile[k];{pairs}
        }}
        workgroupBarrier();
    }}
{passes}}}
"#,
        size = tile.max(256 * chunk)
    )
}

#[derive(encase::ShaderType)]
struct NbodyParams {
    n: u32,
    eps2: f32,
}

const EPS2: f32 = 1e-4;

/// Accelerations from bodies in structure-of-arrays form `[x, y, z, m]`, 8 partners at a time.
fn nbody_cpu(soa: &[Vec<f32>; 4], out: &mut [[f32; 3]]) {
    let [px, py, pz, pm] = soa;
    par_rows(out, 1, |first, chunk| {
        for (i, o) in chunk.iter_mut().enumerate() {
            let (xi, yi, zi) = (px[first + i], py[first + i], pz[first + i]);
            let (mut ax, mut ay, mut az) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
            let chunks = px.as_chunks::<8>().0.iter().zip(py.as_chunks::<8>().0).zip(pz.as_chunks::<8>().0).zip(pm.as_chunks::<8>().0);
            for (((x, y), z), m) in chunks {
                for l in 0..8 {
                    let (dx, dy, dz) = (x[l] - xi, y[l] - yi, z[l] - zi);
                    let r2 = dx * dx + dy * dy + dz * dz + EPS2;
                    let s = m[l] / (r2 * r2.sqrt());
                    ax[l] += dx * s;
                    ay[l] += dy * s;
                    az[l] += dz * s;
                }
            }
            *o = [ax.iter().sum(), ay.iter().sum(), az.iter().sum()];
        }
    });
}

fn nbody(c: &mut Criterion) {
    let n = 32768usize;
    let soa = [values(n, 4, -1.0, 1.0), values(n, 5, -1.0, 1.0), values(n, 6, -1.0, 1.0), values(n, 7, 0.5, 1.5)];
    let aos: Vec<[f32; 4]> = (0..n).map(|i| [soa[0][i], soa[1][i], soa[2][i], soa[3][i]]).collect();
    let pos = Array::from_slice(&aos, n).unwrap();
    let mut acc = Array::<[f32; 4]>::from_slice(&vec![[0.0; 4]; n], n).unwrap();
    let params = NbodyParams { n: n as u32, eps2: EPS2 };
    let ctx = default_context().unwrap();
    let (bodies, chunk) = nbody_tuning(ctx.tuning_profile());
    let kernel = Kernel::new(&nbody_source(bodies, chunk)).unwrap();
    let run = |acc: &mut Array<[f32; 4]>| {
        kernel.launch_raw([n.div_ceil(bodies) as u32, 1, 1], args![&pos, &mut *acc, uniform(&params)]).unwrap();
        ctx.sync().unwrap();
    };
    run(&mut acc);
    let mut out = vec![[0.0f32; 3]; n];
    nbody_cpu(&soa, &mut out);
    let got: Vec<f32> = acc.to_vec().unwrap().iter().flat_map(|a| [a[0], a[1], a[2]]).collect();
    let want: Vec<f32> = out.iter().flatten().copied().collect();
    check("n-body", &got, &want, 1e-3);

    let mut group = c.benchmark_group("nbody_32768_bodies");
    group.throughput(Throughput::Elements((n * n) as u64));
    group.bench_function("luma", |b| b.iter(|| run(&mut acc)));
    group.bench_function("cpu", |b| b.iter(|| nbody_cpu(&soa, black_box(&mut out))));
    group.finish();
}

// -- Matrix multiply --

#[path = "common/sgemm.rs"]
mod sgemm;
use sgemm::{sgemm_source, CoopKind, SgemmParams, TILE};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
fn sgemm_cpu(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    let n = n as i32;
    // SAFETY: the slices hold n * n elements each, as the leading dimensions state.
    unsafe {
        cblas_sgemm(ROW_MAJOR, NO_TRANS, NO_TRANS, n, n, n, 1.0, a.as_ptr(), n, b.as_ptr(), n, 0.0, c.as_mut_ptr(), n)
    };
}

fn sgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm");
    for n in [1024usize, 2048, 4096] {
        let (a, b) = (values(n * n, 8, -1.0, 1.0), values(n * n, 9, -1.0, 1.0));
        let (ga, gb) = (Array::from_slice(&a, [n, n]).unwrap(), Array::from_slice(&b, [n, n]).unwrap());
        let mut gc = Array::<f32>::zeros([n, n]).unwrap();
        let kernel = Kernel::new(&sgemm_source(false, false)).unwrap();
        let params = SgemmParams { m: n as u32, n: n as u32, k: n as u32, relu: 0 };
        let ctx = default_context().unwrap();
        let tiles = (n / TILE) as u32;
        let run = |gc: &mut Array<f32>| {
            kernel.launch_raw([tiles, tiles, 1], args![&ga, &gb, &mut *gc, uniform(&params)]).unwrap();
            ctx.sync().unwrap();
        };
        run(&mut gc);
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        group.bench_function(BenchmarkId::new("luma", n), |bch| bch.iter(|| run(&mut gc)));
        let f32_result = gc.to_vec().unwrap();
        if let Some(kind) = CoopKind::for_context(&ctx) {
            let coop = Kernel::new(&kind.source(false, false)).unwrap();
            let tiles = (n / kind.tile()) as u32;
            let mut cc = Array::<f32>::zeros([n, n]).unwrap();
            let run = |cc: &mut Array<f32>| {
                coop.launch_raw([tiles, tiles, 1], args![&ga, &gb, &mut *cc, uniform(&params)]).unwrap();
                ctx.sync().unwrap();
            };
            run(&mut cc);
            // The CPU reference exists on macOS only; elsewhere the f32 kernel's result stands in.
            #[cfg(target_os = "macos")]
            let want = {
                let mut out = vec![0.0; n * n];
                sgemm_cpu(n, &a, &b, &mut out);
                out
            };
            #[cfg(not(target_os = "macos"))]
            let want = f32_result.clone();
            check("sgemm coop", &cc.to_vec().unwrap(), &want, kind.tolerance(1e-4, 1e-3));
            group.bench_function(BenchmarkId::new("luma_coop", n), |bch| bch.iter(|| run(&mut cc)));
        }
        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0; n * n];
            sgemm_cpu(n, &a, &b, &mut out);
            check("sgemm", &f32_result, &out, 1e-4);
            group.bench_function(BenchmarkId::new("cpu_accelerate", n), |bch| {
                bch.iter(|| sgemm_cpu(n, &a, &b, black_box(&mut out)))
            });
        }
    }
    group.finish();
}

// -- 2D heat diffusion --

const ALPHA: f32 = 0.2;

fn heat_step(u: &Array<f32>) -> Array<f32> {
    let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0;
    u + lap * ALPHA
}

/// One kernel per step.
fn heat_luma_per_step(u0: &Array<f32>, steps: usize) -> Array<f32> {
    let mut u = u0 * 1.0;
    for _ in 0..steps {
        u = heat_step(&u);
    }
    u
}

/// Periodic five-point stencil on a `w`-wide grid, `steps` times, ping-ponging two buffers.
fn heat_cpu(w: usize, u0: &[f32], steps: usize, a: &mut Vec<f32>, b: &mut Vec<f32>) {
    a.copy_from_slice(u0);
    let h = u0.len() / w;
    for _ in 0..steps {
        let src = &*a;
        par_rows(b, w, |first, rows| {
            for (r, out) in rows.chunks_mut(w).enumerate() {
                let y = first + r;
                let row = &src[y * w..][..w];
                let up = &src[((y + h - 1) % h) * w..][..w];
                let down = &src[((y + 1) % h) * w..][..w];
                let at = |x: usize, l: f32, r: f32| row[x] + ALPHA * (l + r + up[x] + down[x] - 4.0 * row[x]);
                out[0] = at(0, row[w - 1], row[1]);
                out[w - 1] = at(w - 1, row[w - 2], row[0]);
                for x in 1..w - 1 {
                    out[x] = row[x] + ALPHA * (row[x - 1] + row[x + 1] + up[x] + down[x] - 4.0 * row[x]);
                }
            }
        });
        std::mem::swap(a, b);
    }
}

fn heat(c: &mut Criterion) {
    let steps = 20usize;
    let mut group = c.benchmark_group("heat_20_steps");
    for w in [2048usize, 4096] {
        let u0 = values(w * w, 10, 0.0, 1.0);
        let gu = Array::from_slice(&u0, [w, w]).unwrap();
        let (mut a, mut b) = (vec![0.0; w * w], vec![0.0; w * w]);
        heat_cpu(w, &u0, steps, &mut a, &mut b);
        check("heat", &gu.iterate(steps, heat_step).to_vec().unwrap(), &a, 1e-5);
        check("heat per step", &heat_luma_per_step(&gu, steps).to_vec().unwrap(), &a, 1e-5);

        group.throughput(Throughput::Elements((w * w * steps) as u64));
        group.bench_function(BenchmarkId::new("luma_iterate", w), |bch| {
            bch.iter(|| gu.iterate(steps, heat_step).eval().unwrap())
        });
        group.bench_function(BenchmarkId::new("luma_per_step", w), |bch| {
            bch.iter(|| heat_luma_per_step(&gu, steps).eval().unwrap())
        });
        group.bench_function(BenchmarkId::new("cpu", w), |bch| bch.iter(|| heat_cpu(w, &u0, steps, &mut a, &mut b)));
    }
    group.finish();
}

// -- FIR filter (full linear convolution) --

/// Overlap-save with `block`-point FFTs, two real blocks per complex transform, blocks spread
/// over all cores.
fn fir_cpu(x: &[f32], h: &[f32], block: usize, out: &mut [f32]) {
    use rustfft::num_complex::Complex32;
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let (fwd, inv): (Arc<dyn rustfft::Fft<f32>>, _) = (planner.plan_fft_forward(block), planner.plan_fft_inverse(block));
    let mut spectrum: Vec<Complex32> = (0..block).map(|i| Complex32::new(h.get(i).copied().unwrap_or(0.0), 0.0)).collect();
    fwd.process(&mut spectrum);
    let lead = h.len() - 1;
    let step = block - lead;
    let pairs = out.len().div_ceil(step).div_ceil(2);
    let per = pairs.div_ceil(threads());
    let scale = 1.0 / block as f32;
    let at = |i: isize| if i >= 0 && (i as usize) < x.len() { x[i as usize] } else { 0.0 };
    std::thread::scope(|s| {
        for (chunk_index, chunk) in out.chunks_mut(2 * step * per).enumerate() {
            let (fwd, inv, spectrum, at) = (&fwd, &inv, &spectrum, &at);
            s.spawn(move || {
                let mut buf = vec![Complex32::default(); block];
                let mut scratch = vec![Complex32::default(); fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
                for q in 0..per {
                    let first = 2 * step * q;
                    if first >= chunk.len() {
                        break;
                    }
                    let origin = (2 * step * (chunk_index * per + q)) as isize - lead as isize;
                    for (e, v) in buf.iter_mut().enumerate() {
                        *v = Complex32::new(at(origin + e as isize), at(origin + (e + step) as isize));
                    }
                    fwd.process_with_scratch(&mut buf, &mut scratch);
                    for (v, s) in buf.iter_mut().zip(spectrum) {
                        *v *= s;
                    }
                    inv.process_with_scratch(&mut buf, &mut scratch);
                    for (e, v) in buf.iter().enumerate().skip(lead) {
                        let o = first + e - lead;
                        if let Some(slot) = chunk.get_mut(o) {
                            *slot = v.re * scale;
                        }
                        if let Some(slot) = chunk.get_mut(o + step) {
                            *slot = v.im * scale;
                        }
                    }
                }
            });
        }
    });
}

fn fir(c: &mut Criterion) {
    let n = 1 << 24;
    let mut group = c.benchmark_group("fir_16M_samples");
    group.throughput(Throughput::Elements(n as u64));
    let x = values(n, 11, -1.0, 1.0);
    let gx = Array::from_slice(&x, n).unwrap();
    for taps in [255usize, 4095, 65535] {
        let h = values(taps, 12, -0.1, 0.1);
        let gh = Array::from_slice(&h, taps).unwrap();
        let block = (4 * taps).next_power_of_two().max(4096);
        let mut out = vec![0.0; n + taps - 1];
        fir_cpu(&x, &h, block, &mut out);
        check("fir", &gx.conv1d_fft(&gh).to_vec().unwrap(), &out, 1e-4);
        group.bench_function(BenchmarkId::new("luma", taps), |b| b.iter(|| gx.conv1d_fft(&gh).eval().unwrap()));
        group.bench_function(BenchmarkId::new("luma_with_transfers", taps), |b| {
            b.iter(|| black_box(Array::from_slice(&x, n).unwrap().conv1d_fft(&gh).to_vec().unwrap()))
        });
        group.bench_function(BenchmarkId::new("cpu_rustfft", taps), |b| b.iter(|| fir_cpu(&x, &h, block, black_box(&mut out))));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = black_scholes, mandelbrot, nbody, sgemm, heat, fir
}
criterion_main!(benches);
