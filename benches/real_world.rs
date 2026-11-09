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
const A: [f32; 5] = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
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

const NBODY: &str = r#"
struct Params { n: u32, eps2: f32 }
@group(1) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> acc: array<vec4<f32>>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> tile: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let me = pos[min(g.x, p.n - 1u)].xyz;
    var a = vec3<f32>(0.0);
    for (var base = 0u; base < p.n; base = base + 256u) {
        let j = base + lid;
        tile[lid] = select(vec4<f32>(0.0), pos[min(j, p.n - 1u)], j < p.n);
        workgroupBarrier();
        for (var k = 0u; k < 256u; k = k + 1u) {
            let other = tile[k];
            let d = other.xyz - me;
            let inv = inverseSqrt(dot(d, d) + p.eps2);
            a = a + d * (other.w * inv * inv * inv);
        }
        workgroupBarrier();
    }
    if (g.x < p.n) { acc[g.x] = vec4<f32>(a, 0.0); }
}
"#;

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
            let chunks = px.chunks_exact(8).zip(py.chunks_exact(8)).zip(pz.chunks_exact(8)).zip(pm.chunks_exact(8));
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
    let kernel = Kernel::new(NBODY).unwrap();
    let params = NbodyParams { n: n as u32, eps2: EPS2 };
    let ctx = default_context().unwrap();
    let run = |acc: &mut Array<[f32; 4]>| {
        kernel.launch([n as u32], args![&pos, &mut *acc, uniform(&params)]).unwrap();
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

/// Row-major `C = A B`; each workgroup computes a 128x128 tile of `C`, each of its 16x16
/// invocations 8x8 of it (two 4-wide groups of rows and of columns, 64 apart, so that workgroup
/// memory reads of neighbouring invocations are contiguous), from 16-wide slices of `A` and `B`
/// staged in workgroup memory. Dimensions must be multiples of 128 (M, N) and 16 (K).
fn sgemm_source() -> String {
    // Accumulator `c{2 i + h}` holds row `i` (of 8), column half `h`; unrolled so they stay in
    // registers.
    let (mut decl, mut update, mut store) = (String::new(), String::new(), String::new());
    for i in 0..8 {
        let a_row = format!("a{}.{}", i / 4, ["x", "y", "z", "w"][i % 4]);
        for h in 0..2 {
            let acc = format!("c{}", i * 2 + h);
            decl += &format!("    var {acc} = vec4<f32>(0.0);\n");
            update += &format!("            {acc} = fma(vec4<f32>({a_row}), b{h}, {acc});\n");
            store += &format!(
                "    {{
        let at_c = (row0 + {} + lid.y * 4u) * p.n + col0 + {} + lid.x * 4u;
        c[at_c] = {acc}.x; c[at_c + 1u] = {acc}.y; c[at_c + 2u] = {acc}.z; c[at_c + 3u] = {acc}.w;
    }}
",
                (i / 4) * 64 + i % 4,
                h * 64
            );
        }
    }
    format!(
        "struct Params {{ m: u32, n: u32, k: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
// Scalar slots: neighbouring invocations write different components of the same row group,
// and a component store into a shared vec4 is not atomic on every backend.
var<workgroup> at: array<f32, 2048>;
var<workgroup> bt: array<f32, 2048>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row0 = wid.y * 128u;
    let col0 = wid.x * 128u;
    let t = lid.y * 16u + lid.x;
{decl}    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {{
        for (var q = 0u; q < 8u; q = q + 1u) {{
            let e = t + 256u * q;
            // A slice: 128 rows x 16 k, stored k-major; B slice: 16 k x 128 columns.
            let r = e / 16u; let kk = e % 16u;
            at[kk * 128u + r] = a[(row0 + r) * p.k + k0 + kk];
            let kb = e / 128u; let col = e % 128u;
            bt[kb * 128u + col] = b[(k0 + kb) * p.n + col0 + col];
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {{
            let ia = kk * 128u + lid.y * 4u;
            let ib = kk * 128u + lid.x * 4u;
            let a0 = vec4<f32>(at[ia], at[ia + 1u], at[ia + 2u], at[ia + 3u]);
            let a1 = vec4<f32>(at[ia + 64u], at[ia + 65u], at[ia + 66u], at[ia + 67u]);
            let b0 = vec4<f32>(bt[ib], bt[ib + 1u], bt[ib + 2u], bt[ib + 3u]);
            let b1 = vec4<f32>(bt[ib + 64u], bt[ib + 65u], bt[ib + 66u], bt[ib + 67u]);
{update}        }}
        workgroupBarrier();
    }}
{store}}}
"
    )
}

#[derive(encase::ShaderType)]
struct SgemmParams {
    m: u32,
    n: u32,
    k: u32,
}

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
        let kernel = Kernel::new(&sgemm_source()).unwrap();
        let params = SgemmParams { m: n as u32, n: n as u32, k: n as u32 };
        let ctx = default_context().unwrap();
        let run = |gc: &mut Array<f32>| {
            kernel.launch_raw([n as u32 / 128, n as u32 / 128, 1], args![&ga, &gb, &mut *gc, uniform(&params)]).unwrap();
            ctx.sync().unwrap();
        };
        run(&mut gc);
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        group.bench_function(BenchmarkId::new("luma", n), |bch| bch.iter(|| run(&mut gc)));
        #[cfg(target_os = "macos")]
        {
            let mut out = vec![0.0; n * n];
            sgemm_cpu(n, &a, &b, &mut out);
            check("sgemm", &gc.to_vec().unwrap(), &out, 1e-4);
            group.bench_function(BenchmarkId::new("cpu_accelerate", n), |bch| {
                bch.iter(|| sgemm_cpu(n, &a, &b, black_box(&mut out)))
            });
        }
    }
    group.finish();
}

// -- 2D heat diffusion --

const ALPHA: f32 = 0.2;

fn heat_luma(u0: &Array<f32>, steps: usize) -> Array<f32> {
    let mut u = u0 * 1.0;
    for _ in 0..steps {
        let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - &u * 4.0;
        u = &u + lap * ALPHA;
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
    let (w, steps) = (2048usize, 20usize);
    let u0 = values(w * w, 10, 0.0, 1.0);
    let gu = Array::from_slice(&u0, [w, w]).unwrap();
    let (mut a, mut b) = (vec![0.0; w * w], vec![0.0; w * w]);
    heat_cpu(w, &u0, steps, &mut a, &mut b);
    check("heat", &heat_luma(&gu, steps).to_vec().unwrap(), &a, 1e-5);

    let mut group = c.benchmark_group("heat_2048x2048_20_steps");
    group.throughput(Throughput::Elements((w * w * steps) as u64));
    group.bench_function("luma", |bch| bch.iter(|| heat_luma(&gu, steps).eval().unwrap()));
    group.bench_function("cpu", |bch| bch.iter(|| heat_cpu(w, &u0, steps, &mut a, &mut b)));
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
