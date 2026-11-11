//! Image-processing workloads on a 4096x4096 image, Luma against the same work on every CPU core.
//!
//! `cargo bench --bench imaging` (add `-- <filter>` for one group). Build with
//! `RUSTFLAGS="-C target-cpu=native"` so the CPU side gets the host's vector units and fused
//! multiply-add. The CPU baselines are parallel over all cores; their inner loops run over fixed
//! blocks of independent lanes that LLVM turns into vector code. All borders clamp to the edge.
//! Every group first checks that both sides compute the same result.

use std::fmt::Write;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use luma::*;

const W: usize = 4096;
const H: usize = 4096;

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

/// A smooth gradient with hard-edged tiles and mild noise, in `0..1`.
fn test_image() -> Vec<f32> {
    let noise = values(W * H, 1, -0.03, 0.03);
    noise
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let (x, y) = ((i % W) as f32, (i / W) as f32);
            let tile = if ((i % W) / 256 + (i / W) / 256) % 2 == 0 { 0.15 } else { 0.0 };
            (0.5 + 0.3 * (x * 0.01).sin() * (y * 0.013).cos() + tile + n).clamp(0.0, 1.0)
        })
        .collect()
}

const SWIZZLE: [&str; 4] = ["x", "y", "z", "w"];

// -- Separable Gaussian blur --

const BLOCK: usize = 64;

/// Normalised Gaussian taps (sigma = radius / 2) for `2 * radius + 1` samples.
fn gaussian_taps(radius: usize) -> Vec<f32> {
    let sigma = radius as f64 / 2.0;
    let w: Vec<f64> = (0..=2 * radius)
        .map(|k| (-((k as f64 - radius as f64).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = w.iter().sum();
    w.iter().map(|v| (v / sum) as f32).collect()
}

/// Horizontal pass: one workgroup blurs 1024 pixels of a row, 4 per invocation. The image is
/// viewed as `vec4`s (problem size `[w / 4, h]`); the row segment plus halo is staged in workgroup
/// memory with aligned `vec4` loads and each invocation reads its window into registers once.
fn gauss_h_kernel(taps: &[f32]) -> Kernel {
    let r = taps.len() / 2;
    let pad = r.div_ceil(4);
    let o = 4 * pad - r;
    let nv = (o + 2 * r + 4).div_ceil(4);
    let rv = 256 + nv - 1;
    let mut src = format!(
        r#"
@group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
var<workgroup> tile: array<vec4<f32>, {rv}>;
@compute @workgroup_size(256, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let wv = i32(luma.size.x);
    let row = wid.y * luma.size.x;
    let cv0 = i32(wid.x * 256u) - {pad};
    for (var i = lid.x; i < {rv}u; i = i + 256u) {{
        let cv = cv0 + i32(i);
        var v: vec4<f32>;
        if (cv < 0) {{
            v = vec4<f32>(src[row].x);
        }} else if (cv >= wv) {{
            v = vec4<f32>(src[row + u32(wv) - 1u].w);
        }} else {{
            v = src[row + u32(cv)];
        }}
        tile[i] = v;
    }}
    workgroupBarrier();
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
"#
    );
    for j in 0..nv {
        writeln!(src, "    let q{j} = tile[lid.x + {j}u];").unwrap();
    }
    for (dx, w) in taps.iter().enumerate() {
        for k in 0..4 {
            let n = o + k + dx;
            writeln!(src, "    a{k} = fma(q{}.{}, {w:e}, a{k});", n / 4, SWIZZLE[n % 4]).unwrap();
        }
    }
    src.push_str("    dst[row + wid.x * 256u + lid.x] = vec4<f32>(a0, a1, a2, a3);\n}\n");
    Kernel::new(&src).unwrap()
}

/// Rows each invocation of the vertical kernel produces.
const V_ROWS: usize = 4;

/// Vertical pass: each invocation owns 4 adjacent columns (one `vec4`) and `V_ROWS` consecutive
/// output rows, walking down the source rows once and feeding every row it touches to all the
/// outputs it contributes to. Problem size `[w / 4, h / V_ROWS]`.
fn gauss_v_kernel(taps: &[f32]) -> Kernel {
    let r = taps.len() / 2;
    let mut src = format!(
        r#"
@group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(32, 2)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let wv = luma.size.x;
    let h = i32(luma.size.y) * {V_ROWS};
    let y0 = i32(g.y) * {V_ROWS} - {r};
"#
    );
    for j in 0..V_ROWS {
        writeln!(src, "    var a{j} = vec4<f32>(0.0);").unwrap();
    }
    for t in 0..2 * r + V_ROWS {
        writeln!(src, "    {{\n        let q = src[u32(clamp(y0 + {t}, 0, h - 1)) * wv + g.x];").unwrap();
        for j in 0..V_ROWS {
            let Some(k) = t.checked_sub(j).filter(|&k| k < taps.len()) else { continue };
            writeln!(src, "        a{j} = fma(q, vec4<f32>({:e}), a{j});", taps[k]).unwrap();
        }
        src.push_str("    }\n");
    }
    for j in 0..V_ROWS {
        writeln!(src, "    dst[(g.y * {V_ROWS}u + {j}u) * wv + g.x] = a{j};").unwrap();
    }
    src.push_str("}\n");
    Kernel::new(&src).unwrap()
}

/// Horizontal pass: a row padded at both ends, taps outermost over a block of output pixels.
/// Vertical pass: one block of columns at a time down the thread's rows, so the `taps` source
/// rows it touches stay in L1.
fn gauss_cpu(taps: &[f32], src: &[f32], tmp: &mut [f32], out: &mut [f32]) {
    let r = taps.len() / 2;
    par_rows(tmp, W, |first, rows| {
        let mut pad = vec![0.0f32; W + 2 * r];
        for (ri, row) in rows.chunks_mut(W).enumerate() {
            let s = &src[(first + ri) * W..][..W];
            pad[..r].fill(s[0]);
            pad[r..r + W].copy_from_slice(s);
            pad[r + W..].fill(s[W - 1]);
            for x0 in (0..W).step_by(BLOCK) {
                let mut acc = [0.0f32; BLOCK];
                for (k, &wk) in taps.iter().enumerate() {
                    let p: &[f32; BLOCK] = pad[x0 + k..][..BLOCK].try_into().unwrap();
                    for (a, &v) in acc.iter_mut().zip(p) {
                        *a = wk.mul_add(v, *a);
                    }
                }
                row[x0..x0 + BLOCK].copy_from_slice(&acc);
            }
        }
    });
    let tmp = &*tmp;
    par_rows(out, W, |first, rows| {
        for x0 in (0..W).step_by(BLOCK) {
            for ri in 0..rows.len() / W {
                let y = first + ri;
                let mut acc = [0.0f32; BLOCK];
                for (k, &wk) in taps.iter().enumerate() {
                    let sy = (y + k).saturating_sub(r).min(H - 1);
                    let p: &[f32; BLOCK] = tmp[sy * W + x0..][..BLOCK].try_into().unwrap();
                    for (a, &v) in acc.iter_mut().zip(p) {
                        *a = wk.mul_add(v, *a);
                    }
                }
                rows[ri * W + x0..][..BLOCK].copy_from_slice(&acc);
            }
        }
    });
}

fn gaussian(c: &mut Criterion) {
    let img = test_image();
    let quads: Vec<[f32; 4]> = img.as_chunks::<4>().0.to_vec();
    let gsrc = Array::from_slice(&quads, [W / 4, H]).unwrap();
    let ctx = default_context().unwrap();
    let (mut tmp, mut out) = (vec![0.0f32; W * H], vec![0.0f32; W * H]);
    let mut group = c.benchmark_group("gaussian_blur_4096x4096");
    group.throughput(Throughput::Elements((W * H) as u64));
    for r in [8usize, 32] {
        let taps = gaussian_taps(r);
        let (kh, kv) = (gauss_h_kernel(&taps), gauss_v_kernel(&taps));
        let mut gtmp = Array::<[f32; 4]>::zeros([W / 4, H]).unwrap();
        let mut gout = Array::<[f32; 4]>::zeros([W / 4, H]).unwrap();
        let run = |gtmp: &mut Array<[f32; 4]>, gout: &mut Array<[f32; 4]>| {
            kh.launch([W as u32 / 4, H as u32], args![&gsrc, &mut *gtmp]).unwrap();
            kv.launch([W as u32 / 4, (H / V_ROWS) as u32], args![&*gtmp, &mut *gout]).unwrap();
            ctx.sync().unwrap();
        };
        run(&mut gtmp, &mut gout);
        gauss_cpu(&taps, &img, &mut tmp, &mut out);
        check("gaussian", gout.to_vec().unwrap().as_flattened(), &out, 1e-5);
        group.bench_function(BenchmarkId::new("luma", format!("r{r}")), |b| b.iter(|| run(&mut gtmp, &mut gout)));
        group.bench_function(BenchmarkId::new("cpu", format!("r{r}")), |b| {
            b.iter(|| gauss_cpu(&taps, &img, &mut tmp, black_box(&mut out)))
        });
    }
    group.finish();
}

// -- Stencils with a padded-band CPU driver: dense convolution and bilateral filter --

/// Fills `out` with `f(pad, pad_stride, origin)` blocks of `B` pixels. `pad` holds the band of
/// source rows (edge-clamped, `r` extra pixels on each side) that the output rows need, built
/// once per 32 output rows; `origin` indexes the top-left of the `(2r+1)`-wide window of the
/// block's first pixel.
fn par_stencil<const B: usize>(
    src: &[f32],
    out: &mut [f32],
    r: usize,
    f: impl Fn(&[f32], usize, usize) -> [f32; B] + Sync,
) {
    const BAND: usize = 32;
    let pw = W + 2 * r;
    par_rows(out, W, |first, rows| {
        let mut pad = vec![0.0f32; (BAND + 2 * r) * pw];
        for (c, chunk) in rows.chunks_mut(BAND * W).enumerate() {
            let y0 = first + c * BAND;
            let n = chunk.len() / W;
            for i in 0..n + 2 * r {
                let s = &src[(y0 + i).saturating_sub(r).min(H - 1) * W..][..W];
                let d = &mut pad[i * pw..][..pw];
                d[..r].fill(s[0]);
                d[r..r + W].copy_from_slice(s);
                d[r + W..].fill(s[W - 1]);
            }
            for x0 in (0..W).step_by(B) {
                for ri in 0..n {
                    let block = f(&pad, pw, ri * pw + x0);
                    chunk[ri * W + x0..][..B].copy_from_slice(&block);
                }
            }
        }
    });
}

/// Tiled stencil kernel template: a 16x16 workgroup stages its tile plus halo (edge-clamped) in
/// workgroup memory; `@BODY@` computes `res` from `tile` at `c`, the window's top-left.
const STENCIL: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
const T: u32 = @T@u;
var<workgroup> tile: array<f32, @TT@>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(global_invocation_id) g: vec3<u32>) {
    let w = i32(luma.size.x);
    let h = i32(luma.size.y);
    let x0 = i32(wid.x * 16u) - @R@;
    let y0 = i32(wid.y * 16u) - @R@;
    let t = lid.y * 16u + lid.x;
    for (var i = t; i < T * T; i = i + 256u) {
        let sx = clamp(x0 + i32(i % T), 0, w - 1);
        let sy = clamp(y0 + i32(i / T), 0, h - 1);
        tile[i] = src[u32(sy * w + sx)];
    }
    workgroupBarrier();
    let c = lid.y * T + lid.x;
    var res = 0.0;
@BODY@
    dst[luma_idx2(g.x, g.y)] = res;
}
"#;

fn stencil_kernel(r: usize, body: &str) -> Kernel {
    let t = 16 + 2 * r;
    let src = STENCIL
        .replace("@T@", &t.to_string())
        .replace("@TT@", &(t * t).to_string())
        .replace("@R@", &r.to_string())
        .replace("@BODY@", body);
    Kernel::new(&src).unwrap()
}

// -- Dense 2-D convolution --

fn conv_weights(size: usize) -> Vec<f32> {
    let w = values(size * size, 20 + size as u64, 0.0, 1.0);
    let sum: f32 = w.iter().sum();
    w.iter().map(|v| v / sum).collect()
}

/// Dense convolution where each invocation produces 4 adjacent pixels of a row, with the image
/// viewed as `vec4`s (problem size `[w / 4, h]`). The workgroup stages a 64-wide tile (plus halo)
/// with aligned `vec4` loads, so each lane reads consecutive `vec4`s (no bank conflicts); a
/// window row is read into registers once and reused by all 4 outputs.
const CONV_RP: usize = 2;

fn conv_kernel(size: usize, weights: &[f32]) -> Kernel {
    let rp = CONV_RP;
    let wy = 16 / rp;
    let r = size / 2;
    let pad = r.div_ceil(4); // halo in vec4s on the left
    let o = 4 * pad - r; // offset of the window's first column within the staged vec4s
    let nv = (o + size + 3).div_ceil(4);
    let rv = 15 + nv;
    let rows = 16 + size - 1;
    let mut src = format!(
        r#"
@group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
var<workgroup> tile: array<vec4<f32>, {n}>;
@compute @workgroup_size(16, {wy})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let wv = i32(luma.size.x);
    let h = i32(luma.size.y) * {rp};
    let cv0 = i32(wid.x * 16u) - {pad};
    let y0 = i32(wid.y * 16u) - {r};
    for (var i = lid.y * 16u + lid.x; i < {n}u; i = i + {nt}u) {{
        let row = clamp(y0 + i32(i / {rv}u), 0, h - 1) * wv;
        let cv = cv0 + i32(i % {rv}u);
        var v: vec4<f32>;
        if (cv < 0) {{
            v = vec4<f32>(src[row].x);
        }} else if (cv >= wv) {{
            v = vec4<f32>(src[row + wv - 1].w);
        }} else {{
            v = src[row + cv];
        }}
        tile[i] = v;
    }}
    workgroupBarrier();
"#,
        n = rows * rv,
        nt = 16 * wy,
    );
    for j in 0..rp {
        for k in 0..4 {
            writeln!(src, "    var a{j}{k} = 0.0;").unwrap();
        }
    }
    for t in 0..size + rp - 1 {
        writeln!(src, "    {{\n        let rb = (lid.y * {rp}u + {t}u) * {rv}u + lid.x;").unwrap();
        for j in 0..nv {
            writeln!(src, "        let q{j} = tile[rb + {j}u];").unwrap();
        }
        for j in 0..rp {
            let Some(dy) = t.checked_sub(j).filter(|&dy| dy < size) else { continue };
            for dx in 0..size {
                let wk = weights[dy * size + dx];
                for k in 0..4 {
                    let n = o + k + dx;
                    writeln!(src, "        a{j}{k} = fma(q{}.{}, {wk:e}, a{j}{k});", n / 4, ["x", "y", "z", "w"][n % 4]).unwrap();
                }
            }
        }
        src.push_str("    }\n");
    }
    writeln!(src, "    let base = (wid.y * 16u + lid.y * {rp}u) * luma.size.x + wid.x * 16u + lid.x;").unwrap();
    for j in 0..rp {
        writeln!(src, "    dst[base + {j}u * luma.size.x] = vec4<f32>(a{j}0, a{j}1, a{j}2, a{j}3);").unwrap();
    }
    src.push_str("}\n");
    Kernel::new(&src).unwrap()
}

fn conv_cpu(size: usize, weights: &[f32], src: &[f32], out: &mut [f32]) {
    par_stencil::<BLOCK>(src, out, size / 2, |pad, pw, origin| {
        let mut acc = [0.0f32; BLOCK];
        for dy in 0..size {
            for dx in 0..size {
                let wk = weights[dy * size + dx];
                let p: &[f32; BLOCK] = pad[origin + dy * pw + dx..][..BLOCK].try_into().unwrap();
                for (a, &v) in acc.iter_mut().zip(p) {
                    *a = wk.mul_add(v, *a);
                }
            }
        }
        acc
    });
}

fn conv(c: &mut Criterion) {
    let img = test_image();
    let quads: Vec<[f32; 4]> = img.as_chunks::<4>().0.to_vec();
    let gsrc = Array::from_slice(&quads, [W / 4, H]).unwrap();
    let ctx = default_context().unwrap();
    let mut out = vec![0.0f32; W * H];
    let mut group = c.benchmark_group("conv2d_dense_4096x4096");
    group.throughput(Throughput::Elements((W * H) as u64));
    for size in [5usize, 9] {
        let weights = conv_weights(size);
        let kernel = conv_kernel(size, &weights);
        let mut gout = Array::<[f32; 4]>::zeros([W / 4, H]).unwrap();
        let run = |gout: &mut Array<[f32; 4]>| {
            kernel.launch([W as u32 / 4, (H / CONV_RP) as u32], args![&gsrc, &mut *gout]).unwrap();
            ctx.sync().unwrap();
        };
        run(&mut gout);
        conv_cpu(size, &weights, &img, &mut out);
        check("conv2d", gout.to_vec().unwrap().as_flattened(), &out, 1e-5);
        group.bench_function(BenchmarkId::new("luma", format!("{size}x{size}")), |b| b.iter(|| run(&mut gout)));
        group.bench_function(BenchmarkId::new("cpu", format!("{size}x{size}")), |b| {
            b.iter(|| conv_cpu(size, &weights, &img, black_box(&mut out)))
        });
    }
    group.finish();
}

// -- Bilateral filter --

const BIL_R: usize = 5;
const BIL_T: usize = 2 * BIL_R + 1;
const SIGMA_SPATIAL: f64 = 2.0;
/// `-1 / (2 sigma_range^2)` with sigma_range = 0.1.
const BIL_K: f32 = -50.0;

fn bilateral_spatial() -> Vec<f32> {
    let mut w = Vec::with_capacity(BIL_T * BIL_T);
    for dy in 0..BIL_T {
        for dx in 0..BIL_T {
            let d2 = (dx as f64 - BIL_R as f64).powi(2) + (dy as f64 - BIL_R as f64).powi(2);
            w.push((-d2 / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL)).exp() as f32);
        }
    }
    w
}

fn bilateral_kernel(spatial: &[f32]) -> Kernel {
    let t = 16 + 2 * BIL_R;
    let mut body = format!("    let ctr = tile[c + {}u];\n    var num = 0.0;\n    var den = 0.0;\n", BIL_R * t + BIL_R);
    for dy in 0..BIL_T {
        for dx in 0..BIL_T {
            writeln!(
                body,
                "    {{ let v = tile[c + {}u]; let d = v - ctr; let wt = {:e} * exp(d * d * {:e}); num = fma(wt, v, num); den = den + wt; }}",
                dy * t + dx,
                spatial[dy * BIL_T + dx],
                BIL_K
            )
            .unwrap();
        }
    }
    body.push_str("    res = num / den;");
    stencil_kernel(BIL_R, &body)
}

/// `exp` for `x <= 0` that LLVM can vectorize (libm's `expf` is a call): Cephes range reduction
/// by `ln 2` and a degree-5 polynomial, about 1e-7 relative error.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = n.mul_add(-0.693_359_4, x);
    let r = n.mul_add(2.121_944_4e-4, r);
    let p = 1.987_569_1e-4f32;
    let p = p.mul_add(r, 1.398_199_9e-3);
    let p = p.mul_add(r, 8.333_452e-3);
    let p = p.mul_add(r, 4.166_579_6e-2);
    let p = p.mul_add(r, 1.666_666_5e-1);
    let p = p.mul_add(r, 5e-1);
    let y = (p * r * r) + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

fn bilateral_cpu(spatial: &[f32], src: &[f32], out: &mut [f32]) {
    const B: usize = 16;
    par_stencil::<B>(src, out, BIL_R, |pad, pw, origin| {
        let ctr: &[f32; B] = pad[origin + BIL_R * pw + BIL_R..][..B].try_into().unwrap();
        let mut num = [0.0f32; B];
        let mut den = [0.0f32; B];
        for dy in 0..BIL_T {
            for dx in 0..BIL_T {
                let sw = spatial[dy * BIL_T + dx];
                let p: &[f32; B] = pad[origin + dy * pw + dx..][..B].try_into().unwrap();
                for i in 0..B {
                    let d = p[i] - ctr[i];
                    let wt = sw * exp_fast(d * d * BIL_K);
                    num[i] = wt.mul_add(p[i], num[i]);
                    den[i] += wt;
                }
            }
        }
        let mut o = [0.0f32; B];
        for i in 0..B {
            o[i] = num[i] / den[i];
        }
        o
    });
}

fn bilateral(c: &mut Criterion) {
    let img = test_image();
    let gsrc = Array::from_slice(&img, [W, H]).unwrap();
    let ctx = default_context().unwrap();
    let spatial = bilateral_spatial();
    let kernel = bilateral_kernel(&spatial);
    let mut gout = Array::<f32>::zeros([W, H]).unwrap();
    let mut out = vec![0.0f32; W * H];
    let run = |gout: &mut Array<f32>| {
        kernel.launch([W as u32, H as u32], args![&gsrc, &mut *gout]).unwrap();
        ctx.sync().unwrap();
    };
    run(&mut gout);
    bilateral_cpu(&spatial, &img, &mut out);
    check("bilateral", &gout.to_vec().unwrap(), &out, 1e-4);

    let mut group = c.benchmark_group("bilateral_r5_4096x4096");
    group.throughput(Throughput::Elements((W * H) as u64));
    group.bench_function("luma", |b| b.iter(|| run(&mut gout)));
    group.bench_function("cpu", |b| b.iter(|| bilateral_cpu(&spatial, &img, black_box(&mut out))));
    group.finish();
}

// -- RGBA colour pipeline --

/// Row-major linear-light colour matrix (a saturation boost).
const COLOR_MATRIX: [[f32; 3]; 3] = [[1.25, -0.15, -0.10], [-0.10, 1.20, -0.10], [-0.05, -0.20, 1.25]];
const EXPOSURE: f32 = 1.5;

/// sRGB decode, 3x3 matrix, exposure and Reinhard tone map, sRGB encode; alpha passes through.
const COLOR: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, c / 12.92, c <= vec3<f32>(0.04045));
}
fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, c * 12.92, c <= vec3<f32>(0.0031308));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= luma.size.x || g.y >= luma.size.y) { return; }
    let i = luma_idx2(g.x, g.y);
    let p = src[i];
    let lin = to_linear(p.rgb);
    let m = max(vec3<f32>(@ROWS@), vec3<f32>(0.0)) * @EXPOSURE@;
    dst[i] = vec4<f32>(to_srgb(m / (m + 1.0)), p.a);
}
"#;

fn color_kernel() -> Kernel {
    let row = |i: usize| {
        let m = COLOR_MATRIX[i];
        format!("dot(vec3<f32>({:e}, {:e}, {:e}), lin)", m[0], m[1], m[2])
    };
    let rows = format!("{}, {}, {}", row(0), row(1), row(2));
    let src = COLOR.replace("@ROWS@", &rows).replace("@EXPOSURE@", &format!("{EXPOSURE:e}"));
    Kernel::new(&src).unwrap()
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn color_cpu(src: &[[f32; 4]], out: &mut [[f32; 4]]) {
    par_rows(out, W, |first, rows| {
        for (o, p) in rows.iter_mut().zip(&src[first * W..]) {
            let lin = [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])];
            let mut res = [0.0f32; 4];
            for (c, m) in COLOR_MATRIX.iter().enumerate() {
                let v = (m[0] * lin[0] + m[1] * lin[1] + m[2] * lin[2]).max(0.0) * EXPOSURE;
                res[c] = linear_to_srgb(v / (v + 1.0));
            }
            res[3] = p[3];
            *o = res;
        }
    });
}

fn to_linear_luma(c: &Array<f32>) -> Array<f32> {
    select(&c.le(0.04045), &(c / 12.92), &(((c + 0.055) / 1.055).powf(2.4)))
}

fn to_srgb_luma(c: &Array<f32>) -> Array<f32> {
    select(&c.le(0.0031308), &(c * 12.92), &(c.powf(1.0 / 2.4) * 1.055 - 0.055))
}

/// The same pipeline on separate colour planes, as lazily fused elementwise operations.
fn color_luma(planes: &[Array<f32>; 3]) -> [Array<f32>; 3] {
    let lin = [to_linear_luma(&planes[0]), to_linear_luma(&planes[1]), to_linear_luma(&planes[2])];
    let channel = |m: &[f32; 3]| {
        let v = (&lin[0] * m[0] + &lin[1] * m[1] + &lin[2] * m[2]).maximum(0.0) * EXPOSURE;
        to_srgb_luma(&(&v / (&v + 1.0)))
    };
    [channel(&COLOR_MATRIX[0]), channel(&COLOR_MATRIX[1]), channel(&COLOR_MATRIX[2])]
}

fn color(c: &mut Criterion) {
    let mut rgba = vec![[0.0f32; 4]; W * H];
    for (k, plane) in (0..4).map(|k| (k, values(W * H, 30 + k as u64, 0.0, 1.0))) {
        for (p, v) in rgba.iter_mut().zip(plane) {
            p[k] = v;
        }
    }
    let gsrc = Array::from_slice(&rgba, [W, H]).unwrap();
    let kernel = color_kernel();
    let ctx = default_context().unwrap();
    let mut gout = Array::<[f32; 4]>::zeros([W, H]).unwrap();
    let run = |gout: &mut Array<[f32; 4]>| {
        kernel.launch([W as u32, H as u32], args![&gsrc, &mut *gout]).unwrap();
        ctx.sync().unwrap();
    };
    run(&mut gout);
    let mut out = vec![[0.0f32; 4]; W * H];
    color_cpu(&rgba, &mut out);
    let want: Vec<f32> = out.iter().flatten().copied().collect();
    let got: Vec<f32> = gout.to_vec().unwrap().iter().flatten().copied().collect();
    check("color kernel", &got, &want, 1e-4);

    let planes: [Array<f32>; 3] = std::array::from_fn(|k| {
        let plane: Vec<f32> = rgba.iter().map(|p| p[k]).collect();
        Array::from_slice(&plane, [W, H]).unwrap()
    });
    let fused = color_luma(&planes);
    for (k, plane) in fused.iter().enumerate() {
        let want: Vec<f32> = out.iter().map(|p| p[k]).collect();
        check("color fused ops", &plane.to_vec().unwrap(), &want, 1e-4);
    }

    let mut group = c.benchmark_group("color_pipeline_rgba_4096x4096");
    group.throughput(Throughput::Elements((W * H) as u64));
    group.bench_function("luma_kernel", |b| b.iter(|| run(&mut gout)));
    group.bench_function("luma_fused_ops", |b| {
        b.iter(|| {
            for plane in color_luma(&planes) {
                plane.eval().unwrap();
            }
        })
    });
    group.bench_function("cpu", |b| b.iter(|| color_cpu(&rgba, black_box(&mut out))));
    group.finish();
}

// -- 256-bin histogram --

/// Each workgroup counts a grid-stride share of the image (4 pixels per load) into workgroup-local
/// atomic bins, then adds them to the global bins.
const HISTOGRAM: &str = r#"
struct Params { n4: u32 }
@group(1) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> bins: array<atomic<u32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    atomicStore(&bins[lid.x], 0u);
    workgroupBarrier();
    let stride = nwg.x * 256u;
    var i = wid.x * 256u + lid.x;
    for (; i + 3u * stride < p.n4; i = i + 4u * stride) {
        let a = src[i] & vec4<u32>(255u);
        let b = src[i + stride] & vec4<u32>(255u);
        let c = src[i + 2u * stride] & vec4<u32>(255u);
        let d = src[i + 3u * stride] & vec4<u32>(255u);
        atomicAdd(&bins[a.x], 1u);
        atomicAdd(&bins[b.x], 1u);
        atomicAdd(&bins[c.x], 1u);
        atomicAdd(&bins[d.x], 1u);
        atomicAdd(&bins[a.y], 1u);
        atomicAdd(&bins[b.y], 1u);
        atomicAdd(&bins[c.y], 1u);
        atomicAdd(&bins[d.y], 1u);
        atomicAdd(&bins[a.z], 1u);
        atomicAdd(&bins[b.z], 1u);
        atomicAdd(&bins[c.z], 1u);
        atomicAdd(&bins[d.z], 1u);
        atomicAdd(&bins[a.w], 1u);
        atomicAdd(&bins[b.w], 1u);
        atomicAdd(&bins[c.w], 1u);
        atomicAdd(&bins[d.w], 1u);
    }
    for (; i < p.n4; i = i + stride) {
        let v = src[i] & vec4<u32>(255u);
        atomicAdd(&bins[v.x], 1u);
        atomicAdd(&bins[v.y], 1u);
        atomicAdd(&bins[v.z], 1u);
        atomicAdd(&bins[v.w], 1u);
    }
    workgroupBarrier();
    let s = atomicLoad(&bins[lid.x]);
    atomicAdd(&hist[lid.x], s);
}
"#;

#[derive(encase::ShaderType)]
struct HistogramParams {
    n4: u32,
}

/// Per-thread histograms over a contiguous chunk (four interleaved copies to avoid store-to-load
/// stalls on repeated bins), merged at the end.
fn histogram_cpu(data: &[u32]) -> [u32; 256] {
    let per = data.len().div_ceil(threads());
    std::thread::scope(|s| {
        let parts: Vec<_> = data
            .chunks(per)
            .map(|chunk| {
                s.spawn(move || {
                    let mut h = [[0u32; 256]; 4];
                    let (quads, rest) = chunk.as_chunks::<4>();
                    for q in quads {
                        for (j, v) in q.iter().enumerate() {
                            h[j][(v & 255) as usize] += 1;
                        }
                    }
                    for v in rest {
                        h[0][(v & 255) as usize] += 1;
                    }
                    h
                })
            })
            .collect();
        let mut total = [0u32; 256];
        for part in parts {
            for h in part.join().unwrap() {
                for (t, c) in total.iter_mut().zip(h) {
                    *t += c;
                }
            }
        }
        total
    })
}

fn histogram(c: &mut Criterion) {
    // Bell-shaped 8-bit intensities, so some bins are much hotter than others.
    let (a, b) = (values(W * H, 40, 0.0, 1.0), values(W * H, 41, 0.0, 1.0));
    let data: Vec<u32> = a.iter().zip(&b).map(|(a, b)| (((a + b) * 0.5) * 255.99) as u32).collect();
    let quads: Vec<[u32; 4]> = data.as_chunks::<4>().0.iter().map(|q| [q[0], q[1], q[2], q[3]]).collect();
    let gsrc = Array::from_slice(&quads, [quads.len()]).unwrap();
    let kernel = Kernel::new(HISTOGRAM).unwrap();
    let ctx = default_context().unwrap();
    let params = HistogramParams { n4: quads.len() as u32 };
    let run = || {
        let mut hist = Array::<u32>::zeros([256]).unwrap();
        kernel.launch_raw([512, 1, 1], args![&gsrc, &mut hist, uniform(&params)]).unwrap();
        ctx.sync().unwrap();
        hist
    };
    let want = histogram_cpu(&data);
    assert_eq!(run().to_vec().unwrap(), want, "histogram: GPU and CPU bins differ");

    let mut group = c.benchmark_group("histogram_256_bins_4096x4096");
    group.throughput(Throughput::Elements((W * H) as u64));
    group.bench_function("luma", |b| b.iter(run));
    group.bench_function("cpu", |b| b.iter(|| black_box(histogram_cpu(&data))));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = gaussian, conv, bilateral, color, histogram
}
criterion_main!(benches);
