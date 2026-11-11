//! Shared parameters, data generation, CPU references, checks, timing and CSV output for the
//! Luma vs ArrayFire benchmarks.

use std::io::Write;
use std::time::Instant;

pub const ALPHA: f32 = 0.2;
pub const RATE: f32 = 0.02;
pub const VOL: f32 = 0.3;
pub const A: [f32; 5] = [0.319_381_54, -0.356_563_78, 1.781_477_9, -1.821_255_9, 1.330_274_5];
pub const INV_SQRT_2PI: f32 = 0.398_942_3;

pub struct Sizes {
    pub heat: Vec<usize>,
    pub heat_steps: usize,
    pub life: Vec<usize>,
    pub life_steps: usize,
    pub bs: Vec<usize>,
    pub mandel: Vec<usize>,
    pub mandel_iters: usize,
    pub fir_n: usize,
    pub fir_taps: Vec<usize>,
    pub fir_direct_max_taps: usize,
}

pub fn sizes(quick: bool) -> Sizes {
    if quick {
        Sizes {
            heat: vec![1024],
            heat_steps: 100,
            life: vec![1024],
            life_steps: 100,
            bs: vec![1 << 20],
            mandel: vec![512],
            mandel_iters: 64,
            fir_n: 1 << 20,
            fir_taps: vec![127],
            fir_direct_max_taps: 4095,
        }
    } else {
        Sizes {
            heat: vec![1024, 4096, 8192],
            heat_steps: 100,
            life: vec![1024, 4096, 8192],
            life_steps: 100,
            bs: vec![1 << 24, 1 << 26],
            mandel: vec![1024, 4096],
            mandel_iters: 256,
            fir_n: 1 << 24,
            fir_taps: vec![127, 255, 4095, 65535],
            fir_direct_max_taps: 4095,
        }
    }
}

pub struct Args {
    pub quick: bool,
    pub only: Option<String>,
    pub backend: Option<String>,
}

pub fn parse_args() -> Args {
    let mut args = Args { quick: false, only: None, backend: None };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--quick" => args.quick = true,
            "--only" => args.only = Some(it.next().expect("--only needs a workload")),
            "--backend" => args.backend = Some(it.next().expect("--backend needs a value")),
            other => panic!("unknown argument {other:?} (flags: --quick, --only <workload>, --backend <cuda|opencl|cpu>)"),
        }
    }
    args
}

pub fn wants(args: &Args, workload: &str) -> bool {
    args.only.as_deref().is_none_or(|o| o == workload)
}

// -- data --

/// Deterministic values in `lo..hi`.
pub fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + ((state >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

pub fn life_grid(w: usize, h: usize) -> Vec<f32> {
    values(w * h, 21, 0.0, 1.0).into_iter().map(|v| if v < 0.3 { 1.0 } else { 0.0 }).collect()
}

// -- checks --

/// max |g-w| / max |w| (scale floored at 1e-30).
pub fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "rel_err: length mismatch");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| {
        let d = (g - w).abs();
        // NaN must poison the result.
        if d.is_nan() || d > m { d } else { m }
    });
    worst / scale
}

pub fn mismatch_frac(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "mismatch_frac: length mismatch");
    got.iter().zip(want).filter(|(g, w)| g != w).count() as f32 / want.len().max(1) as f32
}

// -- CPU references --

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Runs `f(first_row, rows)` over contiguous runs of `row_len`-element rows of `out`, one run per thread.
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

pub fn heat_ref(w: usize, h: usize, u0: &[f32], steps: usize) -> Vec<f32> {
    assert_eq!(u0.len(), w * h);
    let mut a = u0.to_vec();
    let mut b = vec![0.0f32; w * h];
    for _ in 0..steps {
        let src = &a;
        par_rows(&mut b, w, |first, rows| {
            for (r, out) in rows.chunks_mut(w).enumerate() {
                let y = first + r;
                let row = &src[y * w..][..w];
                let up = &src[((y + h - 1) % h) * w..][..w];
                let down = &src[((y + 1) % h) * w..][..w];
                for x in 0..w {
                    let l = row[(x + w - 1) % w];
                    let rr = row[(x + 1) % w];
                    out[x] = row[x] + ALPHA * (l + rr + up[x] + down[x] - 4.0 * row[x]);
                }
            }
        });
        std::mem::swap(&mut a, &mut b);
    }
    a
}

pub fn life_ref(w: usize, h: usize, g0: &[f32], steps: usize) -> Vec<f32> {
    assert_eq!(g0.len(), w * h);
    let mut a = g0.to_vec();
    let mut b = vec![0.0f32; w * h];
    for _ in 0..steps {
        let src = &a;
        par_rows(&mut b, w, |first, rows| {
            for (r, out) in rows.chunks_mut(w).enumerate() {
                let y = first + r;
                let up = &src[((y + h - 1) % h) * w..][..w];
                let row = &src[y * w..][..w];
                let down = &src[((y + 1) % h) * w..][..w];
                for x in 0..w {
                    let (l, rr) = ((x + w - 1) % w, (x + 1) % w);
                    let n = up[l] + up[x] + up[rr] + row[l] + row[rr] + down[l] + down[x] + down[rr];
                    out[x] = if n == 3.0 {
                        1.0
                    } else if n == 2.0 {
                        row[x]
                    } else {
                        0.0
                    };
                }
            }
        });
        std::mem::swap(&mut a, &mut b);
    }
    a
}

fn cnd(d: f32) -> f32 {
    let k = 1.0 / (1.0 + 0.2316419 * d.abs());
    let poly = k * (A[0] + k * (A[1] + k * (A[2] + k * (A[3] + k * A[4]))));
    let tail = INV_SQRT_2PI * (-0.5 * d * d).exp() * poly;
    if d > 0.0 { 1.0 - tail } else { tail }
}

pub fn bs_ref(s: &[f32], k: &[f32], t: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; s.len()];
    par_rows(&mut out, 1, |first, chunk| {
        for (i, o) in chunk.iter_mut().enumerate() {
            let (s, k, t) = (s[first + i], k[first + i], t[first + i]);
            let vol_sqrt_t = VOL * t.sqrt();
            let d1 = ((s / k).ln() + (RATE + 0.5 * VOL * VOL) * t) / vol_sqrt_t;
            let d2 = d1 - vol_sqrt_t;
            *o = s * cnd(d1) - k * (-RATE * t).exp() * cnd(d2);
        }
    });
    out
}

/// cx, cy grids: cx[x,y] = -2.2 + x*(3.2/w), cy[x,y] = -1.4 + y*(2.8/w).
pub fn mandel_c(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let (dx, dy) = (3.2 / w as f32, 2.8 / w as f32);
    let mut cx = Vec::with_capacity(w * h);
    let mut cy = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            cx.push(-2.2 + x as f32 * dx);
            cy.push(-1.4 + y as f32 * dy);
        }
    }
    (cx, cy)
}

/// Fixed-iteration escape count (no early exit across the whole grid, per-pixel freeze once outside).
pub fn mandel_ref(w: usize, h: usize, iters: usize) -> Vec<f32> {
    let (dx, dy) = (3.2 / w as f32, 2.8 / w as f32);
    let mut out = vec![0.0f32; w * h];
    par_rows(&mut out, w, |first, rows| {
        for (r, row) in rows.chunks_mut(w).enumerate() {
            let cy = -1.4 + (first + r) as f32 * dy;
            for (x, o) in row.iter_mut().enumerate() {
                let cx = -2.2 + x as f32 * dx;
                let (mut zx, mut zy, mut cnt) = (0.0f32, 0.0f32, 0.0f32);
                for _ in 0..iters {
                    if zx * zx + zy * zy > 4.0 {
                        break;
                    }
                    let nzx = zx * zx - zy * zy + cx;
                    zy = 2.0 * zx * zy + cy;
                    zx = nzx;
                    cnt += 1.0;
                }
                *o = cnt;
            }
        }
    });
    out
}

/// Full-convolution outputs `y[i] = sum_j h[j] x[i-j]` at `idx`, summed in f64.
pub fn fir_ref_sample(x: &[f32], h: &[f32], idx: &[usize]) -> Vec<f32> {
    let (n, taps) = (x.len(), h.len());
    let mut out = vec![0.0f32; idx.len()];
    par_rows(&mut out, 1, |first, chunk| {
        for (k, o) in chunk.iter_mut().enumerate() {
            let i = idx[first + k];
            let j0 = (i + 1).saturating_sub(n);
            let j1 = i.min(taps - 1);
            let mut acc = 0.0f64;
            for j in j0..=j1 {
                acc += h[j] as f64 * x[i - j] as f64;
            }
            *o = acc as f32;
        }
    });
    out
}

/// 4096 deterministic, evenly spread indices including 0 and `out_len - 1`.
pub fn fir_sample_indices(out_len: usize) -> Vec<usize> {
    const COUNT: usize = 4096;
    assert!(out_len >= 1);
    (0..COUNT).map(|i| (i as u128 * (out_len as u128 - 1) / (COUNT as u128 - 1)) as usize).collect()
}

// -- output --

pub struct Row<'a> {
    pub workload: &'a str,
    pub size: String,
    pub variant: &'a str,
    pub library: &'a str,
    pub backend: &'a str,
    pub work: f64,
    pub unit: &'a str,
    pub err: f32,
}

pub struct Timing {
    pub median_ms: f64,
    pub min_ms: f64,
    pub reps: usize,
}

/// 3 untimed warm-ups, then timed calls until (elapsed >= 1 s && reps >= 5) || reps == 200.
/// The closure must include device completion.
pub fn time(mut f: impl FnMut()) -> Timing {
    for _ in 0..3 {
        f();
    }
    let mut samples = Vec::new();
    let start = Instant::now();
    loop {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
        let reps = samples.len();
        if (start.elapsed().as_secs_f64() >= 1.0 && reps >= 5) || reps == 200 {
            break;
        }
    }
    let reps = samples.len();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = if reps % 2 == 1 { samples[reps / 2] } else { 0.5 * (samples[reps / 2 - 1] + samples[reps / 2]) };
    Timing { median_ms, min_ms: samples[0], reps }
}

pub fn print_header() {
    println!("workload,size,variant,library,backend,median_ms,min_ms,reps,throughput,unit,err");
    let _ = std::io::stdout().flush();
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_owned() }
}

pub fn print_row(row: &Row, t: &Timing) {
    let throughput = row.work / (t.median_ms / 1e3) / 1e9;
    println!(
        "{},{},{},{},{},{:.6},{:.6},{},{:.6},{},{:e}",
        csv_field(row.workload),
        csv_field(&row.size),
        csv_field(row.variant),
        csv_field(row.library),
        csv_field(row.backend),
        t.median_ms,
        t.min_ms,
        t.reps,
        throughput,
        csv_field(row.unit),
        row.err
    );
    let _ = std::io::stdout().flush();
}
