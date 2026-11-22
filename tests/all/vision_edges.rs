//! Canny edges, anisotropic diffusion, mean shift: tests against host references.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::manual_memcpy)]

use crate::common::{assert_close, values};
use luma::{Array, Context, DiffusionEq, FluxFunction};

fn ctx() -> Context {
    Context::new().unwrap()
}

/// An image of 0..255 with some structure (blocks plus noise).
fn scene(h: usize, w: usize, planes: usize, seed: u64) -> Vec<f32> {
    let noise = values(h * w * planes, seed, 0.0, 30.0);
    let blocks = values(64, seed + 1, 0.0, 220.0);
    (0..h * w * planes)
        .map(|k| {
            let (i, j, p) = (k % h, (k / h) % w, k / (h * w));
            (blocks[((i / 7 + p) * 5 + (j / 9) * 3) % 64] + noise[k]).floor()
        })
        .collect()
}

// -- mean shift --

/// ArrayFire's CUDA mean shift for one pixel of a plane of `chans` channels (`data` indexed
/// `[chan][i + h * j]`).
#[allow(clippy::too_many_arguments)]
fn mean_shift_pixel(data: &[f32], h: usize, w: usize, chans: usize, i0: usize, j0: usize, radius: i64, cvar: f32, iters: u32) -> Vec<f32> {
    let at = |c: usize, i: i64, j: i64| data[c * h * w + i as usize + h * j as usize];
    let (mut pi, mut pj) = (i0 as i64, j0 as i64);
    let mut cc: Vec<f32> = (0..chans).map(|c| at(c, pi, pj)).collect();
    for _ in 0..iters {
        let (oi, oj) = (pi, pj);
        let (mut count, mut sx, mut sy) = (0u32, 0i64, 0i64);
        let mut m = vec![0.0f32; chans];
        for wj in -radius..=radius {
            let tj = pj + wj;
            if tj < 0 || tj >= w as i64 {
                continue;
            }
            let mut hit = 0i64;
            for wi in -radius..=radius {
                let ti = pi + wi;
                if ti < 0 || ti >= h as i64 {
                    continue;
                }
                let mut norm = 0.0f32;
                for c in 0..chans {
                    let d = cc[c] - at(c, ti, tj);
                    norm += d * d;
                }
                if norm <= cvar {
                    for c in 0..chans {
                        m[c] += at(c, ti, tj);
                    }
                    sx += ti;
                    hit += 1;
                }
            }
            count += hit as u32;
            sy += tj * hit;
        }
        if count == 0 {
            break;
        }
        let fc = 1.0f32 / count as f32;
        pi = (sx as f32 * fc) as i64;
        pj = (sy as f32 * fc) as i64;
        let mut dist = 0.0f32;
        for c in 0..chans {
            m[c] = ((m[c] * fc) as i32) as f32;
            dist += (cc[c] - m[c]) * (cc[c] - m[c]);
        }
        let stop = (pj == oj && pi == oi) || ((((oj - pj).abs() + (oi - pi).abs()) as f32) + dist <= 1.0);
        cc = m;
        if stop {
            break;
        }
    }
    cc
}

fn mean_shift_ref(data: &[f32], h: usize, w: usize, chans: usize, planes: usize, s: f32, c: f32, iters: u32) -> Vec<f32> {
    let radius = ((s * 1.5) as i64).max(1);
    let mut out = vec![0.0f32; data.len()];
    for p in 0..planes {
        let plane = &data[p * chans * h * w..(p + 1) * chans * h * w];
        for j in 0..w {
            for i in 0..h {
                let px = mean_shift_pixel(plane, h, w, chans, i, j, radius, c * c, iters);
                for (ch, v) in px.iter().enumerate() {
                    out[p * chans * h * w + ch * h * w + i + h * j] = *v;
                }
            }
        }
    }
    out
}

#[test]
fn mean_shift_matches_the_walk() {
    let ctx = ctx();
    // Gray images, one plane and batches over axes 2 and 3.
    for (h, w, planes, shape, s, c, iters) in [
        (20, 27, 1, [20, 27, 1, 1], 1.0f32, 30.0f32, 1u32),
        (20, 27, 1, [20, 27, 1, 1], 2.0, 40.0, 3),
        (33, 18, 1, [33, 18, 1, 1], 1.4, 25.0, 6),
        (11, 9, 4, [11, 9, 2, 2], 2.0, 35.0, 4),
        (40, 40, 1, [40, 40, 1, 1], 8.0, 45.0, 3),
    ] {
        let data = scene(h, w, planes, 5 + iters as u64);
        let img = Array::from_slice_on(&ctx, &data, shape).unwrap();
        let got = img.mean_shift(s, c, iters, false).to_vec().unwrap();
        assert_eq!(got, mean_shift_ref(&data, h, w, 1, planes, s, c, iters), "{h}x{w}x{planes} sigma {s} {c} iterations {iters}");
    }
    // Colour images: [H, W, 3] and a batch of two.
    for (h, w, batch, shape, s, c, iters) in [(13, 17, 1, [13, 17, 3, 1], 1.0f32, 40.0f32, 2u32), (10, 12, 2, [10, 12, 3, 2], 2.0, 60.0, 5), (30, 34, 1, [30, 34, 3, 1], 6.0, 80.0, 4)] {
        let data = scene(h, w, 3 * batch, 21);
        let img = Array::from_slice_on(&ctx, &data, shape).unwrap();
        let got = img.mean_shift(s, c, iters, true).to_vec().unwrap();
        assert_eq!(got, mean_shift_ref(&data, h, w, 3, batch, s, c, iters), "colour {h}x{w}x{batch}");
    }
}

#[test]
fn mean_shift_truncates_and_errors() {
    let ctx = ctx();
    // An image in 0..1 truncates to zero: the means are cut to integers like in ArrayFire.
    let data = values(16 * 16, 3, 0.0, 1.0);
    let img = Array::from_slice_on(&ctx, &data, [16, 16]).unwrap();
    assert!(img.mean_shift(2.0, 0.5, 2, false).to_vec().unwrap().iter().all(|&v| v == 0.0));
    // Errors: no iterations, a colour image of the wrong shape, bad sigmas.
    assert!(img.mean_shift(2.0, 30.0, 0, false).to_vec().is_err());
    assert!(img.mean_shift(2.0, 30.0, 2, true).to_vec().is_err());
    assert!(img.mean_shift(-1.0, 30.0, 2, false).to_vec().is_err());
    assert!(img.mean_shift(2.0, f32::NAN, 2, false).to_vec().is_err());
    // A zero spatial sigma is the radius 1 of ArrayFire.
    let a = img.mean_shift(0.0, 0.0, 1, false).to_vec().unwrap();
    let b = img.mean_shift(1.0, 0.0, 1, false).to_vec().unwrap();
    assert_eq!(a, b);
}

// -- anisotropic diffusion --

/// One step of ArrayFire's update at every pixel (Jacobi), in `f32` like the kernel.
fn diffusion_step(data: &[f32], h: usize, w: usize, planes: usize, dt: f32, k: f32, flux: FluxFunction, eq: DiffusionEq) -> Vec<f32> {
    let at = |p: usize, i: i64, j: i64| data[p * h * w + i.clamp(0, h as i64 - 1) as usize + h * j.clamp(0, w as i64 - 1) as usize];
    // The gradient energy.
    let mut sum = 0.0f64;
    for p in 0..planes {
        for j in 0..w as i64 {
            for i in 0..h as i64 {
                let f0 = if i == 0 || i >= h as i64 - 1 { 1.0 } else { 0.5 };
                let f1 = if j == 0 || j >= w as i64 - 1 { 1.0 } else { 0.5 };
                let g0 = f0 * (at(p, i + 1, j) - at(p, i - 1, j));
                let g1 = f1 * (at(p, i, j + 1) - at(p, i, j - 1));
                sum += f64::from(g0 * g0 + g1 * g1);
            }
        }
    }
    let n = (h * w * planes) as f64;
    let mct = if sum > 0.0 { (-n / (2.0 * f64::from(k) * f64::from(k) * sum)) as f32 } else { 0.0 };
    let cond = |v: f32| match (eq, flux) {
        (DiffusionEq::Gradient, FluxFunction::Quadratic) => 1.0 / (1.0 + v),
        _ => v.exp(),
    };
    let mut out = vec![0.0f32; data.len()];
    for p in 0..planes {
        for j in 0..w as i64 {
            for i in 0..h as i64 {
                let (c, s, nn, ww, e) = (at(p, i, j), at(p, i + 1, j), at(p, i - 1, j), at(p, i, j - 1), at(p, i, j + 1));
                let (se, sw, ne, nw) = (at(p, i + 1, j + 1), at(p, i + 1, j - 1), at(p, i - 1, j + 1), at(p, i - 1, j - 1));
                let dx = (e - ww) * 0.5;
                let dy = (s - nn) * 0.5;
                let sq = |x: f32| x * x;
                let (mut df, mut db) = (e - c, c - ww);
                let (df0, db0) = (df, db);
                let (gf, gb) = (sq(df) + 0.25 * sq(dy + 0.5 * (se - ne)), sq(db) + 0.25 * sq(dy + 0.5 * (sw - nw)));
                let mut delta;
                if eq == DiffusionEq::Mcde {
                    delta = (df / (1.0e-10 + gf).sqrt()) * (gf * mct).exp() - (db / (1.0e-10 + gb).sqrt()) * (gb * mct).exp();
                    df = s - c;
                    db = c - nn;
                    let (gf, gb) = (sq(df) + 0.25 * sq(dx + 0.5 * (se - sw)), sq(db) + 0.25 * sq(dx + 0.5 * (ne - nw)));
                    delta += (df / (1.0e-10 + gf).sqrt()) * (gf * mct).exp() - (db / (1.0e-10 + gb).sqrt()) * (gb * mct).exp();
                    let prop = if delta > 0.0 {
                        sq(db0.min(0.0)) + sq(df0.max(0.0)) + sq(db.min(0.0)) + sq(df.max(0.0))
                    } else {
                        sq(db0.max(0.0)) + sq(df0.min(0.0)) + sq(db.max(0.0)) + sq(df.min(0.0))
                    };
                    delta *= prop.sqrt();
                } else {
                    delta = cond(gf * mct) * df - cond(gb * mct) * db;
                    df = s - c;
                    db = c - nn;
                    let (gf, gb) = (sq(df) + 0.25 * sq(dx + 0.5 * (se - sw)), sq(db) + 0.25 * sq(dx + 0.5 * (ne - nw)));
                    delta += cond(gf * mct) * df - cond(gb * mct) * db;
                }
                out[p * h * w + i as usize + h * j as usize] = c + delta * dt;
            }
        }
    }
    out
}

#[test]
fn anisotropic_diffusion_matches_the_scheme() {
    let ctx = ctx();
    let cases = [
        ([23, 41, 1, 1], 0.1f32, 8.0f32, 1u32, FluxFunction::Exponential, DiffusionEq::Gradient),
        ([40, 33, 1, 1], 0.08, 5.0, 3, FluxFunction::Exponential, DiffusionEq::Gradient),
        ([17, 30, 1, 1], 0.05, 40.0, 2, FluxFunction::Quadratic, DiffusionEq::Gradient),
        ([35, 20, 1, 1], 0.1, 6.0, 4, FluxFunction::Exponential, DiffusionEq::Mcde),
        ([19, 22, 3, 1], 0.1, 6.0, 3, FluxFunction::Exponential, DiffusionEq::Gradient),
        ([15, 21, 2, 2], 0.1, 6.0, 2, FluxFunction::Exponential, DiffusionEq::Mcde),
    ];
    for (shape, dt, k, iters, flux, eq) in cases {
        let (h, w) = (shape[0], shape[1]);
        let planes: usize = shape[2] * shape[3];
        let data = scene(h, w, planes, 11);
        let img = Array::from_slice_on(&ctx, &data, shape).unwrap();
        let got = img.anisotropic_diffusion(dt, k, iters, flux, eq).to_vec().unwrap();
        let mut want = data.clone();
        for _ in 0..iters {
            want = diffusion_step(&want, h, w, planes, dt, k, flux, eq);
        }
        assert_close(&got, &want, 2e-3, 2e-3 * 255.0 * 0.05);
    }
}

#[test]
fn anisotropic_diffusion_properties() {
    let ctx = ctx();
    // A constant image does not change (ArrayFire returns NaN), the mean is kept by the gradient flux.
    let flat = Array::<f32>::full_on(&ctx, [10, 12], 5.0).unwrap();
    assert!(flat.anisotropic_diffusion(0.1, 3.0, 4, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec().unwrap().iter().all(|&v| v == 5.0));
    let data = scene(24, 24, 1, 4);
    let img = Array::from_slice_on(&ctx, &data, [24, 24]).unwrap();
    let out = img.anisotropic_diffusion(0.1, 8.0, 5, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec().unwrap();
    let (a, b): (f32, f32) = (data.iter().sum(), out.iter().sum());
    assert!((a - b).abs() < 1e-3 * a, "mean kept: {a} vs {b}");
    // Errors.
    assert!(img.anisotropic_diffusion(0.1, 8.0, 0, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec().is_err());
    assert!(img.anisotropic_diffusion(0.1, 0.0, 1, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec().is_err());
    assert!(img.anisotropic_diffusion(f32::NAN, 1.0, 1, FluxFunction::Exponential, DiffusionEq::Gradient).to_vec().is_err());
}

// -- Canny --

const G: [f32; 5] = [0.11021, 0.23691, 0.30576, 0.23691, 0.11021];

/// The suppressed gradient magnitudes of one `[h, w]` plane (ArrayFire's CUDA pipeline, in the
/// operation order of the kernels).
fn canny_suppressed(img: &[f32], h: usize, w: usize, fast: bool) -> Vec<f32> {
    let at = |i: i64, j: i64| if i < 0 || j < 0 || i >= h as i64 || j >= w as i64 { 0.0 } else { img[i as usize + h * j as usize] };
    let mut hs = vec![0.0f32; h * w];
    for j in 0..w as i64 {
        for i in 0..h as i64 {
            hs[i as usize + h * j as usize] = at(i - 2, j).mul_add(G[0], at(i - 1, j).mul_add(G[1], at(i, j).mul_add(G[2], at(i + 1, j).mul_add(G[1], at(i + 2, j) * G[0]))));
        }
    }
    let at2 = |i: i64, j: i64| if i < 0 || j < 0 || i >= h as i64 || j >= w as i64 { 0.0 } else { hs[i as usize + h * j as usize] };
    let mut sm = vec![0.0f32; h * w];
    for j in 0..w as i64 {
        for i in 0..h as i64 {
            sm[i as usize + h * j as usize] = at2(i, j - 2).mul_add(G[0], at2(i, j - 1).mul_add(G[1], at2(i, j).mul_add(G[2], at2(i, j + 1).mul_add(G[1], at2(i, j + 2) * G[0]))));
        }
    }
    let refl = |p: i64, n: i64| if p < 0 { -p } else if p >= n { 2 * (n - 1) - p } else { p };
    let s = |i: i64, j: i64| sm[refl(i, h as i64) as usize + h * refl(j, w as i64) as usize];
    let mut gx = vec![0.0f32; h * w];
    let mut gy = vec![0.0f32; h * w];
    let mut mag = vec![0.0f32; h * w];
    for j in 0..w as i64 {
        for i in 0..h as i64 {
            let dx = ((s(i + 1, j - 1) + s(i + 1, j + 1)) - (s(i - 1, j - 1) + s(i - 1, j + 1))) + 2.0 * (s(i + 1, j) - s(i - 1, j));
            let dy = ((s(i - 1, j + 1) + s(i + 1, j + 1)) - (s(i - 1, j - 1) + s(i + 1, j - 1))) + 2.0 * (s(i, j + 1) - s(i, j - 1));
            let k = i as usize + h * j as usize;
            gx[k] = dx;
            gy[k] = dy;
            mag[k] = if fast { dx.abs() + dy.abs() } else { (dx * dx + dy * dy).sqrt() };
        }
    }
    let m = |i: usize, j: usize| mag[i + h * j];
    let mut out = vec![0.0f32; h * w];
    for j in 1..w - 1 {
        for i in 1..h - 1 {
            let k = i + h * j;
            let (dx, dy, c) = (gx[k], gy[k], mag[k]);
            if c <= 0.0 {
                continue;
            }
            let (se, nw, ea, we, ne, sw, no, so) = (m(i + 1, j + 1), m(i - 1, j - 1), m(i + 1, j), m(i - 1, j), m(i + 1, j - 1), m(i - 1, j + 1), m(i, j - 1), m(i, j + 1));
            let (a1, a2, b1, b2, alpha);
            if dx >= 0.0 {
                if dy >= 0.0 {
                    let big = dx - dy >= 0.0;
                    (a1, a2, b1, b2) = (if big { ea } else { so }, if big { we } else { no }, se, nw);
                    alpha = if big { dy / dx } else { dx / dy };
                } else {
                    let big = dx + dy >= 0.0;
                    (a1, a2, b1, b2) = (if big { ea } else { no }, if big { we } else { so }, ne, sw);
                    alpha = if big { -dy / dx } else { dx / -dy };
                }
            } else if dy >= 0.0 {
                let big = dx + dy >= 0.0;
                (a1, a2, b1, b2) = (if big { so } else { we }, if big { no } else { ea }, sw, ne);
                alpha = if big { -dx / dy } else { dy / -dx };
            } else {
                let big = -dx + dy >= 0.0;
                (a1, a2, b1, b2) = (if big { we } else { no }, if big { ea } else { so }, nw, se);
                alpha = if big { dy / dx } else { dx / dy };
            }
            let (m1, m2) = ((1.0 - alpha) * a1 + alpha * b1, (1.0 - alpha) * a2 + alpha * b2);
            if c > m1 && c > m2 {
                out[k] = c;
            }
        }
    }
    out
}

/// Hysteresis of one plane: strong `>= hi`, weak `>= lo`, 8-connected, interior only.
fn hysteresis(sup: &[f32], h: usize, w: usize, lo: f32, hi: f32) -> Vec<bool> {
    let interior = |i: usize, j: usize| i >= 1 && j >= 1 && i + 1 < h && j + 1 < w;
    let mut out = vec![false; h * w];
    let mut stack = Vec::new();
    for j in 0..w {
        for i in 0..h {
            if interior(i, j) && sup[i + h * j] >= hi {
                out[i + h * j] = true;
                stack.push((i, j));
            }
        }
    }
    while let Some((i, j)) = stack.pop() {
        for dj in -1i64..=1 {
            for di in -1i64..=1 {
                let (a, b) = ((i as i64 + di) as usize, (j as i64 + dj) as usize);
                if a < h && b < w && interior(a, b) && !out[a + h * b] && sup[a + h * b] >= lo {
                    out[a + h * b] = true;
                    stack.push((a, b));
                }
            }
        }
    }
    out
}

/// Otsu's threshold (as ArrayFire: `floor(max)` bins, zeros counted, first maximum).
fn otsu(sup: &[f32], max: f32) -> f32 {
    let nb = (max as usize).clamp(1, 16384);
    let step = max / nb as f32;
    let mut hist = vec![0u64; nb];
    for &v in sup {
        hist[((v / step) as usize).min(nb - 1)] += 1;
    }
    let n = sup.len() as f64;
    let total: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut n0, mut s0, mut best, mut bi) = (0.0f64, 0.0f64, -1.0f64, 0usize);
    for i in 0..nb - 1 {
        n0 += hist[i] as f64;
        s0 += i as f64 * hist[i] as f64;
        let n1 = n - n0;
        if n0 > 0.0 && n1 > 0.0 {
            let (mu0, mu1) = (s0 / n0, (total - s0) / n1);
            let sb = (n0 / n) * (n1 / n) * (mu0 - mu1) * (mu0 - mu1);
            if sb > best * (1.0 + 1e-9) {
                best = sb;
                bi = i;
            }
        }
    }
    bi as f32 * step
}

/// An image with edges: smooth blobs plus a little noise.
fn blobs(h: usize, w: usize, planes: usize, seed: u64) -> Vec<f32> {
    let noise = values(h * w * planes, seed, 0.0, 12.0);
    (0..h * w * planes)
        .map(|k| {
            let (i, j, p) = ((k % h) as f32, ((k / h) % w) as f32, (k / (h * w)) as f32);
            let a = ((i * 0.21 + p).sin() + (j * 0.17 - p).cos() + ((i + j) * 0.09).sin()) * 70.0 + 128.0;
            let b = if ((i as usize / 13 + j as usize / 17 + p as usize) % 2) == 0 { 60.0 } else { 0.0 };
            a * 0.6 + b + noise[k]
        })
        .collect()
}

fn canny_ref(data: &[f32], h: usize, w: usize, planes: usize, otsu_mode: bool, low: f32, high: f32, fast: bool) -> Vec<bool> {
    let sups: Vec<Vec<f32>> = (0..planes).map(|p| canny_suppressed(&data[p * h * w..(p + 1) * h * w], h, w, fast)).collect();
    let max = sups.iter().flatten().copied().fold(0.0f32, f32::max);
    let mut out = Vec::new();
    for sup in &sups {
        let (lo, hi) = if otsu_mode {
            let t = otsu(sup, max);
            (low * t, t)
        } else {
            (low * max, high * max)
        };
        out.extend(hysteresis(sup, h, w, lo, hi));
    }
    out
}

fn mismatches(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[test]
fn canny_matches_the_pipeline() {
    use luma::CannyThreshold;
    let ctx = ctx();
    for (shape, otsu_mode, low, high, fast) in [
        ([48, 48, 1, 1], false, 0.1f32, 0.3f32, false),
        ([40, 50, 1, 1], false, 0.1, 0.3, true),
        ([33, 65, 1, 1], false, 0.2, 0.5, false),
        ([100, 70, 1, 1], true, 0.3, 0.0, false),
        ([64, 64, 1, 1], true, 0.5, 0.0, true),
        ([50, 45, 3, 1], false, 0.1, 0.3, false),
        ([37, 41, 2, 2], true, 0.3, 0.0, false),
        ([70, 70, 1, 1], false, 0.0, 0.0, false),
        ([9, 3, 1, 1], false, 0.0, 0.0, false),
        ([4, 7, 2, 1], false, 0.0, 0.0, true),
        ([3, 3, 1, 1], false, 0.0, 0.0, false),
    ] {
        let (h, w, planes) = (shape[0], shape[1], shape[2] * shape[3]);
        let data = blobs(h, w, planes, 5 + h as u64);
        let img = Array::from_slice_on(&ctx, &data, shape).unwrap();
        let mode = if otsu_mode { CannyThreshold::AutoOtsu } else { CannyThreshold::Manual };
        let got = img.canny(mode, low, high, 3, fast).to_vec().unwrap();
        let want = canny_ref(&data, h, w, planes, otsu_mode, low, high, fast);
        assert!(want.iter().filter(|&&b| b).count() > 0, "the case has edges");
        let bad = mismatches(&got, &want);
        assert!(bad == 0, "{shape:?} otsu {otsu_mode} {low} {high} fast {fast}: {bad} of {} edge pixels differ ({} wanted)", got.iter().filter(|&&b| b).count(), want.iter().filter(|&&b| b).count());
    }
}

#[test]
fn canny_edge_cases() {
    use luma::CannyThreshold;
    let ctx = ctx();
    // Images without an interior and bad arguments.
    for shape in [[2, 8, 1, 1], [5, 2, 1, 1], [1, 1, 1, 1]] {
        let img = Array::<f32>::full_on(&ctx, shape, 3.0).unwrap();
        let out = img.canny(CannyThreshold::Manual, 0.1, 0.3, 3, false).to_vec().unwrap();
        assert!(out.iter().all(|&b| !b), "{shape:?}");
    }
    let img = Array::from_slice_on(&ctx, &blobs(30, 30, 1, 2), [30, 30]).unwrap();
    for window in [1, 5, 7] {
        assert!(img.canny(CannyThreshold::Manual, 0.1, 0.3, window, false).to_vec().is_err());
    }
    assert!(img.canny(CannyThreshold::Manual, f32::NAN, 0.3, 3, false).to_vec().is_err());
    // The ring is never an edge, even with thresholds 0 (every interior pixel is one).
    let out = img.canny(CannyThreshold::Manual, 0.0, 0.0, 3, false).to_vec().unwrap();
    for j in 0..30 {
        for i in 0..30 {
            assert_eq!(out[i + 30 * j], i > 0 && j > 0 && i < 29 && j < 29);
        }
    }
    // A constant image has edges near the border only (the smoothing pads with zeros; ArrayFire's fails on it).
    let flat = Array::<f32>::full_on(&ctx, [20, 20], 7.0).unwrap();
    let out = flat.canny(CannyThreshold::Manual, 0.1, 0.3, 3, false).to_vec().unwrap();
    assert_eq!(out, canny_ref(&vec![7.0; 400], 20, 20, 1, false, 0.1, 0.3, false));
    assert!(!out[10 + 20 * 10]);
    // A planes of a batch with different contrast: the thresholds follow the strongest of all.
    let a = blobs(40, 40, 1, 9);
    let faint: Vec<f32> = a.iter().map(|v| v * 0.1).collect();
    let both: Vec<f32> = a.iter().chain(&faint).copied().collect();
    let img = Array::from_slice_on(&ctx, &both, [40, 40, 2]).unwrap();
    let out = img.canny(CannyThreshold::Manual, 0.1, 0.3, 3, false).to_vec().unwrap();
    assert_eq!(out, canny_ref(&both, 40, 40, 2, false, 0.1, 0.3, false));
    assert!(out[1600..].iter().filter(|&&b| b).count() < out[..1600].iter().filter(|&&b| b).count());
}

