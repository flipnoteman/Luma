//! Canny edges, anisotropic diffusion, mean shift: tests against host references.

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
    for (h, w, batch, shape, s, c, iters) in [(13, 17, 1, [13, 17, 3, 1], 1.0f32, 40.0f32, 2u32), (10, 12, 2, [10, 12, 3, 2], 2.0, 60.0, 5)] {
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
                    delta = prop.sqrt() * delta;
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
