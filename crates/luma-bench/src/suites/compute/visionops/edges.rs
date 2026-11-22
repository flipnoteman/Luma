//! `canny`, `anisotropic_diffusion` and `mean_shift` at 1080p and 4K. Each is checked against a
//! small host computation (ArrayFire's CUDA kernels, step by step) before it is timed.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use luma::*;

use super::super::{host, pick, upload};
use crate::harness::*;

/// `[H, W]` sizes: 1080p and 4K (smaller under `--quick` and on the CPU backend).
fn sizes(h: &Harness) -> Vec<[usize; 2]> {
    if h.ctx.is_cpu() {
        vec![[270, 480]]
    } else {
        pick(h, vec![[1080, 1920]], vec![[1080, 1920], [2160, 3840]])
    }
}

fn pix(dims: [usize; 2], per_pixel: usize) -> Work {
    Work::Items((dims[0] * dims[1] * per_pixel) as f64, "Mpix-steps/s")
}

fn label(dims: [usize; 2]) -> String {
    format!("{}x{}", dims[1], dims[0])
}

/// An image of 0..255 with smooth structure, blocks and noise.
fn scene(h: usize, w: usize, planes: usize, seed: u64) -> Vec<f32> {
    let noise = values(h * w * planes, seed, 0.0, 12.0);
    (0..h * w * planes)
        .map(|k| {
            let (i, j, p) = ((k % h) as f32, ((k / h) % w) as f32, (k / (h * w)) as f32);
            let a = ((i * 0.021 + p).sin() + (j * 0.017 - p).cos() + ((i + j) * 0.009).sin()) * 70.0 + 128.0;
            let b = if ((i as usize / 130 + j as usize / 170 + p as usize) % 2) == 0 { 60.0 } else { 0.0 };
            (a * 0.6 + b + noise[k]).floor()
        })
        .collect()
}

// -- host references --

const G: [f32; 5] = [0.11021, 0.23691, 0.30576, 0.23691, 0.11021];

fn suppressed(img: &[f32], h: usize, w: usize, fast: bool) -> Vec<f32> {
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
    let (mut gx, mut gy, mut mag) = (vec![0.0f32; h * w], vec![0.0f32; h * w], vec![0.0f32; h * w]);
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

fn canny_host(data: &[f32], h: usize, w: usize, planes: usize, auto: bool, low: f32, high: f32, fast: bool) -> Vec<bool> {
    let sups: Vec<Vec<f32>> = (0..planes).map(|p| suppressed(&data[p * h * w..(p + 1) * h * w], h, w, fast)).collect();
    let max = sups.iter().flatten().copied().fold(0.0f32, f32::max);
    let mut out = Vec::new();
    for sup in &sups {
        let (lo, hi) = if auto { (low * otsu(sup, max), otsu(sup, max)) } else { (low * max, high * max) };
        out.extend(hysteresis(sup, h, w, lo, hi));
    }
    out
}

fn mean_shift_host(data: &[f32], h: usize, w: usize, chans: usize, planes: usize, s: f32, c: f32, iters: u32) -> Vec<f32> {
    let radius = ((s * 1.5) as i64).max(1);
    let cvar = c * c;
    let mut out = vec![0.0f32; data.len()];
    for p in 0..planes {
        let base = p * chans * h * w;
        let at = |ch: usize, i: i64, j: i64| data[base + ch * h * w + i as usize + h * j as usize];
        for j0 in 0..w {
            for i0 in 0..h {
                let (mut pi, mut pj) = (i0 as i64, j0 as i64);
                let mut cc: Vec<f32> = (0..chans).map(|ch| at(ch, pi, pj)).collect();
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
                            for (ch, v) in cc.iter().enumerate() {
                                let d = v - at(ch, ti, tj);
                                norm += d * d;
                            }
                            if norm <= cvar {
                                for (ch, v) in m.iter_mut().enumerate() {
                                    *v += at(ch, ti, tj);
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
                    for (ch, v) in m.iter_mut().enumerate() {
                        *v = ((*v * fc) as i32) as f32;
                        dist += (cc[ch] - *v) * (cc[ch] - *v);
                    }
                    let stop = (pj == oj && pi == oi) || ((((oj - pj).abs() + (oi - pi).abs()) as f32) + dist <= 1.0);
                    cc = m;
                    if stop {
                        break;
                    }
                }
                for (ch, v) in cc.iter().enumerate() {
                    out[base + ch * h * w + i0 + h * j0] = *v;
                }
            }
        }
    }
    out
}

fn diffusion_host(data: &[f32], h: usize, w: usize, dt: f32, k: f32, mcde: bool) -> Vec<f32> {
    let at = |i: i64, j: i64| data[i.clamp(0, h as i64 - 1) as usize + h * j.clamp(0, w as i64 - 1) as usize];
    let mut sum = 0.0f64;
    for j in 0..w as i64 {
        for i in 0..h as i64 {
            let f0 = if i == 0 || i >= h as i64 - 1 { 1.0 } else { 0.5 };
            let f1 = if j == 0 || j >= w as i64 - 1 { 1.0 } else { 0.5 };
            let (g0, g1) = (f0 * (at(i + 1, j) - at(i - 1, j)), f1 * (at(i, j + 1) - at(i, j - 1)));
            sum += f64::from(g0 * g0 + g1 * g1);
        }
    }
    let mct = (-((h * w) as f64) / (2.0 * f64::from(k) * f64::from(k) * sum)) as f32;
    let sq = |x: f32| x * x;
    let mut out = vec![0.0f32; data.len()];
    for j in 0..w as i64 {
        for i in 0..h as i64 {
            let (c, s, n, wv, e) = (at(i, j), at(i + 1, j), at(i - 1, j), at(i, j - 1), at(i, j + 1));
            let (se, sw, ne, nw) = (at(i + 1, j + 1), at(i + 1, j - 1), at(i - 1, j + 1), at(i - 1, j - 1));
            let (dx, dy) = ((e - wv) * 0.5, (s - n) * 0.5);
            let (mut df, mut db) = (e - c, c - wv);
            let (df0, db0) = (df, db);
            let (gf, gb) = (sq(df) + 0.25 * sq(dy + 0.5 * (se - ne)), sq(db) + 0.25 * sq(dy + 0.5 * (sw - nw)));
            let mut delta;
            if mcde {
                delta = (df / (1.0e-10 + gf).sqrt()) * (gf * mct).exp() - (db / (1.0e-10 + gb).sqrt()) * (gb * mct).exp();
                df = s - c;
                db = c - n;
                let (gf, gb) = (sq(df) + 0.25 * sq(dx + 0.5 * (se - sw)), sq(db) + 0.25 * sq(dx + 0.5 * (ne - nw)));
                delta += (df / (1.0e-10 + gf).sqrt()) * (gf * mct).exp() - (db / (1.0e-10 + gb).sqrt()) * (gb * mct).exp();
                let prop = if delta > 0.0 { sq(db0.min(0.0)) + sq(df0.max(0.0)) + sq(db.min(0.0)) + sq(df.max(0.0)) } else { sq(db0.max(0.0)) + sq(df0.min(0.0)) + sq(db.max(0.0)) + sq(df.min(0.0)) };
                delta *= prop.sqrt();
            } else {
                delta = (gf * mct).exp() * df - (gb * mct).exp() * db;
                df = s - c;
                db = c - n;
                let (gf, gb) = (sq(df) + 0.25 * sq(dx + 0.5 * (se - sw)), sq(db) + 0.25 * sq(dx + 0.5 * (ne - nw)));
                delta += (gf * mct).exp() * df - (gb * mct).exp() * db;
            }
            out[i as usize + h * j as usize] = c + delta * dt;
        }
    }
    out
}

pub fn run(h: &mut Harness) {
    h.guarded("edges", |h| {
        // Small checks against the host.
        let (sh, sw) = (61usize, 77usize);
        let sdata = scene(sh, sw, 2, 3);
        let simg = upload(h, &sdata, [sh, sw, 2]);
        for (name, auto, low, high, fast) in [("manual", false, 0.1f32, 0.3f32, false), ("otsu", true, 0.3, 0.0, false), ("fast", false, 0.1, 0.3, true)] {
            let mode = if auto { CannyThreshold::AutoOtsu } else { CannyThreshold::Manual };
            let got = simg.canny(mode, low, high, 3, fast).to_vec().unwrap();
            let want = canny_host(&sdata, sh, sw, 2, auto, low, high, fast);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert!(bad * 500 <= want.iter().filter(|&&b| b).count(), "edges/canny {name}: {bad} of {} pixels differ", want.len());
        }
        let gray = upload(h, &sdata[..sh * sw], [sh, sw]);
        assert_eq!(host(&gray.mean_shift(2.0, 30.0, 4, false)), mean_shift_host(&sdata[..sh * sw], sh, sw, 1, 1, 2.0, 30.0, 4), "edges/mean_shift");
        let rgb = upload(h, &scene(sh, sw, 3, 5), [sh, sw, 3]);
        assert_eq!(host(&rgb.mean_shift(2.0, 40.0, 3, true)), mean_shift_host(&scene(sh, sw, 3, 5), sh, sw, 3, 1, 2.0, 40.0, 3), "edges/mean_shift colour");
        for (flux, eq, mcde) in [(FluxFunction::Exponential, DiffusionEq::Gradient, false), (FluxFunction::Exponential, DiffusionEq::Mcde, true)] {
            let got = host(&gray.anisotropic_diffusion(0.1, 5.0, 1, flux, eq));
            let want = diffusion_host(&sdata[..sh * sw], sh, sw, 0.1, 5.0, mcde);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-3 + 1e-4 * w.abs(), "edges/diffusion: {g} vs {w}");
            }
        }

        for dims in sizes(h) {
            let data = scene(dims[0], dims[1], 1, 7);
            let img = upload(h, &data, dims);
            let tag = label(dims);
            h.case("edges/canny_manual", &tag, pix(dims, 1), &mut || img.canny(CannyThreshold::Manual, 0.1, 0.3, 3, false).eval().unwrap());
            h.case("edges/canny_manual_fast", &tag, pix(dims, 1), &mut || img.canny(CannyThreshold::Manual, 0.1, 0.3, 3, true).eval().unwrap());
            h.case("edges/canny_otsu", &tag, pix(dims, 1), &mut || img.canny(CannyThreshold::AutoOtsu, 0.3, 0.0, 3, false).eval().unwrap());
            h.case("edges/anisotropic_exp_x5", &tag, pix(dims, 5), &mut || img.anisotropic_diffusion(0.1, 5.0, 5, FluxFunction::Exponential, DiffusionEq::Gradient).eval().unwrap());
            h.case("edges/anisotropic_mcde_x5", &tag, pix(dims, 5), &mut || img.anisotropic_diffusion(0.1, 5.0, 5, FluxFunction::Exponential, DiffusionEq::Mcde).eval().unwrap());
            h.case("edges/mean_shift_s2_x5", &tag, pix(dims, 5), &mut || img.mean_shift(2.0, 30.0, 5, false).eval().unwrap());
            let rgb = upload(h, &scene(dims[0], dims[1], 3, 9), [dims[0], dims[1], 3]);
            h.case("edges/mean_shift_rgb_s2_x5", &tag, pix(dims, 5), &mut || rgb.mean_shift(2.0, 40.0, 5, true).eval().unwrap());
        }
    });
}
