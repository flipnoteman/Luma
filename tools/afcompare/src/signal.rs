//! Signal and image cases: `approx1`/`approx2`, `resize`, `rotate`, `transform`, `fir` and `iir`.
//! Group names (for `--only`): `signal`, or `signal_approx`, `signal_image`, `signal_fir`, `signal_iir`.

use arrayfire as af;
use luma::Interp;

use crate::afx;
use crate::bench::Bench;
use crate::cases::{attempt, human};
use crate::lumax;
use crate::util::*;

fn im(m: Interp) -> af::InterpType {
    match m {
        Interp::Nearest => af::InterpType::NEAREST,
        Interp::Lower => af::InterpType::LOWER,
        Interp::Linear => af::InterpType::LINEAR,
        Interp::Cosine => af::InterpType::LINEAR_COSINE,
        Interp::Cubic => af::InterpType::CUBIC,
        Interp::CubicSpline => af::InterpType::CUBIC_SPLINE,
        Interp::LinearHardware => af::InterpType::LINEAR,
    }
}

/// The 2-D name of a method.
fn im2(m: Interp) -> af::InterpType {
    match m {
        Interp::Nearest => af::InterpType::NEAREST,
        Interp::Lower => af::InterpType::LOWER,
        Interp::Linear => af::InterpType::BILINEAR,
        Interp::Cosine => af::InterpType::BILINEAR_COSINE,
        Interp::Cubic => af::InterpType::BICUBIC,
        Interp::CubicSpline => af::InterpType::BICUBIC_SPLINE,
        Interp::LinearHardware => af::InterpType::BILINEAR,
    }
}

fn name(m: Interp) -> &'static str {
    match m {
        Interp::Nearest => "nearest",
        Interp::Linear => "linear",
        Interp::Cubic => "cubic",
        _ => "other",
    }
}

/// `luma` against `af` like [`compare`], except that up to 2e-5 of the elements may differ: pixels
/// whose source coordinate lies within rounding of an edge of the image (or of a rounding tie of
/// `Nearest`) are in or out of the image depending on the last bit of the coordinate.
fn compare_edges(luma: &[f32], af: &[f32], rel: f64, abs: f64) -> Verdict {
    if luma.len() != af.len() {
        return Verdict::no(format!("length: luma {} vs ArrayFire {}", luma.len(), af.len()));
    }
    let bad = luma.iter().zip(af).filter(|(l, a)| {
        let d = (f64::from(**l) - f64::from(**a)).abs();
        d.is_nan() || d > abs + rel * f64::from(**a).abs()
    });
    let count = bad.count();
    let allowed = (luma.len() as f64 * 2e-5).ceil() as usize;
    if count == 0 {
        compare(luma, af, rel, abs)
    } else if count <= allowed {
        Verdict::yes(format!("yes ({count} of {} pixels at an edge differ)", luma.len()))
    } else {
        compare(luma, af, rel, abs)
    }
}

fn dims2(w: usize, h: usize) -> [u64; 2] {
    [w as u64, h as u64]
}

pub fn signal(b: &mut Bench) {
    approx(b);
    image(b);
    fir(b);
    iir(b);
}

fn approx(b: &mut Bench) {
    b.guarded("signal_approx", |b| {
        let ctx = b.ctx.clone();
        for n in [1_000_000usize, 4_000_000, 16_000_000] {
            let x = uniform(n, 100, -1.0, 1.0);
            // Resampling positions: increasing with jitter, a few outside the grid at the end.
            let jitter = uniform(n, 101, 0.0, 0.5);
            let pos: Vec<f32> = (0..n).map(|i| i as f32 * (1.0 - 1.0 / n as f32) + jitter[i]).collect();
            let (xl, pl) = (lumax::arr(&ctx, &x, n), lumax::arr(&ctx, &pos, n));
            let (xa, pa) = (afx::arr(&x, &[n as u64]), afx::arr(&pos, &[n as u64]));
            for m in [Interp::Nearest, Interp::Linear, Interp::Cubic] {
                let lo = lumax::host(&xl.approx1(&pl, m, 0.0));
                let ao = afx::host(&af::approx1(&xa, &pa, im(m), 0.0));
                let v = compare(&lo, &ao, 1e-5, 1e-5);
                b.run(
                    &format!("approx1 {}", name(m)),
                    &format!("{} samples, {} positions", human(n), human(n)),
                    v,
                    "increasing jittered positions",
                    &mut || {
                        xl.approx1(&pl, m, 0.0).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af::approx1(&xa, &pa, im(m), 0.0).eval();
                        afx::sync();
                    },
                );
            }
        }
        // Random positions (a gather with no locality).
        let n = 4_000_000usize;
        let x = uniform(n, 102, -1.0, 1.0);
        let pos = uniform(n, 103, 0.0, (n - 1) as f32);
        let (xl, pl) = (lumax::arr(&ctx, &x, n), lumax::arr(&ctx, &pos, n));
        let (xa, pa) = (afx::arr(&x, &[n as u64]), afx::arr(&pos, &[n as u64]));
        let v = compare(&lumax::host(&xl.approx1(&pl, Interp::Linear, 0.0)), &afx::host(&af::approx1(&xa, &pa, af::InterpType::LINEAR, 0.0)), 1e-5, 1e-5);
        b.run(
            "approx1 linear",
            "4M samples, 4M random positions",
            v,
            "random positions",
            &mut || {
                xl.approx1(&pl, Interp::Linear, 0.0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::approx1(&xa, &pa, af::InterpType::LINEAR, 0.0).eval();
                afx::sync();
            },
        );
        drop((xl, pl, xa, pa));
        // Batched: 1024 signals of 4096 samples, one position vector each.
        let (len, batch) = (4096usize, 1024usize);
        let x = uniform(len * batch, 104, -1.0, 1.0);
        let pos: Vec<f32> = (0..len * batch).map(|i| (i % len) as f32 * 0.999 + 0.2).collect();
        let (xl, pl) = (lumax::arr(&ctx, &x, [len, batch]), lumax::arr(&ctx, &pos, [len, batch]));
        let (xa, pa) = (afx::arr(&x, &dims2(len, batch)), afx::arr(&pos, &dims2(len, batch)));
        let v = compare(&lumax::host(&xl.approx1(&pl, Interp::Linear, 0.0)), &afx::host(&af::approx1(&xa, &pa, af::InterpType::LINEAR, 0.0)), 1e-5, 1e-5);
        b.run(
            "approx1 linear, batched",
            "1024 x 4096 samples (4M positions)",
            v,
            "",
            &mut || {
                xl.approx1(&pl, Interp::Linear, 0.0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::approx1(&xa, &pa, af::InterpType::LINEAR, 0.0).eval();
                afx::sync();
            },
        );
        drop((xl, pl, xa, pa));

        for (w, h, label) in [(1920usize, 1080usize, "1080p"), (3840, 2160, "4K")] {
            let img = uniform(w * h, 105, 0.0, 1.0);
            // A smooth warp: positions near the pixel grid.
            let mut p0 = Vec::with_capacity(w * h);
            let mut p1 = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    p0.push(x as f32 + 3.0 * (y as f32 * 0.02).sin());
                    p1.push(y as f32 + 3.0 * (x as f32 * 0.02).cos());
                }
            }
            let il = lumax::arr(&ctx, &img, [w, h]);
            let (p0l, p1l) = (lumax::arr(&ctx, &p0, [w, h]), lumax::arr(&ctx, &p1, [w, h]));
            let ia = afx::arr(&img, &dims2(w, h));
            let (p0a, p1a) = (afx::arr(&p0, &dims2(w, h)), afx::arr(&p1, &dims2(w, h)));
            for m in [Interp::Nearest, Interp::Linear, Interp::Cubic] {
                let lo = lumax::host(&il.approx2(&p0l, &p1l, m, 0.0));
                let ao = afx::host(&af::approx2(&ia, &p0a, &p1a, im2(m), 0.0));
                let v = compare_edges(&lo, &ao, 1e-4, 1e-4);
                b.run(
                    &format!("approx2 {}", if m == Interp::Linear { "bilinear" } else if m == Interp::Cubic { "bicubic" } else { "nearest" }),
                    &format!("{label} image, {label} positions"),
                    v,
                    "smooth warp",
                    &mut || {
                        il.approx2(&p0l, &p1l, m, 0.0).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af::approx2(&ia, &p0a, &p1a, im2(m), 0.0).eval();
                        afx::sync();
                    },
                );
            }
            // The texture unit's own bilinear filter (an explicit opt-in of luma, weights of 8 bits
            // of fraction) against the exact ArrayFire result.
            let lo = lumax::host(&il.approx2(&p0l, &p1l, Interp::LinearHardware, 0.0));
            let ao = afx::host(&af::approx2(&ia, &p0a, &p1a, im2(Interp::Linear), 0.0));
            let v = compare_edges(&lo, &ao, 1e-2, 1e-2);
            b.run(
                "approx2 bilinear, hardware filter",
                &format!("{label} image, {label} positions"),
                v,
                "luma opt-in `Interp::LinearHardware` (8-bit weights) vs AF bilinear; smooth warp",
                &mut || {
                    il.approx2(&p0l, &p1l, Interp::LinearHardware, 0.0).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::approx2(&ia, &p0a, &p1a, im2(Interp::Linear), 0.0).eval();
                    afx::sync();
                },
            );
            // Scattered positions: every pixel samples within 48 pixels of itself (data-dependent
            // reads with 2-D locality at the scale of the cache, none of a warp's).
            let j0 = uniform(w * h, 106, -48.0, 48.0);
            let j1 = uniform(w * h, 107, -48.0, 48.0);
            let s0: Vec<f32> = p0.iter().zip(&j0).map(|(p, j)| (p + j).clamp(0.0, (w - 1) as f32)).collect();
            let s1: Vec<f32> = p1.iter().zip(&j1).map(|(p, j)| (p + j).clamp(0.0, (h - 1) as f32)).collect();
            let (s0l, s1l) = (lumax::arr(&ctx, &s0, [w, h]), lumax::arr(&ctx, &s1, [w, h]));
            let (s0a, s1a) = (afx::arr(&s0, &dims2(w, h)), afx::arr(&s1, &dims2(w, h)));
            for m in [Interp::Linear, Interp::Cubic] {
                let lo = lumax::host(&il.approx2(&s0l, &s1l, m, 0.0));
                let ao = afx::host(&af::approx2(&ia, &s0a, &s1a, im2(m), 0.0));
                let v = compare_edges(&lo, &ao, 1e-4, 1e-4);
                b.run(
                    &format!("approx2 {}, scattered", if m == Interp::Linear { "bilinear" } else { "bicubic" }),
                    &format!("{label} image, {label} positions"),
                    v,
                    "positions within 48 px of the pixel, random",
                    &mut || {
                        il.approx2(&s0l, &s1l, m, 0.0).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af::approx2(&ia, &s0a, &s1a, im2(m), 0.0).eval();
                        afx::sync();
                    },
                );
            }
        }
    });
}

fn image(b: &mut Bench) {
    b.guarded("signal_image", |b| {
        let ctx = b.ctx.clone();
        let (w1, h1, w4, h4) = (1920usize, 1080usize, 3840usize, 2160usize);
        // resize: grey up and down with every method; RGB bilinear.
        for (c, label) in [(1usize, ""), (3, " RGB")] {
            let methods: &[Interp] = if c == 1 { &[Interp::Nearest, Interp::Linear, Interp::Cubic] } else { &[Interp::Linear] };
            for (ws, hs, wd, hd, dir) in [(w1, h1, w4, h4, "1080p -> 4K"), (w4, h4, w1, h1, "4K -> 1080p")] {
                let img = uniform(ws * hs * c, 110, 0.0, 1.0);
                let il = lumax::arr(&ctx, &img, [ws, hs, c]);
                let ia = afx::arr(&img, &[ws as u64, hs as u64, c as u64]);
                for &m in methods {
                    let lo = lumax::host(&il.resize(wd, hd, m));
                    let ao = afx::host(&af::resize(&ia, wd as i64, hd as i64, im2(m)));
                    let v = compare_edges(&lo, &ao, 1e-4, 1e-4);
                    b.run(
                        &format!("resize {}{label}", match m { Interp::Linear => "bilinear", Interp::Cubic => "bicubic", _ => "nearest" }),
                        dir,
                        v,
                        "",
                        &mut || {
                            il.resize(wd, hd, m).eval().unwrap();
                            ctx.sync().unwrap();
                        },
                        &mut || {
                            af::resize(&ia, wd as i64, hd as i64, im2(m)).eval();
                            afx::sync();
                        },
                    );
                }
                if c == 1 {
                    // The texture unit's bilinear filter (luma opt-in, 8-bit weights) against exact ArrayFire.
                    let lo = lumax::host(&il.resize(wd, hd, Interp::LinearHardware));
                    let ao = afx::host(&af::resize(&ia, wd as i64, hd as i64, im2(Interp::Linear)));
                    let v = compare_edges(&lo, &ao, 1e-2, 1e-2);
                    b.run(
                        "resize bilinear, hardware filter",
                        dir,
                        v,
                        "luma opt-in `Interp::LinearHardware` (8-bit weights) vs AF bilinear",
                        &mut || {
                            il.resize(wd, hd, Interp::LinearHardware).eval().unwrap();
                            ctx.sync().unwrap();
                        },
                        &mut || {
                            af::resize(&ia, wd as i64, hd as i64, im2(Interp::Linear)).eval();
                            afx::sync();
                        },
                    );
                }
            }
        }
        // rotate and transform.
        let theta = 0.3f32;
        for (w, h, label) in [(w1, h1, "1080p"), (w4, h4, "4K")] {
            let img = uniform(w * h, 111, 0.0, 1.0);
            let il = lumax::arr(&ctx, &img, [w, h]);
            let ia = afx::arr(&img, &dims2(w, h));
            // The matrix luma's rotate applies (cropped), for ArrayFire's transform: ArrayFire's own
            // rotate() rounds the cosine and sine to three decimals, so the two disagree by a fraction of a pixel.
            let (c, s) = ((-theta).cos(), (-theta).sin());
            let (nx, ny) = (0.5 * (w as f32 - 1.0), 0.5 * (h as f32 - 1.0));
            let (tx, ty) = (nx - (nx * c - ny * s), ny - (nx * s + ny * c));
            let tmat = [c, -s, tx, s, c, ty];
            let ta = afx::arr(&tmat, &[3, 2]);
            for m in [Interp::Nearest, Interp::Linear] {
                let lo = lumax::host(&il.rotate(theta, true, m));
                let via_transform = afx::host(&af::transform(&ia, &ta, w as i64, h as i64, im2(m), true));
                let v = compare_edges(&lo, &via_transform, 1e-4, 1e-4);
                let nm = if m == Interp::Linear { "bilinear" } else { "nearest" };
                b.run(
                    &format!("rotate {nm}"),
                    &format!("{label}, 0.3 rad, crop"),
                    v,
                    "AF rotate() timed; agreement checked against AF transform() with luma's matrix (AF rotate() rounds cos/sin to 3 decimals)",
                    &mut || {
                        il.rotate(theta, true, m).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af::rotate(&ia, f64::from(theta), true, im2(m)).eval();
                        afx::sync();
                    },
                );
            }
            // Affine transform (scale + shear + translation), bilinear, inverse map.
            let m6 = [0.9f32, -0.3, 40.0, 0.25, 1.1, -20.0];
            let ta = afx::arr(&m6, &[3, 2]);
            let lo = lumax::host(&il.transform(m6, w, h, Interp::Linear, true));
            let ao = afx::host(&af::transform(&ia, &ta, w as i64, h as i64, af::InterpType::BILINEAR, true));
            let v = compare_edges(&lo, &ao, 1e-4, 1e-4);
            b.run(
                "transform affine bilinear",
                &format!("{label}, inverse map"),
                v,
                "",
                &mut || {
                    il.transform(m6, w, h, Interp::Linear, true).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::transform(&ia, &ta, w as i64, h as i64, af::InterpType::BILINEAR, true).eval();
                    afx::sync();
                },
            );
            let m9 = [1.0f32, 0.05, 20.0, -0.04, 1.0, 10.0, 1e-5, 2e-5, 1.0];
            let ta9 = afx::arr(&m9, &[3, 3]);
            let lo = lumax::host(&il.transform(m9, w, h, Interp::Linear, true));
            let ao = afx::host(&af::transform(&ia, &ta9, w as i64, h as i64, af::InterpType::BILINEAR, true));
            // The projective division rounds differently on the two sides (coordinates of thousands of pixels).
            let v = compare_edges(&lo, &ao, 1e-3, 1e-3);
            b.run(
                "transform perspective bilinear",
                &format!("{label}, inverse map"),
                v,
                "",
                &mut || {
                    il.transform(m9, w, h, Interp::Linear, true).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::transform(&ia, &ta9, w as i64, h as i64, af::InterpType::BILINEAR, true).eval();
                    afx::sync();
                },
            );
        }
        // translate, scale, skew have no separate ArrayFire kernels (they call transform), so one row each at 1080p.
        let img = uniform(w1 * h1, 112, 0.0, 1.0);
        let il = lumax::arr(&ctx, &img, [w1, h1]);
        let ia = afx::arr(&img, &dims2(w1, h1));
        let lo = lumax::host(&il.scale(0.7, 1.3, 0, 0, Interp::Linear));
        let ao = afx::host(&af::scale(&ia, 0.7, 1.3, 0, 0, af::InterpType::BILINEAR));
        // AF's scale matrix entries are not rounded here, but its output extent may differ.
        let v = if lo.len() == ao.len() { compare_edges(&lo, &ao, 1e-3, 1e-3) } else { Verdict::no(format!("length {} vs {}", lo.len(), ao.len())) };
        b.run(
            "scale bilinear",
            "1080p, 0.7 x 1.3",
            v,
            "",
            &mut || {
                il.scale(0.7, 1.3, 0, 0, Interp::Linear).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::scale(&ia, 0.7, 1.3, 0, 0, af::InterpType::BILINEAR).eval();
                afx::sync();
            },
        );
    });
}

/// `y = b * x` zero-padded, in `f64`, for one signal.
fn fir_ref(x: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut y = vec![0.0f32; n];
    for (i, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (k, &c) in b.iter().enumerate().take(i + 1) {
            acc += f64::from(c) * f64::from(x[i - k]);
        }
        *out = acc as f32;
    }
    y
}

fn fir(b: &mut Bench) {
    b.guarded("signal_fir", |b| {
        let ctx = b.ctx.clone();
        let cases: &[(usize, usize, usize)] = &[
            (1_000_000, 1, 8),
            (1_000_000, 1, 64),
            (1_000_000, 1, 128),
            (1_000_000, 1, 256),
            (1_000_000, 1, 512),
            (1_000_000, 1, 1024),
            (1_000_000, 1, 8192),
            (16_000_000, 1, 8),
            (16_000_000, 1, 64),
            (16_000_000, 1, 128),
            (16_000_000, 1, 256),
            (16_000_000, 1, 1024),
            (4096, 1024, 32),
        ];
        for &(n, batch, k) in cases {
            let x = uniform(n * batch, 120, -1.0, 1.0);
            let taps = uniform(k, 121, -1.0, 1.0);
            let (xl, bl) = (lumax::arr(&ctx, &x, [n, batch]), lumax::arr(&ctx, &taps, k));
            let (xa, ba) = (afx::arr(&x, &dims2(n, batch)), afx::arr(&taps, &[k as u64]));
            let lo = lumax::host(&xl.fir(&bl));
            let ao = afx::host(&af::fir(&ba, &xa));
            let sc = (k as f64).sqrt();
            let mut v = compare(&lo, &ao, 1e-4, 1e-4 * sc);
            if batch == 1 && n <= 1_000_000 && k <= 1024 {
                let want: Vec<f64> = fir_ref(&x, &taps).iter().map(|&q| f64::from(q)).collect();
                v = v.and(compare_ref("luma", &lo, &want, 1e-4, 1e-4 * sc));
            }
            let size = if batch == 1 { format!("{} samples, {k} taps", human(n)) } else { format!("{batch} x {n} samples, {k} taps") };
            b.run(
                "fir",
                &size,
                v,
                "",
                &mut || {
                    xl.fir(&bl).eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    af::fir(&ba, &xa).eval();
                    afx::sync();
                },
            );
        }
    });
}

fn iir_ref(x: &[f32], bc: &[f32], ac: &[f32]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for (k, &c) in bc.iter().enumerate().take(i + 1) {
            acc += f64::from(c) * f64::from(x[i - k]);
        }
        for (k, &c) in ac.iter().enumerate().skip(1).take(i) {
            acc -= f64::from(c) * y[i - k];
        }
        y[i] = acc / f64::from(ac[0]);
    }
    y
}

fn iir(b: &mut Bench) {
    b.guarded("signal_iir", |b| {
        let ctx = b.ctx.clone();
        let biquad_b = [0.2f32, 0.4, 0.2];
        let biquad_a = [1.0f32, -0.6, 0.2];
        let order4_b = [0.1f32, 0.2, 0.3, 0.2, 0.1];
        let order4_a = [1.0f32, -1.2, 0.9, -0.3, 0.1];
        let cases: &[(usize, usize, &[f32], &[f32], &str)] = &[
            (1_000_000, 1, &biquad_b, &biquad_a, "biquad"),
            (4_000_000, 1, &biquad_b, &biquad_a, "biquad"),
            (16_000_000, 1, &biquad_b, &biquad_a, "biquad"),
            (1_000_000, 1, &order4_b, &order4_a, "order 4"),
            (4096, 1024, &biquad_b, &biquad_a, "biquad"),
            (65536, 64, &biquad_b, &biquad_a, "biquad"),
        ];
        for &(n, batch, bc, ac, what) in cases {
            let x = uniform(n * batch, 130, -1.0, 1.0);
            let xl = lumax::arr(&ctx, &x, [n, batch]);
            let xa = afx::arr(&x, &dims2(n, batch));
            let (ba, aa) = (afx::arr(bc, &[bc.len() as u64]), afx::arr(ac, &[ac.len() as u64]));
            let lo = lumax::host(&xl.iir(bc, ac));
            let ao = attempt(|| afx::host(&af::iir(&ba, &aa, &xa)));
            let mut v = Verdict::yes("");
            if batch == 1 {
                let want = iir_ref(&x, bc, ac);
                v = compare_ref("luma", &lo, &want, 1e-3, 1e-4);
                if let Some(ao) = &ao {
                    v = v.and(compare_ref("AF", ao, &want, 1e-3, 1e-4));
                }
            } else if let Some(ao) = &ao {
                let want: Vec<f64> = (0..batch.min(4)).flat_map(|j| iir_ref(&x[j * n..(j + 1) * n], bc, ac)).collect();
                let m = want.len();
                v = compare_ref("luma (first signals)", &lo[..m], &want, 1e-3, 1e-4).and(compare(&lo, ao, 1e-3, 1e-4));
            }
            let size = if batch == 1 { format!("{} samples, {what}", human(n)) } else { format!("{batch} x {n} samples, {what}") };
            match ao {
                Some(_) => b.run(
                    "iir",
                    &size,
                    v,
                    "",
                    &mut || {
                        xl.iir(bc, ac).eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        af::iir(&ba, &aa, &xa).eval();
                        afx::sync();
                    },
                ),
                None => b.run_one(
                    "iir",
                    &size,
                    Some(&mut || {
                        xl.iir(bc, ac).eval().unwrap();
                        ctx.sync().unwrap();
                    }),
                    None,
                    "ArrayFire iir failed",
                ),
            }
        }
    });
}
