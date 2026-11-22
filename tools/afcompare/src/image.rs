//! Image and filter operations: luma's `conv2` / `conv3` / rank and morphological filters /
//! `sobel` / `grad` / `bilateral` / `sat` / `hist_equal` / colour conversions / `unwrap` / `wrap`
//! against the ArrayFire functions of the same name, at 1080p, 4K and (3-D) a 256^3 volume.
//!
//! luma's convolutions are cross-correlations, ArrayFire's `convolve*` flip the kernel: ArrayFire is
//! given the flipped kernel, so both compute the same thing. ArrayFire's `dilate` / `erode` read the
//! mask transposed and its `sobel` has its own border handling, so those cases use symmetric masks
//! and compare the interior only.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::{attempt, L};
use crate::lumax;
use crate::util::*;

/// `v` of an `[h, w, ...]` array without the `margin` outermost cells of axes 0 and 1.
fn interior(v: &[f32], h: usize, w: usize, margin: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let planes = v.len() / (h * w);
    for p in 0..planes {
        for j in margin..w - margin {
            for i in margin..h - margin {
                out.push(v[p * h * w + i + h * j]);
            }
        }
    }
    out
}

/// A kernel flipped along every axis (column-major `[k0, k1, k2]`).
fn flipped(k: &[f32]) -> Vec<f32> {
    k.iter().rev().copied().collect()
}

fn lr<'a>(ctx: &'a luma::Context, f: impl Fn() -> L + 'a) -> impl FnMut() + 'a {
    move || {
        f().eval().unwrap();
        ctx.sync().unwrap();
    }
}

fn ar<'a, T: af::HasAfEnum>(f: impl Fn() -> af::Array<T> + 'a) -> impl FnMut() + 'a {
    move || {
        f().eval();
        afx::sync();
    }
}

pub fn images(b: &mut Bench) {
    b.guarded("image", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let d = [h as u64, w as u64];
            let data = uniform(h * w, 301, 0.0, 1.0);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &d));

            // Convolutions, direct.
            for k in [3usize, 7, 15] {
                let kd = uniform(k * k, 302, -1.0, 1.0);
                let (kl, ka) = (lumax::arr(&ctx, &kd, [k, k]), afx::arr(&flipped(&kd), &[k as u64, k as u64]));
                let l = || xl.conv2(&kl, luma::ConvSpec::new());
                let a = || af::convolve2(&xa, &ka, af::ConvMode::DEFAULT, af::ConvDomain::SPATIAL);
                let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
                b.run(&format!("13 conv2 {k}x{k}"), &size, v, "luma cross-correlation = AF convolve2 (spatial) with the flipped kernel; zero border, same size", &mut lr(&ctx, l), &mut ar(a));
            }
            // Separable.
            for k in [5usize, 15] {
                let g = luma::gaussian_kernel_1d_on(&ctx, k, 0.0).unwrap();
                let gd = lumax::host(&g);
                let ga = afx::arr(&gd, &[k as u64]);
                let l = || xl.conv2_sep(&g, &g, luma::ConvSpec::new());
                let a = || af::convolve2_sep(&ga, &ga, &xa, af::ConvMode::DEFAULT);
                let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
                b.run(&format!("13 conv2_sep {k}"), &size, v, "symmetric Gaussian taps along both axes", &mut lr(&ctx, l), &mut ar(a));
            }
            // Rank filters.
            for k in [3usize, 5, 7] {
                let l = || xl.medfilt2([k, k], luma::Border::Zero);
                let a = || af::medfilt(&xa, k as u64, k as u64, af::BorderType::ZERO);
                let v = exact(&lumax::host(&l()), &afx::host(&a()));
                b.run(&format!("13 medfilt2 {k}x{k}"), &size, v, "zero border", &mut lr(&ctx, l), &mut ar(a));
            }
            for k in [3usize, 15] {
                let l = || xl.minfilt([k, k], luma::Border::Ignore);
                let a = || af::minfilt(&xa, k as u64, k as u64, af::BorderType::ZERO);
                if attempt(|| a().eval()).is_none() {
                    b.run_one(&format!("13 minfilt {k}x{k}"), &size, Some(&mut lr(&ctx, l)), None, "ArrayFire rejects this window");
                } else {
                    let v = exact(&lumax::host(&l()), &afx::host(&a()));
                    b.run(&format!("13 minfilt {k}x{k}"), &size, v, "cells outside the image ignored (AF's only mode)", &mut lr(&ctx, l), &mut ar(a));
                }
                let l = || xl.maxfilt([k, k], luma::Border::Ignore);
                let a = || af::maxfilt(&xa, k as u64, k as u64, af::BorderType::ZERO);
                if attempt(|| a().eval()).is_none() {
                    b.run_one(&format!("13 maxfilt {k}x{k}"), &size, Some(&mut lr(&ctx, l)), None, "ArrayFire rejects this window");
                } else {
                    let v = exact(&lumax::host(&l()), &afx::host(&a()));
                    b.run(&format!("13 maxfilt {k}x{k}"), &size, v, "cells outside the image ignored (AF's only mode)", &mut lr(&ctx, l), &mut ar(a));
                }
            }
            // Morphology with a symmetric cross.
            let cross = [0.0f32, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
            let (ml, ma) = (lumax::arr(&ctx, &cross, [3, 3]), afx::arr(&cross, &[3, 3]));
            let (l, a) = (|| xl.dilate(&ml), || af::dilate(&xa, &ma));
            let v = exact(&lumax::host(&l()), &afx::host(&a()));
            b.run("13 dilate 3x3 cross", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
            let (l, a) = (|| xl.erode(&ml), || af::erode(&xa, &ma));
            let v = exact(&lumax::host(&l()), &afx::host(&a()));
            b.run("13 erode 3x3 cross", &size, v, "", &mut lr(&ctx, l), &mut ar(a));

            // Derivatives.
            {
                let lpair = || {
                    let (p, q) = xl.sobel();
                    (p, q)
                };
                let (p, q) = lpair();
                let (ax, ay) = af::sobel(&xa, 3);
                let v = compare(&interior(&lumax::host(&p), h, w, 1), &interior(&afx::host(&ax), h, w, 1), 1e-4, 1e-4)
                    .and(compare(&interior(&lumax::host(&q), h, w, 1), &interior(&afx::host(&ay), h, w, 1), 1e-4, 1e-4));
                b.run(
                    "13 sobel (dx, dy)",
                    &size,
                    v,
                    "both derivatives; interior compared (the borders differ by convention)",
                    &mut || {
                        let (p, q) = lpair();
                        luma::eval_all(&[&p, &q]).unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let (p, q) = af::sobel(&xa, 3);
                        p.eval();
                        q.eval();
                        afx::sync();
                    },
                );
                let (p, q) = xl.grad();
                let (gx, gy) = af::gradient(&xa);
                let v = compare(&lumax::host(&p), &afx::host(&gx), 1e-4, 1e-4).and(compare(&lumax::host(&q), &afx::host(&gy), 1e-4, 1e-4));
                b.run(
                    "13 grad (dx, dy)",
                    &size,
                    v,
                    "",
                    &mut || {
                        let (p, q) = xl.grad();
                        luma::eval_all(&[&p, &q]).unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let (p, q) = af::gradient(&xa);
                        p.eval();
                        q.eval();
                        afx::sync();
                    },
                );
            }

            // Bilateral.
            let l = || xl.bilateral(2.0, 0.1);
            let a = || af::bilateral(&xa, 2.0f32, 0.1f32, false);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-3, 1e-4);
            b.run("13 bilateral", &size, v, "spatial sigma 2, range sigma 0.1", &mut lr(&ctx, l), &mut ar(a));

            // Summed-area table.
            let l = || xl.sat();
            let a = || af::sat(&xa);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 2e-3, 1e-2);
            b.run("13 sat", &size, v, "f32 running sums; summation orders differ", &mut lr(&ctx, l), &mut ar(a));

            // Histogram equalisation of an 8-bit-valued image with a shared histogram.
            let q: Vec<f32> = data.iter().map(|v| (v * 255.99).floor()).collect();
            let (ql, qa) = (lumax::arr(&ctx, &q, [h, w]), afx::arr(&q, &d));
            let hist = ql.histogram(256, 0.0, 256.0);
            let hv: Vec<u32> = lumax::host(&hist);
            let ha = afx::arr(&hv, &[256]);
            let l = || ql.hist_equal(&hist);
            let a = || af::hist_equal(&qa, &ha);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-3);
            b.run("13 hist_equal", &size, v, "same histogram on both sides; lookup by the bin of every pixel", &mut lr(&ctx, l), &mut ar(a));

            // Moments.
            let l = || xl.moments();
            let a = || af::moments(&xa, af::MomentType::FIRST_ORDER);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-3, 1e-1);
            b.run(
                "13 moments (M00 M01 M10 M11)",
                &size,
                v,
                "both read the four sums back (ArrayFire's reductions end in a host read)",
                &mut || {
                    let _ = l().to_vec().unwrap();
                },
                &mut || {
                    let _ = afx::host(&a());
                },
            );

            // Colour.
            let rgb = uniform(h * w * 3, 303, 0.0, 1.0);
            let (rl, ra) = (lumax::arr(&ctx, &rgb, [h, w, 3]), afx::arr(&rgb, &[h as u64, w as u64, 3]));
            let l = || rl.rgb2gray();
            let a = || af::rgb2gray(&ra, 0.2126, 0.7152, 0.0722);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-5, 1e-5);
            b.run("14 rgb2gray", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
            let l = || rl.rgb2hsv();
            let a = || af::rgb2hsv(&ra);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
            b.run("14 rgb2hsv", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
            let l = || rl.hsv2rgb();
            let a = || af::hsv2rgb(&ra);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
            b.run("14 hsv2rgb", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
            let l = || rl.rgb2ycbcr(luma::YccStd::Bt601);
            let a = || af::rgb2ycbcr(&ra, af::YCCStd::YCC_601);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-3);
            b.run("14 rgb2ycbcr 601", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
            let l = || rl.ycbcr2rgb(luma::YccStd::Bt601);
            let a = || af::ycbcr2rgb(&ra, af::YCCStd::YCC_601);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-3);
            b.run("14 ycbcr2rgb 601", &size, v, "", &mut lr(&ctx, l), &mut ar(a));

            // im2col / col2im.
            let l = || xl.unwrap([3, 3], [1, 1], [1, 1], true);
            let a = || af::unwrap(&xa, 3, 3, 1, 1, 1, 1, true);
            let v = exact(&lumax::host(&l()), &afx::host(&a()));
            b.run("14 unwrap 3x3", &size, v, "column layout, pad 1", &mut lr(&ctx, l), &mut ar(a));
            let ul = l();
            ul.eval().unwrap();
            let ua = a();
            ua.eval();
            let l = || ul.wrap([h, w], [3, 3], [1, 1], [1, 1], true);
            let a = || af::wrap(&ua, h as i64, w as i64, 3, 3, 1, 1, 1, 1, true);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
            b.run("14 wrap 3x3", &size, v, "sum of overlapping windows (luma gathers, AF scatters)", &mut lr(&ctx, l), &mut ar(a));
        }

        // 3-D, on a volume.
        let n = if b.quick { 128 } else { 256 };
        let size = format!("{n}^3");
        let d = [n as u64; 3];
        let vol = uniform(n * n * n, 311, 0.0, 1.0);
        let (vl, va) = (lumax::arr(&ctx, &vol, [n, n, n]), afx::arr(&vol, &d));
        for k in [3usize, 5] {
            let kd = uniform(k * k * k, 312, -1.0, 1.0);
            let (kl, ka) = (lumax::arr(&ctx, &kd, [k, k, k]), afx::arr(&flipped(&kd), &[k as u64; 3]));
            let l = || vl.conv3(&kl, luma::ConvSpec::new());
            let a = || af::convolve3(&va, &ka, af::ConvMode::DEFAULT, af::ConvDomain::SPATIAL);
            let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 1e-4);
            b.run(&format!("13 conv3 {k}x{k}x{k}"), &size, v, "flipped kernel for AF; zero border", &mut lr(&ctx, l), &mut ar(a));
        }
        let mask: Vec<f32> = (0..27).map(|i| if [4, 10, 12, 13, 14, 16, 22].contains(&i) { 1.0 } else { 0.0 }).collect();
        let (ml, ma) = (lumax::arr(&ctx, &mask, [3, 3, 3]), afx::arr(&mask, &[3, 3, 3]));
        let (l, a) = (|| vl.dilate3(&ml), || af::dilate3(&va, &ma));
        let v = exact(&lumax::host(&l()), &afx::host(&a()));
        b.run("13 dilate3 3x3x3 (6-neighbourhood)", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
        let (l, a) = (|| vl.erode3(&ml), || af::erode3(&va, &ma));
        let v = exact(&lumax::host(&l()), &afx::host(&a()));
        b.run("13 erode3 3x3x3 (6-neighbourhood)", &size, v, "", &mut lr(&ctx, l), &mut ar(a));
    });
}
