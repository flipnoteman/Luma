//! Interpolation, image warps and digital filters: `approx1`/`approx2`, `resize`, `rotate`,
//! `transform`, `fir` and `iir`. Each case is checked against an `f64` host reference at a
//! sample of outputs before it is timed.

use luma::*;

use super::{check_sampled, host, pick, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    approx(h);
    warp(h);
    filters(h);
}

fn approx(h: &mut Harness) {
    h.guarded("approx", |h| {
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 16, 1 << 20) } else { pick(h, 1 << 20, 1 << 24) };
        let x = values(n, 301, -1.0, 1.0);
        let jitter = values(n, 302, 0.0, 0.5);
        let pos: Vec<f32> = (0..n).map(|i| i as f32 * (1.0 - 1.0 / n as f32) + jitter[i]).collect();
        let (xa, pa) = (upload(h, &x, n), upload(h, &pos, n));
        for (name, method) in [("nearest", Interp::Nearest), ("linear", Interp::Linear), ("cubic", Interp::Cubic)] {
            let got = host(&xa.approx1(&pa, method, 0.0));
            let at = |i: usize| -> f64 {
                let p = f64::from(pos[i]);
                let (f, t) = (p.floor() as i64, p - p.floor());
                let s = |j: i64| f64::from(x[j.clamp(0, n as i64 - 1) as usize]);
                match method {
                    Interp::Nearest => s((p + 0.5).floor() as i64),
                    Interp::Linear => (1.0 - t) * s(f) + t * s(f + 1),
                    _ => {
                        let (y0, y1, y2, y3) = (s(f - 1), s(f), s(f + 1), s(f + 2));
                        let a0 = y3 - y2 - y0 + y1;
                        a0 * t * t * t + (y0 - y1 - a0) * t * t + (y2 - y0) * t + y1
                    }
                }
            };
            check_sampled(&format!("approx/approx1_{name}"), &got, |i| if pos[i] <= (n - 1) as f32 { at(i) } else { 0.0 }, 997, 1e-4, 1e-5);
            h.case(&format!("approx/approx1_{name}"), &format!("{n} samples"), Work::Items(n as f64, "Msamples/s"), &mut || xa.approx1(&pa, method, 0.0).eval().unwrap());
        }
        let (w, hh) = if h.ctx.is_cpu() { pick(h, (256, 256), (1024, 1024)) } else { pick(h, (1024, 1024), (3840, 2160)) };
        let img = values(w * hh, 303, 0.0, 1.0);
        let (mut p0, mut p1) = (Vec::with_capacity(w * hh), Vec::with_capacity(w * hh));
        for y in 0..hh {
            for x in 0..w {
                p0.push(x as f32 + 3.0 * (y as f32 * 0.02).sin());
                p1.push(y as f32 + 3.0 * (x as f32 * 0.02).cos());
            }
        }
        let (ia, p0a, p1a) = (upload(h, &img, [w, hh]), upload(h, &p0, [w, hh]), upload(h, &p1, [w, hh]));
        let got = host(&ia.approx2(&p0a, &p1a, Interp::Linear, 0.0));
        let at = |i: usize| -> f64 {
            let (x, y) = (f64::from(p0[i]), f64::from(p1[i]));
            if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (hh - 1) as f64 {
                return 0.0;
            }
            let (fx, fy) = (x.floor() as usize, y.floor() as usize);
            let (tx, ty) = (x - x.floor(), y - y.floor());
            let g = |i: usize, j: usize| f64::from(img[i.min(w - 1) + w * j.min(hh - 1)]);
            (1.0 - ty) * ((1.0 - tx) * g(fx, fy) + tx * g(fx + 1, fy)) + ty * ((1.0 - tx) * g(fx, fy + 1) + tx * g(fx + 1, fy + 1))
        };
        check_sampled("approx/approx2_bilinear", &got, at, 1009, 1e-4, 1e-5);
        h.case("approx/approx2_bilinear", &format!("{w}x{hh}"), Work::Items((w * hh) as f64, "Mpix/s"), &mut || ia.approx2(&p0a, &p1a, Interp::Linear, 0.0).eval().unwrap());
        h.case("approx/approx2_bicubic", &format!("{w}x{hh}"), Work::Items((w * hh) as f64, "Mpix/s"), &mut || ia.approx2(&p0a, &p1a, Interp::Cubic, 0.0).eval().unwrap());
    });
}

/// The spacing of the `f32` values around `x` (`x > 0`).
fn ulp(x: f64) -> f64 {
    2f64.powi(x.log2().floor() as i32 - 23)
}

fn warp(h: &mut Harness) {
    h.guarded("warp", |h| {
        let (w, hh) = if h.ctx.is_cpu() { pick(h, (256, 256), (1024, 1024)) } else { pick(h, (1024, 1024), (3840, 2160)) };
        let img = values(w * hh, 304, 0.0, 1.0);
        let ia = upload(h, &img, [w, hh]);
        let pix = |n: usize| Work::Items(n as f64, "Mpix/s");
        // resize: pixel (x, y) of the output samples x / (o / i).
        let (ow, oh) = (w / 2 * 3 / 2 + 1, hh / 3 * 2 + 3);
        let got = host(&ia.resize(ow, oh, Interp::Linear));
        let (sx, sy) = (ow as f32 / w as f32, oh as f32 / hh as f32);
        check_sampled(
            "warp/resize_bilinear",
            &got,
            |i| {
                let (px, py) = (f64::from((i % ow) as f32 / sx), f64::from((i / ow) as f32 / sy));
                let g = |a: f64, b: f64| f64::from(img[(a as usize).min(w - 1) + w * (b as usize).min(hh - 1)]);
                let (fx, fy, tx, ty) = (px.floor(), py.floor(), px - px.floor(), py - py.floor());
                (1.0 - ty) * ((1.0 - tx) * g(fx, fy) + tx * g(fx + 1.0, fy)) + ty * ((1.0 - tx) * g(fx, fy + 1.0) + tx * g(fx + 1.0, fy + 1.0))
            },
            1013,
            1e-4,
            // The kernels divide the output index by the scale (like ArrayFire) and CUDA's f32 division is
            // `div.approx.ftz` (2 ulp; wgpu on the same GPU gives the same bits), so a source coordinate is
            // up to 2 ulp off, which moves a value of [0, 1] by that much at most (neighbours differ by less than 1).
            2.0 * (ulp(w as f64) + ulp(hh as f64)),
        );
        for (name, method) in [("nearest", Interp::Nearest), ("bilinear", Interp::Linear), ("bicubic", Interp::Cubic)] {
            h.case(&format!("warp/resize_{name}"), &format!("{w}x{hh} -> {ow}x{oh}"), pix(ow * oh), &mut || ia.resize(ow, oh, method).eval().unwrap());
            h.case(&format!("warp/resize_down_{name}"), &format!("{w}x{hh} -> {}x{}", w / 2, hh / 2), pix(w / 2 * (hh / 2)), &mut || ia.resize(w / 2, hh / 2, method).eval().unwrap());
        }
        let m = [0.9f32, -0.3, 40.0, 0.25, 1.1, -20.0];
        let got = host(&ia.transform(m, w, hh, Interp::Linear, true));
        check_sampled(
            "warp/transform",
            &got,
            |i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let (u, v) = (f64::from(m[0]) * x + f64::from(m[1]) * y + f64::from(m[2]), f64::from(m[3]) * x + f64::from(m[4]) * y + f64::from(m[5]));
                if u < 0.0 || v < 0.0 || u >= w as f64 || v >= hh as f64 {
                    return 0.0;
                }
                let g = |a: f64, b: f64| f64::from(img[(a as usize).min(w - 1) + w * (b as usize).min(hh - 1)]);
                let (fx, fy, tx, ty) = (u.floor(), v.floor(), u - u.floor(), v - v.floor());
                (1.0 - ty) * ((1.0 - tx) * g(fx, fy) + tx * g(fx + 1.0, fy)) + ty * ((1.0 - tx) * g(fx, fy + 1.0) + tx * g(fx + 1.0, fy + 1.0))
            },
            1019,
            2e-4,
            // `m0 x + m1 y + m2` in `f32` is within 3 ulp of its value (a product, a fused product and a sum, each at
            // most one ulp of the largest coordinate), which moves a value of [0, 1] by that much at most.
            2e-5f64.max(3.0 * (ulp(f64::from(m[0].abs()) * w as f64 + f64::from(m[1].abs()) * hh as f64 + f64::from(m[2].abs())) + ulp(f64::from(m[3].abs()) * w as f64 + f64::from(m[4].abs()) * hh as f64 + f64::from(m[5].abs())))),
        );
        h.case("warp/transform_bilinear", &format!("{w}x{hh}"), pix(w * hh), &mut || ia.transform(m, w, hh, Interp::Linear, true).eval().unwrap());
        h.case("warp/rotate_bilinear", &format!("{w}x{hh}"), pix(w * hh), &mut || ia.rotate(0.3, true, Interp::Linear).eval().unwrap());
        h.case("warp/rotate_nearest", &format!("{w}x{hh}"), pix(w * hh), &mut || ia.rotate(0.3, true, Interp::Nearest).eval().unwrap());
        let rgb = values(w * hh * 3, 305, 0.0, 1.0);
        let ra = upload(h, &rgb, [w, hh, 3]);
        h.case("warp/resize_bilinear_rgb", &format!("{w}x{hh}x3 -> {ow}x{oh}x3"), pix(ow * oh * 3), &mut || ra.resize(ow, oh, Interp::Linear).eval().unwrap());
    });
}

fn filters(h: &mut Harness) {
    h.guarded("filters", |h| {
        let n: usize = if h.ctx.is_cpu() { pick(h, 1 << 16, 1 << 20) } else { pick(h, 1 << 20, 1 << 24) };
        let x = values(n, 306, -1.0, 1.0);
        let xa = upload(h, &x, n);
        for k in [8usize, 64, 128, 256, 512, 1024] {
            let taps = values(k, 307 + k as u64, -1.0, 1.0);
            let ta = upload(h, &taps, k);
            let got = host(&xa.fir(&ta));
            let scale = (k as f64).sqrt();
            check_sampled("filters/fir", &got, |i| (0..k.min(i + 1)).map(|j| f64::from(taps[j]) * f64::from(x[i - j])).sum(), 991, 1e-4, 1e-4 * scale);
            h.case("filters/fir", &format!("{n} samples, {k} taps"), Work::Items(n as f64, "Msamples/s"), &mut || xa.fir(&ta).eval().unwrap());
            // The FFT convolution of the same filter (the full convolution, `n + k - 1` outputs): where `fir` should switch to it.
            h.case("filters/fir_fft", &format!("{n} samples, {k} taps"), Work::Items(n as f64, "Msamples/s"), &mut || xa.conv1d_fft(&ta).eval().unwrap());
        }
        let (b, a) = ([0.2f32, 0.4, 0.2], [1.0f32, -0.6, 0.2]);
        let got = host(&xa.iir(&b, &a));
        let mut want = vec![0.0f64; n];
        for i in 0..n {
            let mut acc: f64 = (0..3.min(i + 1)).map(|k| f64::from(b[k]) * f64::from(x[i - k])).sum();
            for k in 1..3.min(i + 1) {
                acc -= f64::from(a[k]) * want[i - k];
            }
            want[i] = acc;
        }
        check_sampled("filters/iir_biquad", &got, |i| want[i], 983, 1e-3, 1e-4);
        h.case("filters/iir_biquad", &format!("{n} samples"), Work::Items(n as f64, "Msamples/s"), &mut || xa.iir(&b, &a).eval().unwrap());
        // 1024 signals at once.
        let (len, batch) = (4096usize, 1024usize);
        let xb = upload(h, &values(len * batch, 308, -1.0, 1.0), [len, batch]);
        h.case("filters/iir_biquad_batched", &format!("{batch} x {len}"), Work::Items((len * batch) as f64, "Msamples/s"), &mut || xb.iir(&b, &a).eval().unwrap());
    });
}
