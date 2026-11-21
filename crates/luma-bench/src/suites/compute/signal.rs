//! Signal and image processing: FFT convolution against direct convolution, and image filters
//! built from shifts, slices, scans and gathers.

use luma::*;

use super::{check64, host, pick, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    fft(h);
    image(h);
}

// -------------------------------------------------------------------------------------------------
// Convolution
// -------------------------------------------------------------------------------------------------

fn fft(h: &mut Harness) {
    h.guarded("fft", |h| {
        let cpu = h.ctx.is_cpu();
        // (signal length, kernel length): power-of-two and awkward lengths, short to very long.
        let mut cases: Vec<(usize, usize)> = vec![(4096, 64), (1 << 16, 1024), (100_003, 777), (1 << 20, 4096)];
        if !h.quick {
            cases.push((3 << 19, 2000));
            cases.push((1 << 22, 16384));
            if !cpu {
                cases.push((1 << 24, 1024));
                cases.push((1 << 26, 4096));
            }
        }
        // Spatial (direct) convolutions at sizes where the FFT competes.
        let direct: Vec<(usize, usize)> = if h.quick { vec![(1 << 16, 64), (1 << 20, 64)] } else { vec![(1 << 16, 64), (1 << 20, 64), (1 << 20, 1024)] };

        for (fft_case, list) in [(true, cases), (false, direct)] {
            for (n, k) in list {
                let x = values(n, 111, -1.0, 1.0);
                let kernel: Vec<f32> = values(k, 112, -1.0, 1.0).iter().map(|v| v / (k as f32).sqrt()).collect();
                let (xa, ka) = (upload(h, &x, n), upload(h, &kernel, k));
                let conv = |xa: &Array<f32>, ka: &Array<f32>| if fft_case { xa.conv1d_fft(ka) } else { xa.conv1d(ka, 1, 0) };
                let got = host(&conv(&xa, &ka));
                let len = if fft_case { n + k - 1 } else { n - k + 1 };
                assert_eq!(got.len(), len, "fft: output length");
                // Direct sums at a sample of outputs, edges included.
                let mut at: Vec<usize> = ints(48, 113, len as u32).into_iter().map(|j| j as usize).collect();
                at.extend([0, 1, len / 2, len - 2, len - 1]);
                let want: Vec<f64> = at
                    .iter()
                    .map(|&j| {
                        if fft_case {
                            (j.saturating_sub(k - 1)..=j.min(n - 1)).map(|i| f64::from(x[i]) * f64::from(kernel[j - i])).sum()
                        } else {
                            (0..k).map(|i| f64::from(x[j + i]) * f64::from(kernel[i])).sum()
                        }
                    })
                    .collect();
                let sampled: Vec<f32> = at.iter().map(|&j| got[j]).collect();
                check64(if fft_case { "fft/conv1d_fft" } else { "fft/conv1d_direct" }, &sampled, &want, 1e-3, 2e-3);
                let name = if fft_case { "fft/conv1d_fft" } else { "fft/conv1d_direct" };
                h.case(name, &format!("{n} * {k}"), Work::Items(n as f64, "Msamples/s"), &mut || conv(&xa, &ka).eval().unwrap());
            }
        }
    });
}

// -------------------------------------------------------------------------------------------------
// Images
// -------------------------------------------------------------------------------------------------

/// A smooth gradient with hard-edged tiles and mild noise, in `[0, 1)`; `n x n`, x fastest.
fn test_image(n: usize) -> Vec<f32> {
    values(n * n, 1, -0.03, 0.03)
        .iter()
        .enumerate()
        .map(|(i, noise)| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            let tile = if ((i % n) / 64 + (i / n) / 64) % 2 == 0 { 0.15 } else { 0.0 };
            (0.5 + 0.3 * (x * 0.01).sin() * (y * 0.013).cos() + tile + noise).clamp(0.0, 0.999)
        })
        .collect()
}

/// Normalised Gaussian taps (sigma = radius / 2) for `2 * radius + 1` samples.
fn gaussian_taps(radius: usize) -> Vec<f32> {
    let sigma = radius as f64 / 2.0;
    let w: Vec<f64> = (0..=2 * radius).map(|k| (-((k as f64 - radius as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.iter().map(|v| (v / sum) as f32).collect()
}

/// Separable blur with edge clamping: horizontal taps, then vertical taps.
fn blur(img: &Array<f32>, taps: &[f32]) -> Array<f32> {
    let r = (taps.len() / 2) as isize;
    let pass = |a: &Array<f32>, axis: usize| {
        let mut acc: Option<Array<f32>> = None;
        for (t, &w) in taps.iter().enumerate() {
            let d = t as isize - r;
            let offsets = if axis == 0 { [-d, 0] } else { [0, -d] };
            let term = a.shift_clamp(offsets) * w;
            acc = Some(match acc {
                Some(sum) => sum + term,
                None => term,
            });
        }
        acc.expect("at least one tap")
    };
    pass(&pass(img, 0), 1)
}

fn clamp_at(img: &[f32], n: usize, x: isize, y: isize) -> f64 {
    let (x, y) = (x.clamp(0, n as isize - 1) as usize, y.clamp(0, n as isize - 1) as usize);
    f64::from(img[x + n * y])
}

fn blur_host(img: &[f32], n: usize, taps: &[f32]) -> Vec<f64> {
    let r = (taps.len() / 2) as isize;
    let mut rows = vec![0.0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            rows[x + n * y] = taps.iter().enumerate().map(|(t, &w)| f64::from(w) * clamp_at(img, n, x as isize + t as isize - r, y as isize)).sum();
        }
    }
    let mut out = vec![0.0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            out[x + n * y] = taps
                .iter()
                .enumerate()
                .map(|(t, &w)| {
                    let yy = (y as isize + t as isize - r).clamp(0, n as isize - 1) as usize;
                    f64::from(w) * rows[x + n * yy]
                })
                .sum();
        }
    }
    out
}

/// Bilinear resampling of `src` (`iw x ih`) to `ow x oh` with pixel-centre coordinates, four
/// gathers per output pixel.
fn resize(ctx: &Context, src: &Array<f32>, (iw, ih): (usize, usize), (ow, oh): (usize, usize)) -> Array<f32> {
    let coord = |axis: usize, i: usize, o: usize| {
        let c = Array::<f32>::range_on(ctx, [ow, oh], axis).unwrap();
        ((c + 0.5) * (i as f32 / o as f32) - 0.5).clamp(0.0, (i - 1) as f32)
    };
    let (fx, fy) = (coord(0, iw, ow), coord(1, ih, oh));
    let (x0f, y0f) = (fx.floor(), fy.floor());
    let (tx, ty) = (&fx - &x0f, &fy - &y0f);
    let (x0, y0) = (x0f.to_u32(), y0f.to_u32());
    let x1 = (&x0 + 1u32).minimum((iw - 1) as u32);
    let y1 = (&y0 + 1u32).minimum((ih - 1) as u32);
    let at = |x: &Array<u32>, y: &Array<u32>| src.take(&(y * (iw as u32) + x));
    let top = at(&x0, &y0) * (1.0 - &tx) + at(&x1, &y0) * &tx;
    let bottom = at(&x0, &y1) * (1.0 - &tx) + at(&x1, &y1) * &tx;
    top * (1.0 - &ty) + bottom * &ty
}

fn resize_host(src: &[f32], (iw, ih): (usize, usize), (ow, oh): (usize, usize), x: usize, y: usize) -> f64 {
    let coord = |o: usize, i: usize, out: usize| ((o as f64 + 0.5) * (i as f64 / out as f64) - 0.5).clamp(0.0, (i - 1) as f64);
    let (fx, fy) = (coord(x, iw, ow), coord(y, ih, oh));
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(iw - 1), (y0 + 1).min(ih - 1));
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let p = |x: usize, y: usize| f64::from(src[x + iw * y]);
    (p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx) * (1.0 - ty) + (p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx) * ty
}

fn image(h: &mut Harness) {
    h.guarded("image", |h| {
        let ctx = h.ctx.clone();
        let n: usize = if h.ctx.is_cpu() { pick(h, 512, 1024) } else { pick(h, 1024, 4096) };
        let pix = |n: usize| Work::Items((n * n) as f64, "Mpix/s");

        // Gaussian blur: a latency-bound small image, then the large one at two radii.
        for (radius, side) in [(4usize, 256usize), (4, n), (8, n)] {
            let data = test_image(side);
            let img = upload(h, &data, [side, side]);
            let taps = gaussian_taps(radius);
            check64("image/gauss", &host(&blur(&img, &taps)), &blur_host(&data, side, &taps), 1e-4, 1e-5);
            h.case(&format!("image/gauss_r{radius}"), &format!("{side}x{side}"), pix(side), &mut || blur(&img, &taps).eval().unwrap());
        }

        let data = test_image(n);
        let img = upload(h, &data, [n, n]);

        // Sobel gradient magnitude: eight shifted reads fused with the magnitude.
        let sobel = |img: &Array<f32>| {
            let s = |dx: isize, dy: isize| img.shift_clamp([-dx, -dy]);
            let gx = (s(1, -1) + s(1, 0) * 2.0 + s(1, 1)) - (s(-1, -1) + s(-1, 0) * 2.0 + s(-1, 1));
            let gy = (s(-1, 1) + s(0, 1) * 2.0 + s(1, 1)) - (s(-1, -1) + s(0, -1) * 2.0 + s(1, -1));
            (&gx * &gx + &gy * &gy).sqrt()
        };
        let want: Vec<f64> = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as isize, (i / n) as isize);
                let p = |dx: isize, dy: isize| clamp_at(&data, n, x + dx, y + dy);
                let gx = (p(1, -1) + 2.0 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2.0 * p(-1, 0) + p(-1, 1));
                let gy = (p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2.0 * p(0, -1) + p(1, -1));
                (gx * gx + gy * gy).sqrt()
            })
            .collect();
        check64("image/sobel", &host(&sobel(&img)), &want, 1e-4, 1e-5);
        h.case("image/sobel", &format!("{n}x{n}"), pix(n), &mut || sobel(&img).eval().unwrap());

        // Box filter (valid region) as a sum of slices of the image, separably.
        let box_slices = |img: &Array<f32>, r: usize| {
            let w = 2 * r + 1;
            let o = n - 2 * r;
            let mut rows = img.slice(0, 0..o);
            for d in 1..w {
                rows = rows + img.slice(0, d..d + o);
            }
            let mut out = rows.slice(1, 0..o);
            for d in 1..w {
                out = out + rows.slice(1, d..d + o);
            }
            out * (1.0 / (w * w) as f32)
        };
        let box_host = |data: &[f32], r: usize| -> Vec<f64> {
            let (w, o) = (2 * r + 1, n - 2 * r);
            let mut cols = vec![0.0f64; o * n];
            for y in 0..n {
                for x in 0..o {
                    cols[x + o * y] = (0..w).map(|d| f64::from(data[x + d + n * y])).sum();
                }
            }
            (0..o * o).map(|i| (0..w).map(|d| cols[i % o + o * (i / o + d)]).sum::<f64>() / (w * w) as f64).collect()
        };
        for r in [4usize, 16] {
            check64("image/box_slices", &host(&box_slices(&img, r)), &box_host(&data, r), 1e-4, 1e-5);
            h.case(&format!("image/box_slices_r{r}"), &format!("{n}x{n}"), pix(n), &mut || box_slices(&img, r).eval().unwrap());
        }

        // Box filter from an integral image (two scans), cost independent of the radius. Zero-mean
        // noise keeps the f32 running sums small.
        let noise = values(n * n, 121, -0.5, 0.5);
        let zero_mean = upload(h, &noise, [n, n]);
        let integral_box = |a: &Array<f32>, r: usize| {
            let w = 2 * r + 1;
            let cp = a.cumsum(0).cumsum(1).pad([(1, 0), (1, 0)], PadMode::Zero);
            let corner = |x0: usize, y0: usize| cp.slice(0, x0..x0 + n + 1 - w).slice(1, y0..y0 + n + 1 - w);
            (corner(w, w) - corner(0, w) - corner(w, 0) + corner(0, 0)) * (1.0 / (w * w) as f32)
        };
        check64("image/box_integral", &host(&integral_box(&zero_mean, 8)), &box_host(&noise, 8), 0.0, 2e-3);
        h.case("image/box_integral_r8", &format!("{n}x{n}"), pix(n), &mut || integral_box(&zero_mean, 8).eval().unwrap());

        // Histogram equalisation: histogram, running sum, lookup table, gather.
        let equalize = |a: &Array<f32>| {
            let cdf = a.histogram(256, 0.0, 1.0).cumsum(0).to_f32();
            let lut = cdf * (1.0 / (n * n) as f32);
            lut.take(&(a * 256.0).to_u32().minimum(255))
        };
        let mut counts = [0u32; 256];
        for &v in &data {
            counts[((v * 256.0) as usize).min(255)] += 1;
        }
        let mut cdf = [0.0f32; 256];
        let mut run = 0u32;
        for (c, &k) in cdf.iter_mut().zip(&counts) {
            run += k;
            *c = run as f32 * (1.0 / (n * n) as f32);
        }
        let want: Vec<f64> = data.iter().map(|&v| f64::from(cdf[((v * 256.0) as usize).min(255)])).collect();
        check64("image/hist_equalize", &host(&equalize(&img)), &want, 0.0, 1e-6);
        h.case("image/hist_equalize", &format!("{n}x{n}"), pix(n), &mut || equalize(&img).eval().unwrap());

        // Bilinear resampling by gather: upscale from half size and downscale to half size.
        let half = n / 2;
        let small_data = test_image(half);
        let small = upload(h, &small_data, [half, half]);
        let check_resize = |name: &str, out: &Array<f32>, src: &[f32], from: (usize, usize), to: (usize, usize)| {
            let got = host(out);
            let (mut g, mut w) = (Vec::new(), Vec::new());
            for y in (0..to.1).step_by(13) {
                for x in (0..to.0).step_by(13) {
                    g.push(got[x + to.0 * y]);
                    w.push(resize_host(src, from, to, x, y));
                }
            }
            check64(name, &g, &w, 1e-4, 1e-5);
        };
        check_resize("image/resize_up2x", &resize(&ctx, &small, (half, half), (n, n)), &small_data, (half, half), (n, n));
        h.case("image/resize_up2x", &format!("{half}x{half} -> {n}x{n}"), pix(n), &mut || resize(&ctx, &small, (half, half), (n, n)).eval().unwrap());
        check_resize("image/resize_down2x", &resize(&ctx, &img, (n, n), (half, half)), &data, (n, n), (half, half));
        h.case("image/resize_down2x", &format!("{n}x{n} -> {half}x{half}"), pix(half), &mut || resize(&ctx, &img, (n, n), (half, half)).eval().unwrap());
    });
}
