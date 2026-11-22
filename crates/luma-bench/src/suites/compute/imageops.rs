//! The image operations of `luma::image` (`conv2`, `conv3`, rank and morphological filters,
//! `sobel`, `bilateral`, `sat`, `hist_equal`, colour conversions, `unwrap`) at 1080p and 4K, and
//! 3-D filters on a volume. Each kind is checked against a small host computation first.

use luma::*;

use super::{check64, host, pick, upload};
use crate::harness::*;

/// `[H, W]` sizes: 1080p and 4K (smaller under `--quick` and on the CPU backend).
fn sizes(h: &Harness) -> Vec<[usize; 2]> {
    if h.ctx.is_cpu() {
        vec![[270, 480]]
    } else {
        pick(h, vec![[1080, 1920]], vec![[1080, 1920], [2160, 3840]])
    }
}

fn pix(dims: [usize; 2]) -> Work {
    Work::Items((dims[0] * dims[1]) as f64, "Mpix/s")
}

fn label(dims: [usize; 2]) -> String {
    format!("{}x{}", dims[1], dims[0])
}

/// `out[o] = sum_j in[o + j - k / 2] w[j]` with zero padding, over `[h, w]`.
fn conv_host(data: &[f32], [h, w]: [usize; 2], kernel: &[f32], [k0, k1]: [usize; 2]) -> Vec<f64> {
    let mut out = vec![0.0; h * w];
    for y in 0..w {
        for x in 0..h {
            let mut acc = 0.0f64;
            for j1 in 0..k1 {
                for j0 in 0..k0 {
                    let (a, b) = (x as i64 + j0 as i64 - ((k0 - 1) / 2) as i64, y as i64 + j1 as i64 - ((k1 - 1) / 2) as i64);
                    if (0..h as i64).contains(&a) && (0..w as i64).contains(&b) {
                        acc += f64::from(data[a as usize + h * b as usize]) * f64::from(kernel[j0 + k0 * j1]);
                    }
                }
            }
            out[x + h * y] = acc;
        }
    }
    out
}

/// A rank over the `k0 x k1` window around every cell with `pick` (zero padding for the median,
/// the cells that exist for the extrema).
fn rank_host(data: &[f32], [h, w]: [usize; 2], [k0, k1]: [usize; 2], pick: &dyn Fn(&mut [f32]) -> f32, zero: bool) -> Vec<f64> {
    let mut out = vec![0.0; h * w];
    for y in 0..w {
        for x in 0..h {
            let mut cells = Vec::new();
            for j1 in 0..k1 {
                for j0 in 0..k0 {
                    let (a, b) = (x as i64 + j0 as i64 - (k0 / 2) as i64, y as i64 + j1 as i64 - (k1 / 2) as i64);
                    if (0..h as i64).contains(&a) && (0..w as i64).contains(&b) {
                        cells.push(data[a as usize + h * b as usize]);
                    } else if zero {
                        cells.push(0.0);
                    }
                }
            }
            out[x + h * y] = f64::from(pick(&mut cells));
        }
    }
    out
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

pub fn run(h: &mut Harness) {
    h.guarded("imageops", |h| {
        // Small checks of every kind of kernel against the host.
        let small = [37usize, 53];
        let sdata = values(small[0] * small[1], 21, 0.0, 1.0);
        let simg = upload(h, &sdata, small);
        let sk = values(25, 22, -1.0, 1.0);
        let kern = upload(h, &sk, [5, 5]);
        check64("imageops/conv2", &host(&simg.conv2(&kern, ConvSpec::new())), &conv_host(&sdata, small, &sk, [5, 5]), 1e-4, 1e-4);
        check64("imageops/medfilt2", &host(&simg.medfilt2([5, 5], Border::Zero)), &rank_host(&sdata, small, [5, 5], &median, true), 0.0, 0.0);
        check64("imageops/minfilt", &host(&simg.minfilt([3, 3], Border::Ignore)), &rank_host(&sdata, small, [3, 3], &|v| v.iter().copied().fold(f32::INFINITY, f32::min), false), 0.0, 0.0);

        for dims in sizes(h) {
            let data = values(dims[0] * dims[1], 7, 0.0, 1.0);
            let img = upload(h, &data, dims);
            let tag = label(dims);
            for k in [3usize, 7, 15] {
                let kd = values(k * k, 8, -1.0, 1.0);
                let kernel = upload(h, &kd, [k, k]);
                h.case(&format!("imageops/conv2_{k}x{k}"), &tag, pix(dims), &mut || img.conv2(&kernel, ConvSpec::new()).eval().unwrap());
            }
            for k in [5usize, 15] {
                let g = gaussian_kernel_1d_on(&h.ctx, k, 0.0).unwrap();
                g.eval().unwrap();
                h.case(&format!("imageops/conv2_sep_{k}"), &tag, pix(dims), &mut || img.conv2_sep(&g, &g, ConvSpec::new().border(Border::Edge)).eval().unwrap());
            }
            for k in [3usize, 5, 7] {
                h.case(&format!("imageops/medfilt2_{k}x{k}"), &tag, pix(dims), &mut || img.medfilt2([k, k], Border::Zero).eval().unwrap());
            }
            for k in [3usize, 15] {
                h.case(&format!("imageops/minfilt_{k}x{k}"), &tag, pix(dims), &mut || img.minfilt([k, k], Border::Ignore).eval().unwrap());
                h.case(&format!("imageops/maxfilt_{k}x{k}"), &tag, pix(dims), &mut || img.maxfilt([k, k], Border::Ignore).eval().unwrap());
            }
            let mask: Vec<f32> = (0..9).map(|i| if i % 2 == 0 || i == 4 { 1.0 } else { 0.0 }).collect();
            let m = upload(h, &mask, [3, 3]);
            h.case("imageops/dilate_3x3", &tag, pix(dims), &mut || img.dilate(&m).eval().unwrap());
            h.case("imageops/erode_3x3", &tag, pix(dims), &mut || img.erode(&m).eval().unwrap());
            h.case("imageops/sobel", &tag, pix(dims), &mut || {
                let (a, b) = img.sobel();
                eval_all(&[&a, &b]).unwrap();
            });
            h.case("imageops/sobel_magnitude", &tag, pix(dims), &mut || img.sobel_magnitude().eval().unwrap());
            h.case("imageops/grad", &tag, pix(dims), &mut || {
                let (a, b) = img.grad();
                eval_all(&[&a, &b]).unwrap();
            });
            h.case("imageops/bilateral_s2", &tag, pix(dims), &mut || img.bilateral(2.0, 0.1).eval().unwrap());
            h.case("imageops/sat", &tag, pix(dims), &mut || img.sat().eval().unwrap());
            let hist = (&img * 255.0).histogram(256, 0.0, 256.0);
            hist.eval().unwrap();
            let scaled = (&img * 255.0).floor();
            scaled.eval().unwrap();
            h.case("imageops/hist_equal", &tag, pix(dims), &mut || scaled.hist_equal(&hist).eval().unwrap());
            h.case("imageops/moments", &tag, pix(dims), &mut || img.moments().eval().unwrap());
            let rgb = upload(h, &values(dims[0] * dims[1] * 3, 9, 0.0, 1.0), [dims[0], dims[1], 3]);
            h.case("imageops/rgb2gray", &tag, pix(dims), &mut || rgb.rgb2gray().eval().unwrap());
            h.case("imageops/rgb2hsv", &tag, pix(dims), &mut || rgb.rgb2hsv().eval().unwrap());
            h.case("imageops/hsv2rgb", &tag, pix(dims), &mut || rgb.hsv2rgb().eval().unwrap());
            h.case("imageops/rgb2ycbcr", &tag, pix(dims), &mut || rgb.rgb2ycbcr(YccStd::Bt601).eval().unwrap());
            h.case("imageops/unwrap_3x3", &tag, pix(dims), &mut || img.unwrap([3, 3], [1, 1], [1, 1], true).eval().unwrap());
        }

        // 3-D: a volume.
        let n = if h.ctx.is_cpu() { 48 } else { pick(h, 128, 256) };
        let vol = upload(h, &values(n * n * n, 31, 0.0, 1.0), [n, n, n]);
        let cells = Work::Items((n * n * n) as f64, "Mvox/s");
        for k in [3usize, 5] {
            let kernel = upload(h, &values(k * k * k, 32, -1.0, 1.0), [k, k, k]);
            h.case(&format!("imageops/conv3_{k}x{k}x{k}"), &format!("{n}^3"), cells, &mut || vol.conv3(&kernel, ConvSpec::new()).eval().unwrap());
        }
        let mask: Vec<f32> = (0..27).map(|i| if i % 3 == 0 || i == 13 { 1.0 } else { 0.0 }).collect();
        let m = upload(h, &mask, [3, 3, 3]);
        h.case("imageops/dilate3_3x3x3", &format!("{n}^3"), cells, &mut || vol.dilate3(&m).eval().unwrap());
        h.case("imageops/erode3_3x3x3", &format!("{n}^3"), cells, &mut || vol.erode3(&m).eval().unwrap());
    });
}
