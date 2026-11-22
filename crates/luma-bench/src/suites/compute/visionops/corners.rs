//! `harris`, `susan` and `dog` at 1080p and 4K. Each is checked against a host computation on a
//! small image before it is timed.

use luma::*;

use super::super::{check64, host, pick, upload};
use crate::harness::*;

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

/// Blocks of different brightness plus noise: corners of every kind, about as many as a photograph.
fn scene(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let noise = values(h * w, seed, 0.0, 12.0);
    let blocks = values(256, seed + 1, 0.0, 255.0);
    (0..h * w)
        .map(|k| {
            let (i, j) = (k % h, k / h);
            blocks[((i / 17) * 13 + (j / 23) * 5) % 256] + noise[k]
        })
        .collect()
}

/// `(x, y)` of the points, sorted.
fn points(k: &Keypoints) -> Vec<(u32, u32)> {
    let (x, y) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap());
    let mut v: Vec<_> = (0..x.len()).map(|i| (x[i] as u32, y[i] as u32)).collect();
    v.sort_unstable();
    v
}

/// The Harris corners of a Gaussian window of `sigma` (5 cells at 1.0) as `(x, y)`, sorted.
fn harris_host(data: &[f32], h: usize, w: usize, sigma: f64, kk: f64) -> Vec<(u32, u32)> {
    let floor = (6.0 * sigma).floor() as usize;
    let n = if floor % 2 == 0 { floor - 1 } else { floor };
    let c = (n - 1) / 2;
    let g: Vec<f64> = (0..n).map(|i| (-((i as f64 - c as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f64 = g.iter().sum();
    let g: Vec<f64> = g.iter().map(|v| v / total).collect();
    let at = |i: usize, j: usize| f64::from(data[i + h * j]);
    let mut prod = vec![[0.0f64; 3]; h * w];
    for j in 1..w - 1 {
        for i in 1..h - 1 {
            let (gy, gx) = ((at(i + 1, j) - at(i - 1, j)) / 2.0, (at(i, j + 1) - at(i, j - 1)) / 2.0);
            prod[i + h * j] = [gx * gx, gy * gy, gx * gy];
        }
    }
    let b = n / 2 + 1;
    let mut r = vec![0.0; h * w];
    for j in b..w - b {
        for i in b..h - b {
            let mut s = [0.0f64; 3];
            for bb in 0..n {
                for a in 0..n {
                    let p = prod[(i + n / 2 - a) + h * (j + n / 2 - bb)];
                    for m in 0..3 {
                        s[m] += g[a] * g[bb] * p[m];
                    }
                }
            }
            r[i + h * j] = s[0] * s[1] - s[2] * s[2] - kk * (s[0] + s[1]).powi(2);
        }
    }
    maxima(&r, h, w, b + 1)
}

/// The strict positive maxima of `r` among the pixels `lo..dim - lo`, as `(x, y)`, sorted.
fn maxima(r: &[f64], h: usize, w: usize, lo: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for j in lo..w - lo {
        for i in lo..h - lo {
            let v = r[i + h * j];
            if v > 0.0 && (0..9).all(|k| k == 4 || r[i + k % 3 - 1 + h * (j + k / 3 - 1)] < v) {
                out.push((j as u32, i as u32));
            }
        }
    }
    out.sort_unstable();
    out
}

fn susan_host(data: &[f32], h: usize, w: usize, r: i64, t: f64, g: f64, edge: usize) -> Vec<(u32, u32)> {
    let mut resp = vec![0.0; h * w];
    for j in edge..w - edge {
        for i in edge..h - edge {
            let c = f64::from(data[i + h * j]);
            let mut n = 0.0;
            for dj in -r..=r {
                for di in -r..=r {
                    if di * di + dj * dj < r * r {
                        n += (-((c - f64::from(data[(i as i64 + di) as usize + h * (j as i64 + dj) as usize])) / t).powi(6)).exp();
                    }
                }
            }
            if n < g {
                resp[i + h * j] = g - n;
            }
        }
    }
    maxima(&resp, h, w, edge + 1)
}

/// The blur with the Gaussian kernel of `2 r + 1` cells (sigma a quarter of that), zero border.
fn blur_host(data: &[f32], h: usize, w: usize, r: usize) -> Vec<f64> {
    let n = 2 * r + 1;
    let sigma = 0.25 * n as f64;
    let g: Vec<f64> = (0..n).map(|i| (-((i as f64 - r as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f64 = g.iter().sum();
    let mut out = vec![0.0; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0;
            for b in 0..n {
                for a in 0..n {
                    let (ii, jj) = (i as i64 + a as i64 - r as i64, j as i64 + b as i64 - r as i64);
                    if (0..h as i64).contains(&ii) && (0..w as i64).contains(&jj) {
                        s += g[a] * g[b] / (total * total) * f64::from(data[ii as usize + h * jj as usize]);
                    }
                }
            }
            out[i + h * j] = s;
        }
    }
    out
}

pub fn run(h: &mut Harness) {
    h.guarded("visionops/corners", |h| {
        // Small checks against the host (the corner sets must be the same).
        let (sh, sw) = (61usize, 83usize);
        let sdata = scene(sh, sw, 3);
        let simg = upload(h, &sdata, [sh, sw]);
        let want = harris_host(&sdata, sh, sw, 1.0, 0.04);
        let got = points(&simg.harris(100000, 0.0, 1.0, 0, 0.04).unwrap().expect("harris corners"));
        assert_eq!(got, want, "visionops/harris: corners");
        let want = susan_host(&sdata, sh, sw, 3, 32.0, 10.0, 3);
        let got = points(&simg.susan(3, 32.0, 10.0, 1.0, 3).unwrap().expect("susan corners"));
        assert_eq!(got, want, "visionops/susan: corners");
        let want: Vec<f64> = blur_host(&sdata, sh, sw, 2).iter().zip(blur_host(&sdata, sh, sw, 4)).map(|(a, b)| a - b).collect();
        check64("visionops/dog", &host(&simg.dog(2, 4)), &want, 1e-4, 2e-3);

        for dims in sizes(h) {
            let data = scene(dims[0], dims[1], 7);
            let img = upload(h, &data, dims);
            let tag = label(dims);
            h.case("visionops/corners_harris_500", &tag, pix(dims), &mut || {
                img.harris(500, 1e5, 1.0, 0, 0.04).unwrap();
            });
            h.case("visionops/corners_harris_all", &tag, pix(dims), &mut || {
                img.harris(0, 1e5, 1.0, 0, 0.04).unwrap();
            });
            h.case("visionops/corners_harris_box3", &tag, pix(dims), &mut || {
                img.harris(500, 1e5, 0.1, 3, 0.04).unwrap();
            });
            h.case("visionops/corners_harris_sigma3", &tag, pix(dims), &mut || {
                img.harris(500, 1e5, 3.0, 0, 0.04).unwrap();
            });
            h.case("visionops/corners_susan_r3", &tag, pix(dims), &mut || {
                img.susan(3, 32.0, 10.0, 0.05, 3).unwrap();
            });
            h.case("visionops/corners_susan_r5", &tag, pix(dims), &mut || {
                img.susan(5, 32.0, 30.0, 0.05, 5).unwrap();
            });
            for (r1, r2) in [(1usize, 2usize), (3, 5), (8, 8 + 4)] {
                h.case(&format!("visionops/corners_dog_{r1}_{r2}"), &tag, pix(dims), &mut || img.dog(r1, r2).eval().unwrap());
            }
        }
    });
}
