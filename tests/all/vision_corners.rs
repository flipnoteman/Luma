//! Harris and SUSAN corners, difference of Gaussians: tests against host references.
use crate::common::{assert_close, values};
use luma::{Array, Context};

fn ctx() -> Context {
    Context::new().unwrap()
}

/// A noisy image of blocks of different brightness: corners of every kind.
fn scene(h: usize, w: usize, seed: u64, noise: f32) -> Vec<f32> {
    let n = values(h * w, seed, 0.0, noise);
    let blocks = values(64, seed + 1, 0.0, 255.0);
    (0..h * w)
        .map(|k| {
            let (i, j) = (k % h, k / h);
            blocks[((i / 9) * 7 + (j / 11) * 3) % 64] + n[k]
        })
        .collect()
}

/// `(x, y, score)` of the points, `x` along axis 1, sorted by position.
fn points(k: &luma::Keypoints) -> Vec<(u32, u32, f32)> {
    let (x, y, s) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap());
    let mut v: Vec<_> = (0..x.len()).map(|i| (x[i] as u32, y[i] as u32, s[i])).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// The corners `(i, j, v)` (`i` along axis 0) that are strict maxima of the response `r` among
/// the pixels `lo..dim - lo` and positive.
fn strict_maxima(r: &[f64], h: usize, w: usize, lo: usize) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for j in lo..w - lo {
        for i in lo..h - lo {
            let v = r[i + h * j];
            if v <= 0.0 {
                continue;
            }
            let mut keep = true;
            for dj in 0..3 {
                for di in 0..3 {
                    if (di, dj) != (1, 1) && r[i + di - 1 + h * (j + dj - 1)] >= v {
                        keep = false;
                    }
                }
            }
            if keep {
                out.push((i, j, v));
            }
        }
    }
    out
}

/// The 1-D window of Harris in ArrayFire: a Gaussian with the centre at `(n - 1) / 2` or a box.
fn window(n: usize, sigma: f32, gaussian: bool) -> Vec<f64> {
    if !gaussian {
        return vec![1.0 / n as f64; n];
    }
    let c = ((n - 1) / 2) as f64;
    let g: Vec<f64> = (0..n).map(|i| (-(i as f64 - c).powi(2) / (2.0 * (sigma as f64).powi(2))).exp()).collect();
    let sum: f64 = g.iter().sum();
    g.iter().map(|v| v / sum).collect()
}

/// The Harris response of every pixel whose window (and its neighbours' windows) lie inside the
/// image, zero elsewhere.
fn harris_response(data: &[f32], h: usize, w: usize, k: &[f64], kk: f64) -> Vec<f64> {
    let at = |i: usize, j: usize| f64::from(data[i + h * j]);
    let n = k.len();
    let c = n / 2;
    let border = n / 2 + 1;
    // Central differences everywhere they exist (the window never reaches the others).
    let mut ixx = vec![0.0; h * w];
    let mut iyy = vec![0.0; h * w];
    let mut ixy = vec![0.0; h * w];
    for j in 1..w - 1 {
        for i in 1..h - 1 {
            let gy = (at(i + 1, j) - at(i - 1, j)) / 2.0;
            let gx = (at(i, j + 1) - at(i, j - 1)) / 2.0;
            ixx[i + h * j] = gx * gx;
            iyy[i + h * j] = gy * gy;
            ixy[i + h * j] = gx * gy;
        }
    }
    let sum = |p: &[f64], i: usize, j: usize| {
        let mut s = 0.0;
        for a in 0..n {
            for b in 0..n {
                s += k[a] * k[b] * p[(i + c - a) + h * (j + c - b)];
            }
        }
        s
    };
    let mut r = vec![0.0; h * w];
    for j in border..w - border {
        for i in border..h - border {
            let (a, d, e) = (sum(&ixx, i, j), sum(&iyy, i, j), sum(&ixy, i, j));
            r[i + h * j] = a * d - e * e - kk * (a + d) * (a + d);
        }
    }
    r
}

fn harris_ref(data: &[f32], h: usize, w: usize, k: &[f64], kk: f64) -> Vec<(u32, u32, f64)> {
    let r = harris_response(data, h, w, k, kk);
    let n = k.len();
    let mut v: Vec<_> = strict_maxima(&r, h, w, n / 2 + 2).into_iter().map(|(i, j, s)| (j as u32, i as u32, s)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn assert_points(got: &[(u32, u32, f32)], want: &[(u32, u32, f64)], what: &str) {
    let pos = |v: &[(u32, u32, f32)]| v.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>();
    let want_pos: Vec<_> = want.iter().map(|p| (p.0, p.1)).collect();
    assert_eq!(pos(got), want_pos, "{what}: positions");
    let g: Vec<f32> = got.iter().map(|p| p.2).collect();
    let w: Vec<f32> = want.iter().map(|p| p.2 as f32).collect();
    assert_close(&g, &w, 2e-3, 1e-2);
}

#[test]
fn harris_matches_the_definition() {
    let ctx = ctx();
    // (h, w, sigma, block, window length, gaussian)
    for (h, w, sigma, block, n, gaussian) in [
        (50, 64, 1.0f32, 0usize, 5usize, true),
        (64, 50, 2.0, 0, 11, true),
        (37, 71, 0.6, 0, 3, true),
        (60, 45, 0.1, 3, 3, false),
        (45, 60, 0.1, 4, 4, false),
        (48, 48, 1.0, 6, 6, true),
        (70, 33, 1.5, 0, 9, true),
        (50, 50, 0.1, 9, 9, false),
        (90, 100, 5.0, 0, 29, true),
        (80, 85, 0.1, 32, 32, false),
        (80, 85, 3.0, 31, 31, true),
        (30, 30, 0.5, 1, 1, true),
    ] {
        let data = scene(h, w, (h * w) as u64, 60.0);
        let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
        let want = harris_ref(&data, h, w, &window(n, sigma, gaussian), 0.04);
        let got = img.harris(0, 1e-20, sigma, block, 0.04).unwrap();
        match got {
            None => assert!(want.is_empty(), "{h}x{w} {sigma} {block}: none, want {}", want.len()),
            Some(k) => {
                assert!(k.orientation.to_vec().unwrap().iter().all(|&o| o == 0.0));
                assert!(k.size.to_vec().unwrap().iter().all(|&s| s == 1.0));
                assert_points(&points(&k), &want, &format!("{h}x{w} sigma {sigma} block {block}"));
            }
        }
    }
}

#[test]
fn harris_limits_and_threshold() {
    let ctx = ctx();
    let (h, w) = (80, 90);
    let data = scene(h, w, 9, 60.0);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    let want = harris_ref(&data, h, w, &window(5, 1.0, true), 0.04);
    assert!(want.len() > 40);
    // The strongest ones, in descending order.
    let mut by_score = want.clone();
    by_score.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let k = img.harris(25, 1e5, 1.0, 0, 0.04).unwrap().unwrap();
    assert_eq!(k.count(), 25);
    let (x, y, s) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap());
    for i in 0..25 {
        assert_eq!((x[i] as u32, y[i] as u32), (by_score[i].0, by_score[i].1), "rank {i}");
        assert!((f64::from(s[i]) - by_score[i].2).abs() <= 1e-3 * by_score[i].2.abs() + 1e-2);
    }
    // More than there are: all of them.
    let all = img.harris(100000, 0.0, 1.0, 0, 0.04).unwrap().unwrap();
    assert_eq!(all.count(), want.len());
    // The threshold counts (`>=`), max_corners 0.
    let min = by_score[want.len() / 2].2 as f32;
    let k = img.harris(0, min, 1.0, 0, 0.04).unwrap().unwrap();
    let expect = want.iter().filter(|p| p.2 as f32 >= min * (1.0 + 1e-4)).count();
    assert!(k.count() >= expect && k.count() <= expect + 2, "{} vs {}", k.count(), expect);
    assert!(k.score.to_vec().unwrap().iter().all(|&s| s >= min));
    // A flat image has no corner.
    let flat = Array::<f32>::full_on(&ctx, [40, 40], 3.0).unwrap();
    assert!(flat.harris(10, 0.0, 1.0, 0, 0.04).unwrap().is_none());
    // Too small for any.
    let small = Array::from_slice_on(&ctx, &scene(8, 8, 1, 50.0), [8, 8]).unwrap();
    assert!(small.harris(10, 0.0, 1.0, 0, 0.04).unwrap().is_none());
}

#[test]
fn harris_errors() {
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &scene(40, 40, 1, 50.0), [40, 40]).unwrap();
    assert!(img.harris(0, 0.0, 1.0, 0, 0.04).is_err());
    assert!(img.harris(0, -1.0, 1.0, 0, 0.04).is_err());
    assert!(img.harris(10, 0.0, 0.4, 0, 0.04).is_err());
    assert!(img.harris(10, 0.0, 5.1, 0, 0.04).is_err());
    assert!(img.harris(10, 0.0, 1.0, 33, 0.04).is_err());
    assert!(img.harris(10, 0.0, 1.0, 0, 0.009).is_err());
    assert!(img.harris(10, 0.0, 1.0, 0, f32::NAN).is_err());
    let cube = Array::<f32>::zeros_on(&ctx, [20, 20, 2]).unwrap();
    assert!(cube.harris(10, 0.0, 1.0, 0, 0.04).is_err());
    // Accepted by ArrayFire as well: 0 < block <= 32, any sigma.
    assert!(img.harris(10, 0.0, 9.0, 3, 0.04).is_ok());
    assert!(img.harris(10, 0.0, 1.0, 1, 0.04).is_ok());
}

/// The SUSAN response of every pixel at least `edge` from the border, zero elsewhere.
fn susan_response(data: &[f32], h: usize, w: usize, r: usize, t: f64, g: f64, edge: usize) -> Vec<f64> {
    let mut out = vec![0.0; h * w];
    let ri = r as i64;
    for j in edge..w - edge {
        for i in edge..h - edge {
            let c = f64::from(data[i + h * j]);
            let mut n = 0.0;
            for dj in -ri..=ri {
                for di in -ri..=ri {
                    if di * di + dj * dj < ri * ri {
                        let v = f64::from(data[(i as i64 + di) as usize + h * (j as i64 + dj) as usize]);
                        n += (-((c - v) / t).powi(6)).exp();
                    }
                }
            }
            if n < g {
                out[i + h * j] = g - n;
            }
        }
    }
    out
}

fn susan_ref(data: &[f32], h: usize, w: usize, r: usize, t: f64, g: f64, edge: usize) -> Vec<(u32, u32, f64)> {
    let resp = susan_response(data, h, w, r, t, g, edge);
    let mut v: Vec<_> = strict_maxima(&resp, h, w, edge + 1).into_iter().map(|(i, j, s)| (j as u32, i as u32, s)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn susan_matches_the_definition() {
    let ctx = ctx();
    for (h, w, r, t, g, edge) in [
        (48, 48, 3, 32.0f32, 10.0f32, 3),
        (40, 56, 3, 32.0, 10.0, 5),
        (56, 40, 2, 32.0, 4.0, 3),
        (50, 61, 4, 20.0, 20.0, 4),
        (47, 53, 5, 25.0, 30.0, 6),
        (45, 45, 1, 32.0, 3.0, 3),
        (60, 41, 9, 25.0, 40.0, 9),
        (41, 60, 0, 25.0, 40.0, 2),
    ] {
        let data = scene(h, w, (h + w) as u64, 30.0);
        let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
        let want = susan_ref(&data, h, w, r, f64::from(t), f64::from(g), edge);
        match img.susan(r, t, g, 1.0, edge).unwrap() {
            None => assert!(want.is_empty(), "{h}x{w} r {r}: none, want {}", want.len()),
            Some(k) => {
                assert!(k.orientation.to_vec().unwrap().iter().all(|&o| o == 0.0));
                assert!(k.size.to_vec().unwrap().iter().all(|&s| s == 1.0));
                assert_points(&points(&k), &want, &format!("{h}x{w} r {r} edge {edge}"));
            }
        }
    }
}

#[test]
fn susan_limit_keeps_the_first_in_memory_order() {
    let ctx = ctx();
    let (h, w) = (64, 72);
    let data = scene(h, w, 5, 30.0);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    let all = susan_ref(&data, h, w, 3, 32.0, 10.0, 3);
    assert!(all.len() > 12);
    // Memory order: axis 1 (x) major, axis 0 (y) minor.
    let ratio = 12.5 / (h * w) as f32;
    let k = img.susan(3, 32.0, 10.0, ratio, 3).unwrap().unwrap();
    assert_eq!(k.count(), 12);
    let got: Vec<(u32, u32)> = points(&k).iter().map(|p| (p.0, p.1)).collect();
    let want: Vec<(u32, u32)> = all.iter().take(12).map(|p| (p.0, p.1)).collect();
    assert_eq!(got, want);
    // A limit of no point is no result.
    assert!(img.susan(3, 32.0, 10.0, 1e-6, 3).unwrap().is_none());
}

#[test]
fn susan_errors() {
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &scene(40, 40, 1, 30.0), [40, 40]).unwrap();
    assert!(img.susan(4, 32.0, 10.0, 0.1, 3).is_err());
    assert!(img.susan(10, 32.0, 10.0, 0.1, 10).is_err());
    assert!(img.susan(3, 0.0, 10.0, 0.1, 3).is_err());
    assert!(img.susan(3, 32.0, -1.0, 0.1, 3).is_err());
    assert!(img.susan(3, 32.0, 10.0, 0.0, 3).is_err());
    assert!(img.susan(3, 32.0, 10.0, 1.5, 3).is_err());
    assert!(img.susan(3, f32::NAN, 10.0, 0.5, 3).is_err());
    let cube = Array::<f32>::zeros_on(&ctx, [20, 20, 2]).unwrap();
    assert!(cube.susan(3, 32.0, 10.0, 0.5, 3).is_err());
    // An edge wider than the image: nothing.
    assert!(img.susan(3, 32.0, 10.0, 0.5, 30).unwrap().is_none());
}

/// The blur of `data` (zero border) with the Gaussian kernel of `2 r + 1` cells a side of
/// ArrayFire's `gaussian_kernel` (sigma `0.25 * n`).
fn blur(data: &[f32], h: usize, w: usize, r: usize) -> Vec<f64> {
    let n = 2 * r + 1;
    let sigma = 0.25 * n as f64;
    let g: Vec<f64> = (0..n).map(|i| (-((i as f64 - r as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = g.iter().sum();
    let g: Vec<f64> = g.iter().map(|v| v / sum).collect();
    let mut out = vec![0.0; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0;
            for b in 0..n {
                for a in 0..n {
                    let (ii, jj) = (i as i64 + a as i64 - r as i64, j as i64 + b as i64 - r as i64);
                    if ii >= 0 && jj >= 0 && (ii as usize) < h && (jj as usize) < w {
                        s += g[a] * g[b] * f64::from(data[ii as usize + h * jj as usize]);
                    }
                }
            }
            out[i + h * j] = s;
        }
    }
    out
}

#[test]
fn dog_is_the_difference_of_blurs() {
    let ctx = ctx();
    for (h, w, r1, r2) in [(12, 14, 1, 2), (12, 14, 2, 1), (20, 17, 3, 5), (9, 31, 0, 1), (16, 16, 2, 2), (10, 10, 1, 7)] {
        let data = values(h * w, (h * w + r1 + r2) as u64, 0.0, 255.0);
        let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
        let (a, b) = (blur(&data, h, w, r1), blur(&data, h, w, r2));
        let want: Vec<f32> = a.iter().zip(&b).map(|(a, b)| (a - b) as f32).collect();
        let got = img.dog(r1, r2);
        assert_eq!(got.shape().dims(), [h, w, 1, 1]);
        assert_close(&got.to_vec().unwrap(), &want, 1e-4, 2e-3);
    }
}

#[test]
fn dog_batches_planes() {
    let ctx = ctx();
    let (h, w, n) = (15, 13, 3);
    let data = values(h * w * n, 4, 0.0, 100.0);
    let vol = Array::from_slice_on(&ctx, &data, [h, w, n]).unwrap();
    let got = vol.dog(2, 4).to_vec().unwrap();
    for p in 0..n {
        let plane = &data[p * h * w..(p + 1) * h * w];
        let (a, b) = (blur(plane, h, w, 2), blur(plane, h, w, 4));
        let want: Vec<f32> = a.iter().zip(&b).map(|(a, b)| (a - b) as f32).collect();
        assert_close(&got[p * h * w..(p + 1) * h * w], &want, 1e-4, 2e-3);
    }
}
