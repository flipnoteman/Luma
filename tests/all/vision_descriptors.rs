//! ORB, SIFT and GLOH: tests against host references.

use std::collections::HashMap;

use luma::{Array, Context, Keypoints};

fn ctx() -> Context {
    Context::new().unwrap()
}

// ---------------------------------------------------------------------------------------------
// Test images
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

/// A natural-looking image in `0..=255` (integers): value noise at four scales and a few
/// rectangles, so there are corners and blobs of every size. Axis 0 is the row.
fn natural(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut img = vec![0f32; h * w];
    for &(cell, amp) in &[(2usize, 0.25f32), (5, 0.5), (12, 1.0), (30, 1.5)] {
        let (gh, gw) = (h / cell + 3, w / cell + 3);
        let g: Vec<f32> = (0..gh * gw).map(|_| rng.next()).collect();
        for j in 0..w {
            for i in 0..h {
                let (fy, fx) = (i as f32 / cell as f32, j as f32 / cell as f32);
                let (y0, x0) = (fy as usize, fx as usize);
                let sm = |t: f32| t * t * (3.0 - 2.0 * t);
                let (ty, tx) = (sm(fy - y0 as f32), sm(fx - x0 as f32));
                let v = g[y0 + gh * x0] * (1.0 - ty) * (1.0 - tx) + g[y0 + 1 + gh * x0] * ty * (1.0 - tx) + g[y0 + gh * (x0 + 1)] * (1.0 - ty) * tx + g[y0 + 1 + gh * (x0 + 1)] * ty * tx;
                img[i + h * j] += amp * v;
            }
        }
    }
    for _ in 0..h * w / 3000 {
        let (r0, c0) = ((rng.next() * h as f32) as usize, (rng.next() * w as f32) as usize);
        let (dh, dw) = (3 + (rng.next() * 20.0) as usize, 3 + (rng.next() * 20.0) as usize);
        let v = rng.next() * 1.5 - 0.5;
        for j in c0..(c0 + dw).min(w) {
            for i in r0..(r0 + dh).min(h) {
                img[i + h * j] += v;
            }
        }
    }
    let (mn, mx) = img.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    img.iter().map(|v| ((v - mn) / (mx - mn) * 255.0).round()).collect()
}

// ---------------------------------------------------------------------------------------------
// ORB reference
// ---------------------------------------------------------------------------------------------

/// The BRIEF pattern, read from the source file that holds it.
fn pattern() -> Vec<i32> {
    let text = include_str!("../../src/vision/descriptors/orb_pattern.rs");
    let start = text.find("PATTERN: [i32; 1024] = [").unwrap();
    let body = &text[start + 24..];
    let body = &body[..body.find("];").unwrap()];
    let v: Vec<i32> = body.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    assert_eq!(v.len(), 1024);
    v
}

/// ArrayFire's bilinear resize (no half-pixel offset, clamped at the last cell).
fn resize(src: &[f32], ih: usize, iw: usize, oh: usize, ow: usize) -> Vec<f32> {
    let (sy, sx) = (ih as f32 / oh as f32, iw as f32 / ow as f32);
    let mut out = vec![0f32; oh * ow];
    for oy in 0..ow {
        for ox in 0..oh {
            let (xf, yf) = (ox as f32 * sy, oy as f32 * sx);
            let (mut ix, mut iy) = (xf.floor() as usize, yf.floor() as usize);
            ix = ix.min(ih - 1);
            iy = iy.min(iw - 1);
            let (b, a) = (xf - ix as f32, yf - iy as f32);
            let ix2 = if ix + 1 < ih { ix + 1 } else { ix };
            let iy2 = if iy + 1 < iw { iy + 1 } else { iy };
            let p = |i: usize, j: usize| src[i + ih * j];
            out[ox + oh * oy] = ((1.0 - a) * (1.0 - b)) * p(ix, iy) + (a * (1.0 - b)) * p(ix, iy2) + ((1.0 - a) * b) * p(ix2, iy) + (a * b) * p(ix2, iy2);
        }
    }
    out
}

const RING: [(i64, i64); 16] = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3), (0, -3), (1, -3), (2, -2), (3, -1)];

/// The FAST (arc 9) score of every pixel in `[edge, n - edge)`, by the definition.
fn fast_scores(data: &[f32], h: usize, w: usize, thr: f32, edge: usize) -> Vec<f32> {
    let mut score = vec![0.0f32; h * w];
    for j in edge..w - edge {
        for i in edge..h - edge {
            let p = data[i + h * j];
            let ring: Vec<f32> = RING.iter().map(|&(d0, d1)| data[(i as i64 + d0) as usize + h * (j as i64 + d1) as usize]).collect();
            let (mut sb, mut sd) = (0.0f32, 0.0f32);
            let bright: Vec<bool> = ring.iter().map(|&r| r > p + thr).collect();
            let dark: Vec<bool> = ring.iter().map(|&r| r < p - thr).collect();
            for k in 0..16 {
                if bright[k] {
                    sb += ring[k] - (p + thr);
                }
                if dark[k] {
                    sd += (p - thr) - ring[k];
                }
            }
            let run = |f: &[bool]| (0..16).any(|s| (0..9).all(|d| f[(s + d) % 16]));
            let mut out = 0.0f32;
            if run(&bright) {
                out = sb;
            }
            if run(&dark) {
                out = out.max(sd);
            }
            score[i + h * j] = out;
        }
    }
    score
}

#[derive(Clone, Debug)]
struct Feature {
    x: f32,
    y: f32,
    score: f32,
    ori: f32,
    size: f32,
    desc: [u32; 8],
}

fn gauss9() -> [f32; 9] {
    let mut k = [0f64; 9];
    for (i, v) in k.iter_mut().enumerate() {
        let x = i as f64 - 4.0;
        *v = (-x * x / 8.0).exp();
    }
    let s: f64 = k.iter().sum();
    let mut out = [0f32; 9];
    for i in 0..9 {
        out[i] = (k[i] / s) as f32;
    }
    out
}

/// Zero-padded separable correlation with a 9-tap kernel.
fn blur9(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let k = gauss9();
    let mut tmp = vec![0f32; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0;
            for t in 0..9 {
                let ii = i as i64 + t as i64 - 4;
                if ii >= 0 && (ii as usize) < h {
                    s += k[t] * img[ii as usize + h * j];
                }
            }
            tmp[i + h * j] = s;
        }
    }
    let mut out = vec![0f32; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0;
            for t in 0..9 {
                let jj = j as i64 + t as i64 - 4;
                if jj >= 0 && (jj as usize) < w {
                    s += k[t] * tmp[i + h * jj as usize];
                }
            }
            out[i + h * j] = s;
        }
    }
    out
}

fn round_away(v: f32) -> f32 {
    v.round()
}

/// ORB as ArrayFire's CUDA backend does it, on the host.
fn orb_ref(data: &[f32], h: usize, w: usize, thr: f32, max_feat: u32, scl: f32, levels: u32, blur: bool) -> Vec<Feature> {
    let pat = pattern();
    // Levels.
    let mut min_side = h.min(w) as u32;
    let (mut n, mut sum) = (0u32, 0f32);
    for i in 0..levels {
        min_side = (min_side as f32 / scl) as u32;
        if min_side < 31 || n == levels {
            break;
        }
        n += 1;
        sum += 1.0 / scl.powf(i as f32);
    }
    let n = n as usize;
    let scales: Vec<f32> = (0..n).map(|i| scl.powf(i as f32)).collect();
    let mut best = Vec::new();
    let mut used = 0u32;
    for s in scales.iter().take(n.saturating_sub(1)) {
        let b = ((max_feat as f32 / sum) / s).ceil() as u32;
        used += b;
        best.push(b);
    }
    if n > 0 {
        best.push(max_feat.saturating_sub(used));
    }
    let mut out = Vec::new();
    let mut prev: Vec<f32> = data.to_vec();
    let (mut ph, mut pw) = (h, w);
    for i in 0..n {
        let (lh, lw, img) = if i == 0 {
            (h, w, data.to_vec())
        } else {
            let lh = ((h as f32 / scales[i]).round() as usize).max(1);
            let lw = ((w as f32 / scales[i]).round() as usize).max(1);
            (lh, lw, resize(&prev, ph, pw, lh, lw))
        };
        prev = img.clone();
        (ph, pw) = (lh, lw);
        if lh <= 48 || lw <= 48 {
            continue;
        }
        let score = fast_scores(&img, lh, lw, thr, 22);
        let at = |i: i64, j: i64| if i < 0 || j < 0 || i >= lh as i64 || j >= lw as i64 { 0.0 } else { score[i as usize + lh * j as usize] };
        let mut cands = Vec::new();
        for j in 24..=lw - 24 {
            for ii in 24..=lh - 24 {
                let s0 = score[ii + lh * j];
                if s0 <= 0.0 {
                    continue;
                }
                let mut keep = true;
                for a in -1i64..=1 {
                    for b in -1i64..=1 {
                        if (a != 0 || b != 0) && s0 <= at(ii as i64 + a, j as i64 + b) {
                            keep = false;
                        }
                    }
                }
                if keep {
                    cands.push((ii, j));
                }
            }
        }
        // Harris.
        let rscale = 0.001f32 * 0.001 * 0.001 * 0.001;
        let harris = |row: usize, col: usize| -> f32 {
            let (mut ixx, mut iyy, mut ixy) = (0f32, 0f32, 0f32);
            for k in 0..49usize {
                let (di, dj) = ((k / 7) as i64 - 3, (k % 7) as i64 - 3);
                let px = |r: i64, c: i64| img[(row as i64 + r) as usize + lh * (col as i64 + c) as usize];
                let ix = px(dj, di + 1) - px(dj, di - 1);
                let iy = px(dj + 1, di) - px(dj - 1, di);
                ixx += ix * ix;
                iyy += iy * iy;
                ixy += ix * iy;
            }
            let tr = ixx + iyy;
            let det = ixx * iyy - ixy * ixy;
            (det - 0.04 * (tr * tr)) * rscale
        };
        let mut ranked: Vec<(usize, usize, f32)> = cands.iter().map(|&(r, c)| (r, c, harris(r, c))).collect();
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        ranked.truncate(best[i] as usize);
        let desc_img = if blur { blur9(&img, lh, lw) } else { img.clone() };
        for (row, col, hr) in ranked {
            let (mut m0, mut m1) = (0f64, 0f64);
            for jj in -15i64..=15 {
                for ii in -15i64..=15 {
                    let p = img[(row as i64 + jj) as usize + lh * (col as i64 + ii) as usize] as f64;
                    m0 += jj as f64 * p;
                    m1 += ii as f64 * p;
                }
            }
            let ori = (m0 as f32).atan2(m1 as f32);
            let (sn, cs) = (ori.sin(), ori.cos());
            let mut desc = [0u32; 8];
            for t in 0..256 {
                let (x1, y1, x2, y2) = (pat[4 * t] as f32, pat[4 * t + 1] as f32, pat[4 * t + 2] as f32, pat[4 * t + 3] as f32);
                let px = |x: f32, y: f32| {
                    let c = col as i64 + round_away(x * cs - y * sn) as i64;
                    let r = row as i64 + round_away(x * sn + y * cs) as i64;
                    desc_img[r as usize + lh * c as usize]
                };
                if px(x1, y1) < px(x2, y2) {
                    desc[t / 32] |= 1 << (t % 32);
                }
            }
            out.push(Feature { x: round_away(col as f32 * scales[i]), y: round_away(row as f32 * scales[i]), score: hr, ori, size: 31.0 * scales[i], desc });
        }
    }
    out
}

fn read_orb(r: Option<(Keypoints, Array<u32>)>) -> Vec<Feature> {
    let Some((k, d)) = r else { return Vec::new() };
    let (x, y, s, o, z) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap(), k.orientation.to_vec().unwrap(), k.size.to_vec().unwrap());
    let d = d.to_vec().unwrap();
    (0..x.len()).map(|i| Feature { x: x[i], y: y[i], score: s[i], ori: o[i], size: z[i], desc: d[8 * i..8 * i + 8].try_into().unwrap() }).collect()
}

fn hamming(a: &[u32; 8], b: &[u32; 8]) -> u32 {
    a.iter().zip(b).map(|(p, q)| (p ^ q).count_ones()).sum()
}

fn key(f: &Feature) -> (i32, i32, i32) {
    (f.x as i32, f.y as i32, (f.size * 100.0) as i32)
}

/// Checks `got` against `want`: the same features (up to a few ties at the cut), the same
/// order of levels and scores, descriptors within a few bits.
fn check_orb(got: &[Feature], want: &[Feature], what: &str) {
    assert!(!want.is_empty(), "{what}: the reference found nothing");
    let wmap: HashMap<_, _> = want.iter().map(|f| (key(f), f)).collect();
    let mut common = 0;
    let mut bits = 0;
    for g in got {
        let Some(w) = wmap.get(&key(g)) else { continue };
        common += 1;
        assert!((g.score - w.score).abs() <= 1e-3 * w.score.abs() + 1e-9, "{what}: score {} vs {}", g.score, w.score);
        let d = (g.ori - w.ori).abs();
        assert!(d.min((d - 2.0 * std::f32::consts::PI).abs()) < 1e-3, "{what}: orientation {} vs {}", g.ori, w.ori);
        bits += hamming(&g.desc, &w.desc);
    }
    let n = got.len().max(want.len());
    assert!(common * 100 >= n * 98, "{what}: {common} common features of {} and {}", got.len(), want.len());
    assert!(bits <= common as u32 / 2 + 2, "{what}: {bits} differing descriptor bits over {common} features");
    // Levels in order (finest first), each sorted by decreasing score.
    for w in got.windows(2) {
        assert!(w[0].size <= w[1].size, "{what}: levels follow each other");
        if w[0].size == w[1].size {
            assert!(w[0].score >= w[1].score, "{what}: scores of a level are sorted");
        }
    }
}

#[test]
fn orb_matches_the_reference() {
    let ctx = ctx();
    for &(h, w, blur) in &[(200usize, 240usize, false), (240, 200, true)] {
        let data = natural(h, w, 7);
        let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
        let got = read_orb(img.orb(30.0, 300, 1.5, 4, blur).unwrap());
        let want = orb_ref(&data, h, w, 30.0, 300, 1.5, 4, blur);
        check_orb(&got, &want, &format!("orb {h}x{w} blur={blur}"));
    }
}

#[test]
fn orb_scale_and_levels() {
    let ctx = ctx();
    let (h, w) = (260usize, 260usize);
    let data = natural(h, w, 11);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    for &(thr, max_feat, scl, levels) in &[(25.0f32, 2000u32, 1.3f32, 6u32), (40.0, 100, 2.0, 3), (35.0, 50, 1.5, 1)] {
        let got = read_orb(img.orb(thr, max_feat, scl, levels, false).unwrap());
        let want = orb_ref(&data, h, w, thr, max_feat, scl, levels, false);
        check_orb(&got, &want, &format!("orb thr={thr} max={max_feat} scale={scl} levels={levels}"));
    }
}

#[test]
fn orb_keypoint_fields() {
    let ctx = ctx();
    let (h, w) = (180usize, 220usize);
    let data = natural(h, w, 3);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    let (k, d) = img.orb(30.0, 200, 1.5, 3, false).unwrap().expect("features");
    let n = k.count();
    assert_eq!(d.shape().dims(), [8, n, 1, 1]);
    for v in [&k.x, &k.y, &k.score, &k.orientation, &k.size] {
        assert_eq!(v.len(), n);
    }
    // The coordinates are in the input image and inside it; sizes are 31 * 1.5^level.
    let (x, y, z) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.size.to_vec().unwrap());
    for i in 0..n {
        assert!(x[i] >= 0.0 && (x[i] as usize) < w && y[i] >= 0.0 && (y[i] as usize) < h);
        assert!([31.0, 46.5, 69.75].iter().any(|s: &f32| (s - z[i]).abs() < 1e-3), "size {}", z[i]);
    }
    assert!(n <= 200 + 3);
}

#[test]
fn orb_flat_and_small_images() {
    let ctx = ctx();
    let flat = Array::from_slice_on(&ctx, &vec![7.0f32; 100 * 100], [100, 100]).unwrap();
    assert!(flat.orb(20.0, 100, 1.5, 4, false).unwrap().is_none());
    let tiny = Array::from_slice_on(&ctx, &natural(40, 40, 1), [40, 40]).unwrap();
    assert!(tiny.orb(5.0, 100, 1.5, 4, false).unwrap().is_none());
}

#[test]
fn orb_argument_errors() {
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &natural(100, 100, 1), [100, 100]).unwrap();
    assert!(img.orb(0.0, 100, 1.5, 4, false).is_err());
    assert!(img.orb(20.0, 0, 1.5, 4, false).is_err());
    assert!(img.orb(20.0, 100, 1.0, 4, false).is_err());
    assert!(img.orb(20.0, 100, 1.5, 0, false).is_err());
    let small = Array::from_slice_on(&ctx, &vec![0.0f32; 36], [6, 6]).unwrap();
    assert!(small.orb(20.0, 100, 1.5, 4, false).is_err());
    let cube = Array::from_slice_on(&ctx, &vec![0.0f32; 100 * 100 * 2], [100, 100, 2]).unwrap();
    assert!(cube.orb(20.0, 100, 1.5, 4, false).is_err());
}

// ---------------------------------------------------------------------------------------------
// SIFT / GLOH reference (ArrayFire's algorithm, on the host)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SiftFeature {
    x: f32,
    y: f32,
    score: f32,
    /// In degrees, as ArrayFire returns it.
    ori: f32,
    size: f32,
    desc: Vec<f32>,
}

const PI: f32 = 3.14159265358979323846;

fn gauss_taps(sigma: f32) -> Vec<f32> {
    let len = ((sigma * 6.0 + 1.0).round() as usize | 1).min(31);
    let sigma = sigma as f64;
    let mut out: Vec<f32> = (0..len)
        .map(|i| {
            let x = i as i32 - (len as i32 - 1) / 2;
            (1.0 / (2.0 * PI as f64 * sigma * sigma).sqrt() * (-((x * x) as f64) / (2.0 * sigma * sigma)).exp()) as f32
        })
        .collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// The separable zero-padded correlation of an `[h, w]` image with `taps` along both axes.
fn blur_sep(img: &[f32], h: usize, w: usize, taps: &[f32]) -> Vec<f32> {
    let r = (taps.len() / 2) as i64;
    let mut tmp = vec![0f32; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0f32;
            for (t, &k) in taps.iter().enumerate() {
                let ii = i as i64 + t as i64 - r;
                if ii >= 0 && (ii as usize) < h {
                    s += k * img[ii as usize + h * j];
                }
            }
            tmp[i + h * j] = s;
        }
    }
    let mut out = vec![0f32; h * w];
    for j in 0..w {
        for i in 0..h {
            let mut s = 0.0f32;
            for (t, &k) in taps.iter().enumerate() {
                let jj = j as i64 + t as i64 - r;
                if jj >= 0 && (jj as usize) < w {
                    s += k * tmp[i + h * jj as usize];
                }
            }
            out[i + h * j] = s;
        }
    }
    out
}

/// ArrayFire's three-by-three "Gaussian elimination": the forward elimination is right, the
/// back substitution runs forwards and never computes `x[2]` (so the scale offset is 0).
fn af_solve(a: &mut [f32; 9], b: &mut [f32; 3]) -> [f32; 3] {
    for i in 0..2 {
        for j in i + 1..3 {
            let s = a[j * 3 + i] / a[i * 3 + i];
            for k in i..3 {
                a[j * 3 + k] -= s * a[i * 3 + k];
            }
            b[j] -= s * b[i];
        }
    }
    let mut x = [0f32; 3];
    for i in 0..2 {
        let mut sum = b[i];
        for j in i + 1..3 {
            sum -= a[i * 3 + j] * x[j];
        }
        x[i] = sum / a[i * 3 + i];
    }
    x
}

#[allow(clippy::too_many_arguments)]
fn sift_ref(data: &[f32], h: usize, w: usize, n_layers: u32, contrast_thr: f32, edge_thr: f32, init_sigma: f32, double_input: bool, img_scale: f32, feature_ratio: f32, gloh: bool) -> Vec<SiftFeature> {
    let nl = n_layers as usize;
    let mut min_dim = h.min(w);
    if double_input {
        min_dim *= 2;
    }
    let n_octaves = ((min_dim as f64).ln() / 2f64.ln()).floor() as usize - 2;
    // Initial image.
    let s = if double_input { ((init_sigma * init_sigma - 0.25 * 4.0).sqrt()).max(0.1) } else { ((init_sigma * init_sigma - 0.25).sqrt()).max(0.1) };
    let (mut ih, mut iw) = (h, w);
    let mut init = data.to_vec();
    if double_input {
        init = resize(data, h, w, 2 * h, 2 * w);
        (ih, iw) = (2 * h, 2 * w);
    }
    init = blur_sep(&init, ih, iw, &gauss_taps(s));
    // Gaussian pyramid.
    let k = 2f32.powf(1.0 / n_layers as f32);
    let mut sig = vec![init_sigma; nl + 3];
    for (i, v) in sig.iter_mut().enumerate().skip(1) {
        let prev = k.powf((i - 1) as f32) * init_sigma;
        let total = prev * k;
        *v = (total * total - prev * prev).sqrt();
    }
    let mut dims: Vec<(usize, usize)> = Vec::new();
    let mut gauss: Vec<Vec<Vec<f32>>> = Vec::new();
    for o in 0..n_octaves {
        let (oh, ow) = if o == 0 { (ih, iw) } else { (dims[o - 1].0 / 2, dims[o - 1].1 / 2) };
        dims.push((oh, ow));
        let mut layers: Vec<Vec<f32>> = Vec::new();
        for l in 0..nl + 3 {
            if o == 0 && l == 0 {
                layers.push(init.clone());
            } else if l == 0 {
                let src = &gauss[o - 1][nl];
                layers.push(resize(src, dims[o - 1].0, dims[o - 1].1, oh, ow));
            } else {
                layers.push(blur_sep(&layers[l - 1], oh, ow, &gauss_taps(sig[l])));
            }
        }
        gauss.push(layers);
    }
    let first_deriv = img_scale * 0.5;
    let second_deriv = img_scale;
    let cross_deriv = img_scale * 0.25;
    let mut out = Vec::new();
    for o in 0..n_octaves {
        let (d0, d1) = dims[o];
        if d0 < 11 || d1 < 11 {
            continue;
        }
        let dog: Vec<Vec<f32>> = (0..nl + 2).map(|l| gauss[o][l + 1].iter().zip(&gauss[o][l]).map(|(a, b)| a - b).collect()).collect();
        let max_feat = ((d0 * d1) as f32 * feature_ratio).ceil() as usize;
        // Extrema (x is the column, y the row).
        let thr = 0.5 * contrast_thr / n_layers as f32;
        let mut ext: Vec<(usize, usize, usize)> = Vec::new();
        for l in 1..=nl {
            for y in 5..d1 - 5 {
                for x in 5..d0 - 5 {
                    let p = dog[l][y * d0 + x];
                    if p.abs() > thr {
                        let mut gt = p > 0.0;
                        let mut lt = p < 0.0;
                        for dl in [l - 1, l, l + 1] {
                            for dy in -1i64..=1 {
                                for dx in -1i64..=1 {
                                    if dl == l && dy == 0 && dx == 0 {
                                        continue;
                                    }
                                    let q = dog[dl][((y as i64 + dy) as usize) * d0 + (x as i64 + dx) as usize];
                                    gt &= p > q;
                                    lt &= p < q;
                                }
                            }
                        }
                        if (gt || lt) && ext.len() < max_feat {
                            ext.push((y, x, l));
                        }
                    }
                }
            }
        }
        // Interpolation (the first element is the column).
        struct Interp {
            x: f32,
            y: f32,
            layer: usize,
            resp: f32,
            size: f32,
        }
        let mut interp: Vec<Interp> = Vec::new();
        for &(ex, ey, el) in &ext {
            let (mut x, mut y, mut layer) = (ex as i64, ey as i64, el as i64);
            let at = |l: i64, c: i64, r: i64| dog[l as usize][(c as usize) * d0 + r as usize];
            let (mut xl, mut xy, mut xx) = (0f32, 0f32, 0f32);
            let mut converged = false;
            let mut failed = false;
            for _ in 0..5 {
                let cp = |c: i64, r: i64| at(layer, c, r);
                let pp = |c: i64, r: i64| at(layer - 1, c, r);
                let np = |c: i64, r: i64| at(layer + 1, c, r);
                let mut dd = [(cp(x + 1, y) - cp(x - 1, y)) * first_deriv, (cp(x, y + 1) - cp(x, y - 1)) * first_deriv, (np(x, y) - pp(x, y)) * first_deriv];
                let d2 = cp(x, y) * 2.0;
                let dxx = (cp(x + 1, y) + cp(x - 1, y) - d2) * second_deriv;
                let dyy = (cp(x, y + 1) + cp(x, y - 1) - d2) * second_deriv;
                let dss = (np(x, y) + pp(x, y) - d2) * second_deriv;
                let dxy = (cp(x + 1, y + 1) - cp(x - 1, y + 1) - cp(x + 1, y - 1) + cp(x - 1, y - 1)) * cross_deriv;
                let dxs = (np(x + 1, y) - np(x - 1, y) - pp(x + 1, y) + pp(x - 1, y)) * cross_deriv;
                let dys = (np(x, y + 1) - np(x - 1, y - 1) - pp(x, y - 1) + pp(x - 1, y - 1)) * cross_deriv;
                let mut hm = [dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss];
                let sol = af_solve(&mut hm, &mut dd);
                xl = -sol[2];
                xy = -sol[1];
                xx = -sol[0];
                if xl.abs() < 0.5 && xy.abs() < 0.5 && xx.abs() < 0.5 {
                    converged = true;
                    break;
                }
                x += xx.round() as i64;
                y += xy.round() as i64;
                layer += xl.round() as i64;
                if layer < 1 || layer > nl as i64 || x < 5 || x >= d1 as i64 - 5 || y < 5 || y >= d0 as i64 - 5 {
                    failed = true;
                    break;
                }
            }
            if !converged || failed {
                continue;
            }
            let at = |l: i64, c: i64, r: i64| dog[l as usize][(c as usize) * d0 + r as usize];
            let cp = |c: i64, r: i64| at(layer, c, r);
            let pp = |c: i64, r: i64| at(layer - 1, c, r);
            let np = |c: i64, r: i64| at(layer + 1, c, r);
            let dd = [(cp(x + 1, y) - cp(x - 1, y)) * first_deriv, (cp(x, y + 1) - cp(x, y - 1)) * first_deriv, (np(x, y) - pp(x, y)) * first_deriv];
            let xv = [xx, xy, xl];
            let pv = dd[0] * xv[0] + dd[1] * xv[1] + dd[2] * xv[2];
            let contr = cp(x, y) * img_scale + pv * 0.5;
            if contr.abs() < contrast_thr / n_layers as f32 {
                continue;
            }
            let d2 = cp(x, y) * 2.0;
            let dxx = (cp(x + 1, y) + cp(x - 1, y) - d2) * second_deriv;
            let dyy = (cp(x, y + 1) + cp(x, y - 1) - d2) * second_deriv;
            let dxy = (cp(x + 1, y + 1) - cp(x - 1, y + 1) - cp(x + 1, y - 1) + cp(x - 1, y - 1)) * cross_deriv;
            let tr = dxx + dyy;
            let det = dxx * dyy - dxy * dxy;
            if det <= 0.0 || tr * tr * edge_thr >= (edge_thr + 1.0) * (edge_thr + 1.0) * det + f32::EPSILON {
                continue;
            }
            if interp.len() < max_feat {
                interp.push(Interp {
                    x: (x as f32 + xx) * (1u32 << o) as f32,
                    y: (y as f32 + xy) * (1u32 << o) as f32,
                    layer: layer as usize,
                    resp: contr.abs(),
                    size: init_sigma * 2f32.powf(o as f32 + (layer as f32 + xl) / n_layers as f32) * 2.0,
                });
            }
        }
        // Sort and remove duplicates.
        interp.sort_by(|a, b| {
            a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()).then(a.resp.partial_cmp(&b.resp).unwrap()).then(a.size.partial_cmp(&b.size).unwrap()).then(a.layer.cmp(&b.layer))
        });
        let mut nodup: Vec<&Interp> = Vec::new();
        for (f, a) in interp.iter().enumerate() {
            if f + 1 < interp.len() {
                let b = &interp[f + 1];
                let q = |v: f32| (v * 1e4).round();
                if q(a.x) == q(b.x) && q(a.y) == q(b.y) && q(a.resp) == q(b.resp) && q(a.size) == q(b.size) && a.layer == b.layer {
                    continue;
                }
            }
            nodup.push(a);
        }
        // Orientation.
        let nbins = 36usize;
        let mut oriented: Vec<(f32, f32, usize, f32, f32, f32)> = Vec::new();
        for f in &nodup {
            let pt_x = (f.x / (1u32 << o) as f32).round() as i64;
            let pt_y = (f.y / (1u32 << o) as f32).round() as i64;
            let scl_octv = f.size * 0.5 / (1u32 << o) as f32;
            let radius = (4.5 * scl_octv).round() as i64;
            let sigma = 1.5 * scl_octv;
            let len = radius * 2 + 1;
            let exp_denom = 2.0 * sigma * sigma;
            let img = &gauss[o][f.layer];
            let mut hist = vec![0f32; nbins];
            for l in 0..len * len {
                let i = l / len - radius;
                let j = l % len - radius;
                let y = pt_y + i;
                let x = pt_x + j;
                if y < 1 || y >= d0 as i64 - 1 || x < 1 || x >= d1 as i64 - 1 {
                    continue;
                }
                let px = |c: i64, r: i64| img[(c as usize) * d0 + r as usize];
                let dx = px(x + 1, y) - px(x - 1, y);
                let dy = px(x, y - 1) - px(x, y + 1);
                let mag = (dx * dx + dy * dy).sqrt();
                let ori = dy.atan2(dx);
                let wgt = (-((i * i + j * j) as f32) / exp_denom).exp();
                let mut bin = (nbins as f32 * (ori + PI) / (2.0 * PI)).round() as usize;
                if bin >= nbins {
                    bin = 0;
                }
                hist[bin] += wgt * mag;
            }
            for _ in 0..2 {
                let t = hist.clone();
                for j in 0..nbins {
                    let prev = if j == 0 { t[nbins - 1] } else { t[j - 1] };
                    let next = if j + 1 == nbins { t[0] } else { t[j + 1] };
                    hist[j] = 0.25 * prev + 0.5 * t[j] + 0.25 * next;
                }
            }
            let omax = hist.iter().cloned().fold(hist[0], f32::max);
            let mag_thr = omax * 0.8;
            for j in 0..nbins {
                let l = if j == 0 { nbins - 1 } else { j - 1 };
                let r = (j + 1) % nbins;
                if hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= mag_thr {
                    let mut bin = j as f32 + 0.5 * (hist[l] - hist[r]) / (hist[l] - 2.0 * hist[j] + hist[r]);
                    bin = if bin < 0.0 { bin + nbins as f32 } else if bin >= nbins as f32 { bin - nbins as f32 } else { bin };
                    let ori = 360.0 - (360.0 / nbins as f32) * bin;
                    let (mut nx, mut ny, mut ns) = (f.x, f.y, f.size);
                    if double_input {
                        nx *= 0.5;
                        ny *= 0.5;
                        ns *= 0.5;
                    }
                    oriented.push((nx, ny, f.layer, f.resp, ns, ori));
                }
            }
        }
        // Descriptors.
        let mut scale = 1.0 / (1u32 << o) as f32;
        if double_input {
            scale *= 2.0;
        }
        let (d, nb) = (4usize, 8usize);
        let desc_len = if gloh { 272 } else { d * d * nb };
        for &(fx, fy, layer, resp, size, ori_deg) in &oriented {
            let mut ori = (360.0 - ori_deg) * PI / 180.0;
            if ori > PI {
                ori -= PI * 2.0;
            }
            let (cx, cy) = ((fx * scale).round() as i64, (fy * scale).round() as i64);
            let img = &gauss[o][layer];
            let (cos_t, sin_t) = (ori.cos(), ori.sin());
            let hist_width = 3.0 * size * scale * 0.5;
            let radius = (hist_width * 2f32.sqrt() * (d as f32 + 1.0) * 0.5 + 0.5) as i64;
            let len = radius * 2 + 1;
            let mut desc = vec![0f32; desc_len];
            let px = |c: i64, r: i64| img[(c as usize) * d0 + r as usize];
            for l in 0..len * len {
                let i = l / len - radius;
                let j = l % len - radius;
                let y = cy + i;
                let x = cx + j;
                if !(y > 0 && y < d0 as i64 - 1 && x > 0 && x < d1 as i64 - 1) {
                    continue;
                }
                let dx = px(x + 1, y) - px(x - 1, y);
                let dy = px(x, y - 1) - px(x, y + 1);
                let grad_mag = (dx * dx + dy * dy).sqrt();
                let mut grad_ori = dy.atan2(dx) - ori;
                while grad_ori < 0.0 {
                    grad_ori += PI * 2.0;
                }
                while grad_ori >= PI * 2.0 {
                    grad_ori -= PI * 2.0;
                }
                if !gloh {
                    let x_rot = (j as f32 * cos_t - i as f32 * sin_t) / hist_width;
                    let y_rot = (j as f32 * sin_t + i as f32 * cos_t) / hist_width;
                    let mut xbin = x_rot + (d / 2) as f32 - 0.5;
                    let mut ybin = y_rot + (d / 2) as f32 - 0.5;
                    if !(ybin > -1.0 && ybin < d as f32 && xbin > -1.0 && xbin < d as f32) {
                        continue;
                    }
                    let wgt = (-(x_rot * x_rot + y_rot * y_rot) / (d * d) as f32 * 2.0).exp();
                    let mut obin = grad_ori * (nb as f32 / (PI * 2.0));
                    let mag = grad_mag * wgt;
                    let (x0, y0, o0) = (xbin.floor() as i64, ybin.floor() as i64, obin.floor() as i64);
                    xbin -= x0 as f32;
                    ybin -= y0 as f32;
                    obin -= o0 as f32;
                    for yl in 0..=1i64 {
                        let yb = y0 + yl;
                        if yb >= 0 && yb < d as i64 {
                            let v_y = mag * if yl == 0 { 1.0 - ybin } else { ybin };
                            for xl in 0..=1i64 {
                                let xb = x0 + xl;
                                if xb >= 0 && xb < d as i64 {
                                    let v_x = v_y * if xl == 0 { 1.0 - xbin } else { xbin };
                                    for ol in 0..=1i64 {
                                        let ob = ((o0 + ol) % nb as i64) as usize;
                                        let v_o = v_x * if ol == 0 { 1.0 - obin } else { obin };
                                        desc[(yb as usize * d + xb as usize) * nb + ob] += v_o;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let radii = [6.0f32, 11.0, 15.0];
                    let (rb, ab, hb) = (3usize, 8usize, 16usize);
                    let x_rot = j as f32 * cos_t - i as f32 * sin_t;
                    let y_rot = j as f32 * sin_t + i as f32 * cos_t;
                    let r = (x_rot * x_rot + y_rot * y_rot).sqrt() / radius as f32 * radii[rb - 1];
                    let mut theta = y_rot.atan2(x_rot);
                    while theta < 0.0 {
                        theta += PI * 2.0;
                    }
                    while theta >= PI * 2.0 {
                        theta -= PI * 2.0;
                    }
                    let mut tbin = theta * (ab as f32 / (PI * 2.0));
                    let mut rbin = if r < radii[0] {
                        r / radii[0]
                    } else if r < radii[1] {
                        1.0 + (r - radii[0]) / (radii[1] - radii[0])
                    } else {
                        (2.0 + (r - radii[1]) / (radii[2] - radii[1])).min(3.0 - f32::EPSILON)
                    };
                    if !(r <= radii[rb - 1]) {
                        continue;
                    }
                    let wgt = (-r / (radii[rb - 1] * 0.5)).exp();
                    let mut obin = grad_ori * (hb as f32 / (PI * 2.0));
                    let mag = grad_mag * wgt;
                    let (t0, r0, o0) = (tbin.floor() as i64, rbin.floor() as i64, obin.floor() as i64);
                    tbin -= t0 as f32;
                    rbin -= r0 as f32;
                    obin -= o0 as f32;
                    for rl in 0..=1i64 {
                        let rbb = if rbin > 0.5 { r0 + rl } else { r0 - rl };
                        let v_r = mag * if rl == 0 { 1.0 - rbin } else { rbin };
                        if (0..=2).contains(&rbb) {
                            for tl in 0..=1i64 {
                                let tb = ((t0 + tl) % ab as i64) as usize;
                                let v_t = v_r * if tl == 0 { 1.0 - tbin } else { tbin };
                                for ol in 0..=1i64 {
                                    let ob = ((o0 + ol) % hb as i64) as usize;
                                    let v_o = v_t * if ol == 0 { 1.0 - obin } else { obin };
                                    let idx = if rbb > 0 { hb + ((rbb as usize - 1) * ab + tb) * hb } else { 0 } + ob;
                                    desc[idx] += v_o;
                                }
                            }
                        }
                    }
                }
            }
            let normalize = |d: &mut [f32]| {
                let s: f32 = d.iter().map(|v| v * v).sum();
                let inv = 1.0 / s.sqrt();
                for v in d.iter_mut() {
                    *v *= inv;
                }
            };
            normalize(&mut desc);
            for v in desc.iter_mut() {
                *v = v.min(0.2);
            }
            normalize(&mut desc);
            for v in desc.iter_mut() {
                *v = (*v * 512.0).min(255.0).round();
            }
            out.push(SiftFeature { x: fx, y: fy, score: resp, ori: ori_deg, size, desc });
        }
    }
    out
}

fn read_sift(r: Option<(Keypoints, Array<f32>)>) -> Vec<SiftFeature> {
    let Some((k, d)) = r else { return Vec::new() };
    let (x, y, sc, o, z) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap(), k.orientation.to_vec().unwrap(), k.size.to_vec().unwrap());
    let len = d.shape().dims()[0];
    let d = d.to_vec().unwrap();
    (0..x.len()).map(|i| SiftFeature { x: x[i], y: y[i], score: sc[i], ori: o[i].to_degrees(), size: z[i], desc: d[len * i..len * (i + 1)].to_vec() }).collect()
}

/// `got` against `want` as sets: every reference point has a point of the same position, size
/// and orientation, with the same score and (almost) the same descriptor.
fn check_sift(got: &[SiftFeature], want: &[SiftFeature], what: &str) {
    assert!(!want.is_empty(), "{what}: the reference found nothing");
    let mut matched = 0usize;
    let mut worst = 0f32;
    let mut used = vec![false; got.len()];
    for r in want {
        let found = got.iter().enumerate().find(|(i, g)| {
            !used[*i] && (g.x - r.x).abs() < 2e-3 * (1.0 + r.x.abs()) && (g.y - r.y).abs() < 2e-3 * (1.0 + r.y.abs()) && (g.size - r.size).abs() < 2e-3 * r.size && {
                let d = (g.ori - r.ori).abs();
                d.min(360.0 - d) < 0.2
            }
        });
        if let Some((i, g)) = found {
            used[i] = true;
            assert!((g.score - r.score).abs() <= 2e-3 * r.score + 1e-6, "{what}: score {} vs {}", g.score, r.score);
            let dd = g.desc.iter().zip(&r.desc).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            worst = worst.max(dd);
            if dd <= 4.0 {
                matched += 1;
            }
        }
    }
    assert!(matched * 100 >= want.len() * 97, "{what}: {matched} of {} reference points matched ({} found), worst descriptor difference {worst}", want.len(), got.len());
    assert!(got.len().abs_diff(want.len()) * 100 <= want.len() * 3 + 3, "{what}: {} points, reference {}", got.len(), want.len());
}

#[test]
fn sift_matches_the_reference() {
    let ctx = ctx();
    let (h, w) = (150usize, 170usize);
    let data = natural(h, w, 5);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    for &(layers, double) in &[(3u32, true), (3, false), (2, false)] {
        let got = read_sift(img.sift(layers, 0.04, 10.0, 1.6, double, 1.0 / 256.0, 0.05).unwrap());
        let want = sift_ref(&data, h, w, layers, 0.04, 10.0, 1.6, double, 1.0 / 256.0, 0.05, false);
        check_sift(&got, &want, &format!("sift layers={layers} double={double}"));
    }
}

#[test]
fn gloh_matches_the_reference() {
    let ctx = ctx();
    let (h, w) = (170usize, 150usize);
    let data = natural(h, w, 9);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    for &double in &[true, false] {
        let got = read_sift(img.gloh(3, 0.04, 10.0, 1.6, double, 1.0 / 256.0, 0.05).unwrap());
        let want = sift_ref(&data, h, w, 3, 0.04, 10.0, 1.6, double, 1.0 / 256.0, 0.05, true);
        check_sift(&got, &want, &format!("gloh double={double}"));
    }
}

#[test]
fn sift_parameters() {
    let ctx = ctx();
    let (h, w) = (120usize, 120usize);
    let data = natural(h, w, 2);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    // A stricter contrast threshold keeps fewer points; the ratio caps them per octave.
    let loose = img.sift(3, 0.02, 10.0, 1.6, false, 1.0 / 256.0, 0.05).unwrap().unwrap().0.count();
    let strict = img.sift(3, 0.08, 10.0, 1.6, false, 1.0 / 256.0, 0.05).unwrap().unwrap().0.count();
    assert!(strict < loose, "{strict} < {loose}");
    let want = sift_ref(&data, h, w, 3, 0.04, 5.0, 1.6, false, 1.0 / 256.0, 0.05, false);
    let got = read_sift(img.sift(3, 0.04, 5.0, 1.6, false, 1.0 / 256.0, 0.05).unwrap());
    check_sift(&got, &want, "edge_thr 5");
    // Intensity scale: a scaled image with the inverse scale is the same image.
    let scaled: Vec<f32> = data.iter().map(|v| v / 255.0).collect();
    let img1 = Array::from_slice_on(&ctx, &scaled, [h, w]).unwrap();
    let a = img.sift(3, 0.04, 10.0, 1.6, false, 1.0 / 256.0, 0.05).unwrap().unwrap().0.count();
    let b = img1.sift(3, 0.04, 10.0, 1.6, false, 1.0, 0.05).unwrap().unwrap().0.count();
    assert!(a.abs_diff(b) * 20 <= a, "{a} vs {b}");
}

#[test]
fn sift_fields_and_errors() {
    let ctx = ctx();
    let (h, w) = (140usize, 160usize);
    let img = Array::from_slice_on(&ctx, &natural(h, w, 4), [h, w]).unwrap();
    let (k, d) = img.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap().expect("points");
    let n = k.count();
    assert_eq!(d.shape().dims(), [128, n, 1, 1]);
    let (x, y, o, z) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.orientation.to_vec().unwrap(), k.size.to_vec().unwrap());
    let dv = d.to_vec().unwrap();
    for i in 0..n {
        assert!(x[i] >= 0.0 && x[i] < w as f32 && y[i] >= 0.0 && y[i] < h as f32);
        assert!((0.0..2.0 * std::f32::consts::PI).contains(&o[i]) && z[i] > 1.0);
    }
    assert!(dv.iter().all(|v| (0.0..=255.0).contains(v) && v.fract() == 0.0));
    let flat = Array::from_slice_on(&ctx, &vec![3.0f32; 64 * 64], [64, 64]).unwrap();
    assert!(flat.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap().is_none());
    assert!(img.sift(0, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).is_err());
    assert!(img.sift(3, 0.0, 10.0, 1.6, true, 1.0 / 256.0, 0.05).is_err());
    assert!(img.sift(3, 0.04, 0.5, 1.6, true, 1.0 / 256.0, 0.05).is_err());
    assert!(img.sift(3, 0.04, 10.0, 0.5, true, 1.0 / 256.0, 0.05).is_err());
    assert!(img.gloh(3, 0.04, 10.0, 1.6, true, 0.0, 0.05).is_err());
    let small = Array::from_slice_on(&ctx, &vec![0.0f32; 14 * 14], [14, 14]).unwrap();
    assert!(small.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).is_err());
}
