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
