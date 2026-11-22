//! Feature detection and image analysis (`fast`, ...) against naive host computations.

use crate::common::values;
use luma::{Array, Context};

fn ctx() -> Context {
    Context::new().unwrap()
}

/// A noisy image with structure: blocks of different brightness plus noise, so there are corners
/// of every kind.
fn scene(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let noise = values(h * w, seed, 0.0, 24.0);
    let blocks = values(64, seed + 1, 0.0, 255.0);
    (0..h * w)
        .map(|k| {
            let (i, j) = (k % h, k / h);
            blocks[((i / 9) * 7 + (j / 11) * 3) % 64] + noise[k]
        })
        .collect()
}

const RING: [(i64, i64); 16] = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3), (0, -3), (1, -3), (2, -2), (3, -1)];

/// The FAST score of every pixel (0 where it is not a corner), by the definition.
fn fast_scores(data: &[f32], h: usize, w: usize, thr: f32, arc: usize, edge: usize) -> Vec<f32> {
    let mut score = vec![0.0f32; h * w];
    for j in edge..w.saturating_sub(edge) {
        for i in edge..h.saturating_sub(edge) {
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
            let run = |f: &[bool]| (0..16).any(|s| (0..arc).all(|d| f[(s + d) % 16]));
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

/// The corners `(x, y, score)` with `x` along axis 1, sorted.
fn fast_ref(data: &[f32], h: usize, w: usize, thr: f32, arc: usize, non_max: bool, edge: usize) -> Vec<(u32, u32, f32)> {
    let score = fast_scores(data, h, w, thr, arc, edge);
    let mut out = Vec::new();
    for j in 0..w {
        for i in 0..h {
            let s = score[i + h * j];
            if s <= 0.0 {
                continue;
            }
            let mut keep = true;
            if non_max {
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        if (di, dj) == (0, 0) {
                            continue;
                        }
                        let (a, b) = (i as i64 + di, j as i64 + dj);
                        if a >= 0 && b >= 0 && (a as usize) < h && (b as usize) < w && score[a as usize + h * b as usize] >= s {
                            keep = false;
                        }
                    }
                }
            }
            if keep {
                out.push((j as u32, i as u32, s));
            }
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn sorted_points(k: &luma::Keypoints) -> Vec<(u32, u32, f32)> {
    let (x, y, s) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap());
    let mut v: Vec<_> = (0..x.len()).map(|i| (x[i] as u32, y[i] as u32, s[i])).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn fast_matches_the_definition() {
    let ctx = ctx();
    for (h, w, arc, non_max, thr) in [(64, 80, 9, true, 20.0f32), (45, 70, 12, true, 30.0), (100, 33, 9, false, 25.0), (37, 91, 16, true, 15.0), (70, 70, 10, true, 20.0), (290, 1010, 9, true, 20.0)] {
        let data = scene(h, w, 7 + arc as u64);
        let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
        let want = fast_ref(&data, h, w, thr, arc, non_max, 3);
        let got = img.fast(thr, arc as u32, non_max, 1.0, 3).unwrap();
        match got {
            None => assert!(want.is_empty(), "{h}x{w} arc {arc}: got none, want {}", want.len()),
            Some(k) => {
                assert_eq!(sorted_points(&k), want, "{h}x{w} arc {arc} non_max {non_max}");
                assert!(k.orientation.to_vec().unwrap().iter().all(|&o| o == 0.0));
                assert!(k.size.to_vec().unwrap().iter().all(|&s| s == 1.0));
            }
        }
    }
}

#[test]
fn fast_edge_and_limit() {
    let ctx = ctx();
    let (h, w) = (64, 64);
    let data = scene(h, w, 3);
    let img = Array::from_slice_on(&ctx, &data, [h, w]).unwrap();
    let want = fast_ref(&data, h, w, 20.0, 9, true, 8);
    let got = img.fast(20.0, 9, true, 1.0, 8).unwrap().unwrap();
    assert_eq!(sorted_points(&got), want);
    // The limit keeps that many of the points, all of them valid.
    let all = fast_ref(&data, h, w, 20.0, 9, true, 3);
    let limited = img.fast(20.0, 9, true, 0.01, 3).unwrap().unwrap();
    assert_eq!(limited.count(), (64.0f64 * 64.0 * 0.01).ceil() as usize);
    for p in sorted_points(&limited) {
        assert!(all.contains(&p), "{p:?} is not a corner");
    }
}

#[test]
fn fast_nothing_and_errors() {
    let ctx = ctx();
    let flat = Array::<f32>::full_on(&ctx, [32, 32], 5.0).unwrap();
    assert!(flat.fast(20.0, 9, true, 0.1, 3).unwrap().is_none());
    assert!(flat.fast(20.0, 8, true, 0.1, 3).is_err());
    assert!(flat.fast(20.0, 9, true, 0.1, 2).is_err());
    assert!(flat.fast(20.0, 9, true, 0.0, 3).is_err());
    let small = Array::<f32>::full_on(&ctx, [6, 6], 5.0).unwrap();
    assert!(small.fast(20.0, 9, true, 0.1, 3).unwrap().is_none());
    let cube = Array::<f32>::zeros_on(&ctx, [8, 8, 2]).unwrap();
    assert!(cube.fast(20.0, 9, true, 0.1, 3).is_err());
}

// ---------------------------------------------------------------------------------------------
// homography
// ---------------------------------------------------------------------------------------------

use luma::{homography, HomographyMethod};

/// Points under `h` (row-major), with every `outlier_every`-th destination replaced by noise.
fn warp(h: &[f64; 9], n: usize, outlier_every: usize, seed: u64) -> [Vec<f32>; 4] {
    let r = values(4 * n, seed, 0.0, 1.0);
    let (mut xs, mut ys, mut xd, mut yd) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let (x, y) = (f64::from(r[i]) * 300.0, f64::from(r[n + i]) * 200.0);
        let w = h[6] * x + h[7] * y + h[8];
        let (u, v) = ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w);
        xs.push(x as f32);
        ys.push(y as f32);
        if outlier_every > 0 && i % outlier_every == 0 {
            xd.push(r[2 * n + i] * 300.0);
            yd.push(r[3 * n + i] * 200.0);
        } else {
            xd.push(u as f32);
            yd.push(v as f32);
        }
    }
    [xs, ys, xd, yd]
}

#[test]
fn homography_recovers_a_perspective_transform_among_outliers() {
    let ctx = ctx();
    let truth = [1.1, 0.1, 5.0, -0.05, 0.95, -3.0, 0.0004, 0.0002, 1.0];
    let n = 200;
    let [xs, ys, xd, yd] = warp(&truth, n, 5, 99);
    let up = |v: &[f32]| Array::from_slice_on(&ctx, v, v.len()).unwrap();
    let outliers = (0..n).filter(|i| i % 5 == 0).count();
    for method in [HomographyMethod::Ransac, HomographyMethod::Lmeds] {
        let (h, inliers) = homography(&up(&xs), &up(&ys), &up(&xd), &up(&yd), method, 1.0, 500);
        assert_eq!(h.shape().dims(), [3, 3, 1, 1]);
        let h = h.to_vec().unwrap();
        for k in 0..9 {
            let tol = if k >= 6 { 5e-5 } else { 5e-3 * truth[k].abs().max(1.0) };
            assert!((f64::from(h[k]) - truth[k]).abs() <= tol, "{method:?} h[{k}] = {} want {}", h[k], truth[k]);
        }
        let inliers = inliers.item().unwrap() as usize;
        // The robust threshold of the median method is not a hard bound on numerical noise.
        assert!(inliers >= (n - outliers) * 9 / 10 && inliers <= n, "{method:?}: {inliers} inliers");
    }
}

#[test]
fn homography_is_deterministic_and_checks_arguments() {
    let ctx = ctx();
    let truth = [1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0];
    let [xs, ys, xd, yd] = warp(&truth, 40, 0, 5);
    let up = |v: &[f32]| Array::from_slice_on(&ctx, v, v.len()).unwrap();
    let a = homography(&up(&xs), &up(&ys), &up(&xd), &up(&yd), HomographyMethod::Ransac, 0.5, 100);
    let b = homography(&up(&xs), &up(&ys), &up(&xd), &up(&yd), HomographyMethod::Ransac, 0.5, 100);
    assert_eq!(a.0.to_vec().unwrap(), b.0.to_vec().unwrap());
    assert_eq!(a.1.item().unwrap(), 40);
    let few = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    assert!(homography(&few, &few, &few, &few, HomographyMethod::Ransac, 1.0, 10).0.to_vec().is_err());
    assert!(homography(&up(&xs), &up(&ys), &up(&xd), &up(&yd), HomographyMethod::Ransac, 1.0, 0).1.item().is_err());
    assert!(homography(&up(&xs), &up(&ys), &up(&xd), &few, HomographyMethod::Ransac, 1.0, 10).1.item().is_err());
}
