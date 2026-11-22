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
    for (h, w, arc, non_max, thr) in [(64, 80, 9, true, 20.0f32), (45, 70, 12, true, 30.0), (100, 33, 9, false, 25.0), (37, 91, 16, true, 15.0), (70, 70, 10, true, 20.0)] {
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
