//! Connected-component labelling, confidence connected components, deconvolution: tests against host references.

use crate::common::values;
use luma::{Array, Connectivity, Context};

fn ctx() -> Context {
    Context::new().unwrap()
}

/// Labels by flood fill from the first unlabelled pixel in memory order: the numbering of ArrayFire.
fn regions_ref(data: &[bool], h: usize, w: usize, eight: bool) -> Vec<u32> {
    let mut out = vec![0u32; h * w];
    let mut next = 0;
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !data[start] || out[start] != 0 {
            continue;
        }
        next += 1;
        out[start] = next;
        stack.push(start);
        while let Some(q) = stack.pop() {
            let (i, j) = ((q % h) as i64, (q / h) as i64);
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    if (di == 0 && dj == 0) || (!eight && di != 0 && dj != 0) {
                        continue;
                    }
                    let (a, b) = (i + di, j + dj);
                    if a < 0 || b < 0 || a >= h as i64 || b >= w as i64 {
                        continue;
                    }
                    let n = a as usize + h * b as usize;
                    if data[n] && out[n] == 0 {
                        out[n] = next;
                        stack.push(n);
                    }
                }
            }
        }
    }
    out
}

fn check(data: &[bool], h: usize, w: usize) {
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, data, [h, w]).unwrap();
    for (conn, eight) in [(Connectivity::Four, false), (Connectivity::Eight, true)] {
        let got = img.regions(conn).to_vec().unwrap();
        assert_eq!(got, regions_ref(data, h, w, eight), "{h}x{w} {conn:?}");
    }
}

fn noise(h: usize, w: usize, seed: u64, density: f32) -> Vec<bool> {
    values(h * w, seed, 0.0, 1.0).into_iter().map(|v| v < density).collect()
}

#[test]
fn regions_noise() {
    for (k, &(h, w)) in [(1, 1), (1, 40), (40, 1), (7, 9), (32, 8), (33, 9), (64, 16), (100, 70), (257, 129), (31, 300)].iter().enumerate() {
        for (d, density) in [0.2f32, 0.45, 0.6, 0.8].into_iter().enumerate() {
            check(&noise(h, w, (k * 7 + d) as u64, density), h, w);
        }
    }
}

#[test]
fn regions_uniform() {
    for (h, w) in [(1, 1), (50, 50), (130, 77)] {
        check(&vec![true; h * w], h, w);
        check(&vec![false; h * w], h, w);
    }
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &vec![false; 12], [3, 4]).unwrap();
    assert_eq!(img.regions(Connectivity::Eight).to_vec().unwrap(), vec![0; 12]);
}

#[test]
fn regions_structures() {
    let (h, w) = (97, 83);
    let mut checker = vec![false; h * w];
    let mut diagonal = vec![false; h * w];
    let mut snake = vec![false; h * w];
    let mut grid = vec![false; h * w];
    let mut rings = vec![false; h * w];
    for j in 0..w {
        for i in 0..h {
            checker[i + h * j] = (i + j) % 2 == 0;
            diagonal[i + h * j] = (i + 2 * j) % 5 == 0 || i == j;
            // Rows of full lines joined alternately at the left and right ends.
            snake[i + h * j] = i % 2 == 0 || (i % 4 == 1 && j == w - 1) || (i % 4 == 3 && j == 0);
            grid[i + h * j] = i % 16 == 5 || j % 12 == 3;
            let d = ((i as f32 - 48.0).powi(2) + (j as f32 - 41.0).powi(2)).sqrt() as usize;
            rings[i + h * j] = d % 6 < 2;
        }
    }
    for data in [checker, diagonal, snake, grid, rings] {
        check(&data, h, w);
    }
    // The snake is one component.
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &(0..h * w).map(|k| (k % h) % 2 == 0 || ((k % h) % 4 == 1 && k / h == w - 1) || ((k % h) % 4 == 3 && k / h == 0)).collect::<Vec<_>>(), [h, w]).unwrap();
    let labels = img.regions(Connectivity::Four).to_vec().unwrap();
    assert_eq!(labels.iter().max(), Some(&1));
}

#[test]
fn regions_spiral() {
    // A one pixel wide spiral: a single component whose path crosses every tile border.
    let (h, w) = (120, 90);
    let mut data = vec![false; h * w];
    let (mut top, mut left, mut bottom, mut right) = (0i64, 0i64, h as i64 - 1, w as i64 - 1);
    while top <= bottom && left <= right {
        for j in left..=right {
            data[top as usize + h * j as usize] = true;
        }
        for i in top..=bottom {
            data[i as usize + h * right as usize] = true;
        }
        if top < bottom {
            for j in left..=right {
                data[bottom as usize + h * j as usize] = true;
            }
        }
        if left < right {
            for i in top + 1..=bottom {
                data[i as usize + h * left as usize] = true;
            }
        }
        top += 2;
        left += 2;
        bottom -= 2;
        right -= 2;
    }
    check(&data, h, w);
}

#[test]
fn regions_errors() {
    let ctx = ctx();
    let batch = Array::from_slice_on(&ctx, &[true; 8], [2, 2, 2]).unwrap();
    assert!(batch.regions(Connectivity::Four).to_vec().is_err());
}

#[test]
fn regions_compose() {
    // The result is lazy-shaped and feeds further operations.
    let ctx = ctx();
    let data = noise(50, 40, 3, 0.5);
    let img = Array::from_slice_on(&ctx, &data, [50, 40]).unwrap();
    let n = img.regions(Connectivity::Eight).max().item().unwrap();
    assert_eq!(n, *regions_ref(&data, 50, 40, true).iter().max().unwrap());
}

// ---- confidence connected components ----

/// ArrayFire's algorithm by the definition (see `confidence_cc`), on the host.
#[allow(clippy::too_many_arguments)]
fn confidence_ref(img: &[f32], h: usize, w: usize, seeds: &[(usize, usize)], radius: usize, mult: u32, iterations: u32) -> Vec<bool> {
    let (mn, mx) = img.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let range = mx - mn;
    let flat = !(range.abs() >= 1e-6);
    let inv = 1.0 / range;
    let j: Vec<f32> = img.iter().map(|&v| if flat { 1.0 } else { (v - mn) * inv + 1.0 }).collect();
    let seeds: Vec<(usize, usize)> = seeds.iter().copied().filter(|&(x, y)| x < h && y < w).collect();
    let m = mult as f32 as f64;
    let (mut lo, mut hi) = (1.0f64, 0.0f64);
    let mut stop = true;
    let seed_vals: Vec<f64> = seeds.iter().map(|&(x, y)| j[x + h * y] as f64).collect();
    let (smin, smax) = seed_vals.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let interval = |count: usize, s1: f64, s2: f64| {
        let n = count as f64;
        let mean = 1.5 + s1 / n;
        let var = if count > 1 { (s2 - s1 * s1 / n) / (n - 1.0) } else { 0.0 };
        let sd = var.max(0.0).sqrt();
        (smin.min(mean - m * sd), smax.max(mean + m * sd), var.abs() < 1e-6)
    };
    if !seeds.is_empty() {
        let (mut s1, mut s2, mut count) = (0.0f64, 0.0f64, 0usize);
        for &(x, y) in &seeds {
            for b in y.saturating_sub(radius)..=(y + radius).min(w - 1) {
                for a in x.saturating_sub(radius)..=(x + radius).min(h - 1) {
                    let d = j[a + h * b] as f64 - 1.5;
                    s1 += d;
                    s2 += d * d;
                    count += 1;
                }
            }
        }
        (lo, hi, stop) = interval(count, s1, s2);
    }
    let grow = |lo: f64, hi: f64| -> Vec<bool> {
        let mut seg = vec![false; h * w];
        let mut stack = Vec::new();
        for &(x, y) in &seeds {
            let q = x + h * y;
            if !seg[q] && (j[q] as f64) >= lo && (j[q] as f64) <= hi {
                seg[q] = true;
                stack.push(q);
            }
        }
        while let Some(q) = stack.pop() {
            let (i, jj) = ((q % h) as i64, (q / h) as i64);
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    let (a, b) = (i + di, jj + dj);
                    if a < 0 || b < 0 || a >= h as i64 || b >= w as i64 {
                        continue;
                    }
                    let n = a as usize + h * b as usize;
                    if !seg[n] && (j[n] as f64) >= lo && (j[n] as f64) <= hi {
                        seg[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        seg
    };
    let mut seg = grow(lo, hi);
    for _ in 0..iterations {
        if stop {
            break;
        }
        let count = seg.iter().filter(|&&b| b).count();
        if count == 0 {
            break;
        }
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for (q, _) in seg.iter().enumerate().filter(|(_, b)| **b) {
            let d = j[q] as f64 - 1.5;
            s1 += d;
            s2 += d * d;
        }
        (lo, hi, stop) = interval(count, s1, s2);
        seg = grow(lo, hi);
    }
    seg
}

/// Noise with a few bright discs.
fn blobs(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let mut img = values(h * w, seed, 0.0, 10.0);
    let c = values(12, seed + 100, 0.0, 1.0);
    for k in 0..3 {
        let (cy, cx, r, add) = (c[k * 4] * h as f32, c[k * 4 + 1] * w as f32, 2.0 + c[k * 4 + 2] * 6.0, c[k * 4 + 3] * 30.0);
        for j in 0..w {
            for i in 0..h {
                if (i as f32 - cy).powi(2) + (j as f32 - cx).powi(2) <= r * r {
                    img[i + h * j] += add;
                }
            }
        }
    }
    img
}

fn check_confidence(img: &[f32], h: usize, w: usize, seeds: &[(usize, usize)], radius: u32, mult: u32, iterations: u32) {
    let ctx = ctx();
    let a = Array::from_slice_on(&ctx, img, [h, w]).unwrap();
    let sx: Vec<u32> = seeds.iter().map(|s| s.0 as u32).collect();
    let sy: Vec<u32> = seeds.iter().map(|s| s.1 as u32).collect();
    let got = a.confidence_cc_at(&sx, &sy, radius, mult, iterations, 3.0).unwrap().to_vec().unwrap();
    let want: Vec<f32> = confidence_ref(img, h, w, seeds, radius as usize, mult, iterations).into_iter().map(|b| if b { 3.0 } else { 0.0 }).collect();
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "{h}x{w} seeds {seeds:?} r {radius} m {mult} it {iterations}: {bad} pixels differ ({} vs {} set)", got.iter().filter(|&&v| v != 0.0).count(), want.iter().filter(|&&v| v != 0.0).count());
}

#[test]
fn confidence_single_seed() {
    let mut k = 0;
    for (h, w) in [(24, 26), (40, 70), (70, 33)] {
        for radius in [0u32, 1, 3] {
            for (mult, iterations) in [(1u32, 0u32), (2, 1), (1, 3), (3, 5)] {
                k += 1;
                let img = blobs(h, w, k);
                let c = values(2, k + 7, 0.0, 1.0);
                let seed = ((c[0] * h as f32) as usize % h, (c[1] * w as f32) as usize % w);
                check_confidence(&img, h, w, &[seed], radius, mult, iterations);
            }
        }
    }
}

#[test]
fn confidence_several_seeds_and_borders() {
    let (h, w) = (45, 60);
    for k in 0..12u64 {
        let img = blobs(h, w, 50 + k);
        let c = values(8, k + 3, 0.0, 1.0);
        let seeds: Vec<(usize, usize)> = (0..1 + (k as usize % 4)).map(|s| ((c[2 * s] * h as f32) as usize % h, (c[2 * s + 1] * w as f32) as usize % w)).collect();
        check_confidence(&img, h, w, &seeds, (k % 3) as u32, 1 + (k % 3) as u32, (k % 4) as u32);
    }
    let img = blobs(h, w, 99);
    // Seeds in the corners (windows over the border) and outside the image.
    check_confidence(&img, h, w, &[(0, 0), (h - 1, w - 1)], 2, 2, 2);
    check_confidence(&img, h, w, &[(0, 0), (h + 5, 3)], 1, 3, 1);
}

#[test]
fn confidence_constant_image_is_one_segment() {
    check_confidence(&vec![4.0; 30 * 20], 30, 20, &[(3, 4)], 2, 1, 2);
    let ctx = ctx();
    let img = Array::from_slice_on(&ctx, &vec![4.0f32; 30 * 20], [30, 20]).unwrap();
    assert_eq!(img.confidence_cc_at(&[3], &[4], 1, 1, 0, 7.0).unwrap().to_vec().unwrap(), vec![7.0; 600]);
}

#[test]
fn confidence_two_levels() {
    // Left half 1, right half 9: a seed on the left segments exactly the left half, with any number of iterations.
    let (h, w) = (20, 40);
    let img: Vec<f32> = (0..h * w).map(|k| if k / h < 20 { 1.0 } else { 9.0 }).collect();
    let ctx = ctx();
    let a = Array::from_slice_on(&ctx, &img, [h, w]).unwrap();
    for iterations in [0, 1, 4] {
        let got = a.confidence_cc_at(&[10], &[5], 1, 2, iterations, 1.0).unwrap().to_vec().unwrap();
        let want: Vec<f32> = (0..h * w).map(|k| if k / h < 20 { 1.0 } else { 0.0 }).collect();
        assert_eq!(got, want, "iterations {iterations}");
    }
}

#[test]
fn confidence_seed_forms_and_errors() {
    let ctx = ctx();
    let (h, w) = (20, 30);
    let img = blobs(h, w, 5);
    let a = Array::from_slice_on(&ctx, &img, [h, w]).unwrap();
    let by_arrays = a.confidence_cc(&Array::from_slice_on(&ctx, &[3u32, 9], [2]).unwrap(), &Array::from_slice_on(&ctx, &[4u32, 20], [2]).unwrap(), 1, 2, 2, 1.0).to_vec().unwrap();
    let by_matrix = a.confidence_cc_seeds(&Array::from_slice_on(&ctx, &[3u32, 9, 4, 20], [2, 2]).unwrap(), 1, 2, 2, 1.0).to_vec().unwrap();
    let by_slices = a.confidence_cc_at(&[3, 9], &[4, 20], 1, 2, 2, 1.0).unwrap().to_vec().unwrap();
    assert_eq!(by_arrays, by_matrix);
    assert_eq!(by_arrays, by_slices);
    assert!(a.confidence_cc_at(&[3, 9], &[4], 1, 2, 2, 1.0).is_err());
    assert!(a.confidence_cc_at(&[], &[], 1, 2, 2, 1.0).is_err());
    let batch = Array::from_slice_on(&ctx, &[0.0f32; 8], [2, 2, 2]).unwrap();
    assert!(batch.confidence_cc_at(&[0], &[0], 1, 1, 1, 1.0).unwrap().to_vec().is_err());
}
