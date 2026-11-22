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
