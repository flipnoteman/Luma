//! Template matching, nearest-neighbour and Hamming matchers: tests against host references.

use luma::{Array, MatchType};

/// A deterministic pseudo-random image of small integers (exact in f32).
fn pattern(n: usize, seed: u64, modulus: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) % modulus) as f32
        })
        .collect()
}

/// The metric of the template against the window starting at `(ox, oy)` of every plane, in f64.
fn template_reference(img: &[f32], dims: [usize; 3], tpl: &[f32], t: [usize; 2], metric: MatchType) -> Vec<f64> {
    let [h, w, planes] = dims;
    let n = (t[0] * t[1]) as f64;
    let tm = tpl.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let t2: f64 = tpl.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let tz2: f64 = tpl.iter().map(|&v| (f64::from(v) - tm).powi(2)).sum();
    let mut out = vec![0.0; h * w * planes];
    for p in 0..planes {
        for oy in 0..w {
            for ox in 0..h {
                let at = |a: usize, b: usize| if ox + a < h && oy + b < w { f64::from(img[p * h * w + ox + a + h * (oy + b)]) } else { 0.0 };
                let mut wm = 0.0;
                for b in 0..t[1] {
                    for a in 0..t[0] {
                        wm += at(a, b);
                    }
                }
                wm /= n;
                let (mut acc, mut sq) = (0.0, 0.0);
                for b in 0..t[1] {
                    for a in 0..t[0] {
                        let (v, tv) = (at(a, b), f64::from(tpl[a + t[0] * b]));
                        match metric {
                            MatchType::Sad => acc += (v - tv).abs(),
                            MatchType::Zsad => acc += (v - wm - tv + tm).abs(),
                            MatchType::Lsad => acc += (v - wm / tm * tv).abs(),
                            MatchType::Ssd => acc += (v - tv).powi(2),
                            MatchType::Zssd => acc += (v - wm - tv + tm).powi(2),
                            MatchType::Lssd => acc += (v - wm / tm * tv).powi(2),
                            MatchType::Ncc => {
                                acc += v * tv;
                                sq += v * v;
                            }
                            MatchType::Zncc => {
                                acc += (v - wm) * (tv - tm);
                                sq += (v - wm).powi(2);
                            }
                        }
                    }
                }
                out[p * h * w + ox + h * oy] = match metric {
                    MatchType::Ncc => {
                        let den = (sq * t2).sqrt();
                        if den > 0.0 { acc / den } else { 0.0 }
                    }
                    MatchType::Zncc => {
                        let den = (sq * tz2).sqrt();
                        if den > 0.0 { acc / den } else { 0.0 }
                    }
                    _ => acc,
                };
            }
        }
    }
    out
}

fn close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(g) - w).abs() <= 2e-4 * (1.0 + w.abs()), "{what}: element {i}: got {g}, want {w}");
    }
}

const ALL: [MatchType; 8] = [MatchType::Sad, MatchType::Zsad, MatchType::Lsad, MatchType::Ssd, MatchType::Zssd, MatchType::Lssd, MatchType::Ncc, MatchType::Zncc];

fn check_template(h: usize, w: usize, planes: usize, th: usize, tw: usize) {
    let img = pattern(h * w * planes, 1, 10);
    let tpl = pattern(th * tw, 2, 9).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let image = Array::from_slice(&img, [h, w, planes]).unwrap();
    let template = Array::from_slice(&tpl, [th, tw]).unwrap();
    for metric in ALL {
        let got = image.match_template(&template, metric).to_vec().unwrap();
        let want = template_reference(&img, [h, w, planes], &tpl, [th, tw], metric);
        close(&got, &want, &format!("{h}x{w}x{planes} template {th}x{tw} {metric:?}"));
    }
}

#[test]
fn template_small() {
    check_template(5, 4, 1, 2, 3);
    check_template(17, 13, 1, 3, 3);
    check_template(1, 1, 1, 1, 1);
}

#[test]
fn template_tile_edges() {
    // Past one tile along both axes, template reaching beyond the image, batches.
    check_template(70, 45, 1, 7, 5);
    check_template(131, 37, 2, 4, 9);
    check_template(9, 9, 1, 12, 11);
}

#[test]
fn template_slabbed() {
    // Templates too large for the workgroup memory of the smallest tiles are staged in slabs.
    check_template(40, 40, 1, 40, 33);
    check_template(100, 70, 1, 32, 32);
    check_template(30, 30, 1, 50, 40);
}

#[test]
fn template_batch_axes() {
    let (h, w) = (20, 18);
    let img = pattern(h * w * 6, 3, 10);
    let tpl = pattern(12, 4, 9);
    let image = Array::from_slice(&img, [h, w, 3, 2]).unwrap();
    let template = Array::from_slice(&tpl, [3, 4]).unwrap();
    let got = image.match_template(&template, MatchType::Ssd).to_vec().unwrap();
    close(&got, &template_reference(&img, [h, w, 6], &tpl, [3, 4], MatchType::Ssd), "4-D batch");
}

#[test]
fn template_errors() {
    let image = Array::from_slice(&pattern(16, 1, 10), [4, 4]).unwrap();
    let batch = Array::from_slice(&pattern(16, 1, 10), [2, 2, 2, 2]).unwrap();
    assert!(image.match_template(&batch, MatchType::Sad).to_vec().is_err());
}

// ---- nearest neighbours ----

use luma::{hamming_matcher, nearest_neighbour, DistType};

/// Features as columns of `[len, n]` (`dist_dim` 0) or rows of `[n, len]` (1), from `data` in the
/// order feature-major, as the host keeps them: `feature(i)[l] = data[i * len + l]`.
fn layout<T: Copy>(data: &[T], len: usize, n: usize, dist_dim: usize) -> Vec<T> {
    if dist_dim == 0 {
        data.to_vec()
    } else {
        (0..n * len).map(|p| data[(p % n) * len + p / n]).collect()
    }
}

/// The reference: distances of every query to every train feature, the `k` smallest by (distance,
/// index), as `[k, nq]` column-major indices and distances.
fn neighbours_reference(q: &[i64], t: &[i64], len: usize, k: usize, metric: DistType) -> (Vec<u32>, Vec<i64>) {
    let (nq, nt) = (q.len() / len, t.len() / len);
    let (mut idx, mut dist) = (Vec::new(), Vec::new());
    for i in 0..nq {
        let mut d: Vec<(i64, u32)> = (0..nt)
            .map(|j| {
                let sum: i64 = (0..len)
                    .map(|l| {
                        let (a, b) = (q[i * len + l], t[j * len + l]);
                        match metric {
                            DistType::Sad => (a - b).abs(),
                            DistType::Ssd => (a - b) * (a - b),
                            DistType::Shd => i64::from(((a as u32) ^ (b as u32)).count_ones()),
                        }
                    })
                    .sum();
                (sum, j as u32)
            })
            .collect();
        d.sort();
        for &(dd, j) in &d[..k] {
            idx.push(j);
            dist.push(dd);
        }
    }
    (idx, dist)
}

fn ints(n: usize, seed: u64, modulus: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 24) % modulus) as i64
        })
        .collect()
}

fn check_f32(nq: usize, nt: usize, len: usize, k: usize, metric: DistType, dist_dim: usize, modulus: u64) {
    let (q, t) = (ints(nq * len, 5, modulus), ints(nt * len, 6, modulus));
    let (wi, wd) = neighbours_reference(&q, &t, len, k, metric);
    let to_f32 = |v: &[i64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let shape = |n: usize| if dist_dim == 0 { [len, n] } else { [n, len] };
    let qa = Array::from_slice(&layout(&to_f32(&q), len, nq, dist_dim), shape(nq)).unwrap();
    let ta = Array::from_slice(&layout(&to_f32(&t), len, nt, dist_dim), shape(nt)).unwrap();
    let (gi, gd) = nearest_neighbour(&qa, &ta, dist_dim, k, metric).unwrap();
    assert_eq!(gi.shape().dims(), [k, nq, 1, 1]);
    let what = format!("{nq}x{nt} len {len} k {k} {metric:?} dim {dist_dim}");
    assert_eq!(gi.to_vec().unwrap(), wi, "{what}: indices");
    assert_eq!(gd.to_vec().unwrap(), wd.iter().map(|&d| d as f32).collect::<Vec<_>>(), "{what}: distances");
}

#[test]
fn neighbours_f32() {
    for metric in [DistType::Sad, DistType::Ssd] {
        for dist_dim in [0, 1] {
            check_f32(3, 5, 4, 2, metric, dist_dim, 1000);
            check_f32(70, 150, 20, 1, metric, dist_dim, 1000);
            check_f32(70, 150, 20, 5, metric, dist_dim, 7); // many ties
            check_f32(10, 300, 37, 16, metric, dist_dim, 100);
            check_f32(130, 70, 5, 70, metric, dist_dim, 3);
        }
    }
}

#[test]
fn neighbours_wide_k() {
    check_f32(5, 600, 8, 256, DistType::Ssd, 0, 50);
    check_f32(5, 600, 8, 129, DistType::Sad, 1, 50);
    check_f32(3, 300, 4, 300.min(256), DistType::Ssd, 0, 5);
}

#[test]
fn neighbours_u32_i32() {
    let (nq, nt, len, k) = (40, 100, 9, 4);
    let (q, t) = (ints(nq * len, 7, 5000), ints(nt * len, 8, 5000));
    for metric in [DistType::Sad, DistType::Ssd, DistType::Shd] {
        let (wi, wd) = neighbours_reference(&q, &t, len, k, metric);
        let to_u32 = |v: &[i64]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
        let qa = Array::from_slice(&to_u32(&q), [len, nq]).unwrap();
        let ta = Array::from_slice(&to_u32(&t), [len, nt]).unwrap();
        let (gi, gd) = nearest_neighbour(&qa, &ta, 0, k, metric).unwrap();
        assert_eq!(gi.to_vec().unwrap(), wi, "u32 {metric:?}");
        assert_eq!(gd.to_vec().unwrap(), wd.iter().map(|&d| d as u32).collect::<Vec<_>>(), "u32 {metric:?}");
    }
    // Signed features: differences across zero.
    let centred = |v: &[i64]| v.iter().map(|&x| x - 2500).collect::<Vec<_>>();
    let (qs, ts) = (centred(&q), centred(&t));
    for metric in [DistType::Sad, DistType::Ssd] {
        let (wi, wd) = neighbours_reference(&qs, &ts, len, k, metric);
        let to_i32 = |v: &[i64]| v.iter().map(|&x| x as i32).collect::<Vec<_>>();
        let qa = Array::from_slice(&to_i32(&qs), [len, nq]).unwrap();
        let ta = Array::from_slice(&to_i32(&ts), [len, nt]).unwrap();
        let (gi, gd) = nearest_neighbour(&qa, &ta, 0, k, metric).unwrap();
        assert_eq!(gi.to_vec().unwrap(), wi, "i32 {metric:?}");
        assert_eq!(gd.to_vec().unwrap(), wd.iter().map(|&d| d as i32).collect::<Vec<_>>(), "i32 {metric:?}");
    }
}

#[test]
fn neighbours_errors() {
    let f = Array::from_slice(&[0.0f32; 12], [4, 3]).unwrap();
    let g = Array::from_slice(&[0.0f32; 10], [5, 2]).unwrap();
    let u = Array::from_slice(&[0u32; 12], [4, 3]).unwrap();
    assert!(nearest_neighbour(&f, &g, 0, 1, DistType::Ssd).is_err(), "feature lengths differ");
    assert!(nearest_neighbour(&f, &f, 2, 1, DistType::Ssd).is_err(), "dist_dim");
    assert!(nearest_neighbour(&f, &f, 0, 0, DistType::Ssd).is_err(), "n_dist 0");
    assert!(nearest_neighbour(&f, &f, 0, 4, DistType::Ssd).is_err(), "n_dist above the train features");
    assert!(nearest_neighbour(&f, &f, 0, 1, DistType::Shd).is_err(), "hamming of floats");
    assert!(nearest_neighbour(&u, &u, 0, 1, DistType::Shd).is_ok());
}

#[test]
fn hamming() {
    // 256-bit descriptors (8 words), both layouts, ties and several chunks of queries.
    let (nq, nt, len, k) = (150, 333, 8, 3);
    let (q, t) = (ints(nq * len, 9, 1 << 32), ints(nt * len, 10, 1 << 32));
    let (wi, wd) = neighbours_reference(&q, &t, len, k, DistType::Shd);
    for dist_dim in [0, 1] {
        let to_u32 = |v: &[i64]| layout(&v.iter().map(|&x| x as u32).collect::<Vec<_>>(), len, v.len() / len, dist_dim);
        let shape = |n: usize| if dist_dim == 0 { [len, n] } else { [n, len] };
        let qa = Array::from_slice(&to_u32(&q), shape(nq)).unwrap();
        let ta = Array::from_slice(&to_u32(&t), shape(nt)).unwrap();
        let (gi, gd) = hamming_matcher(&qa, &ta, dist_dim, k).unwrap();
        assert_eq!(gi.to_vec().unwrap(), wi, "dim {dist_dim}");
        assert_eq!(gd.to_vec().unwrap(), wd.iter().map(|&d| d as u32).collect::<Vec<_>>(), "dim {dist_dim}");
    }
}
