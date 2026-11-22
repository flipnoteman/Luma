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
