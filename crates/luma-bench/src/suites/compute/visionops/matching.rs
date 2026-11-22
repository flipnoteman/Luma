//! `match_template` at 1080p and 4K, `nearest_neighbour` and `hamming_matcher` on descriptor sets
//! of 1k to 50k features. Each kind is checked against a host computation first.

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

/// The metric of the template against the window starting at every pixel (zero past the image).
fn template_host(img: &[f32], [h, w]: [usize; 2], tpl: &[f32], [th, tw]: [usize; 2], metric: MatchType) -> Vec<f64> {
    let n = (th * tw) as f64;
    let tm = tpl.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mut out = vec![0.0; h * w];
    for oy in 0..w {
        for ox in 0..h {
            let at = |a: usize, b: usize| if ox + a < h && oy + b < w { f64::from(img[ox + a + h * (oy + b)]) } else { 0.0 };
            let wm = (0..tw).flat_map(|b| (0..th).map(move |a| (a, b))).map(|(a, b)| at(a, b)).sum::<f64>() / n;
            let mut acc = 0.0;
            for b in 0..tw {
                for a in 0..th {
                    let (v, t) = (at(a, b), f64::from(tpl[a + th * b]));
                    acc += match metric {
                        MatchType::Sad => (v - t).abs(),
                        MatchType::Zsad => (v - wm - t + tm).abs(),
                        MatchType::Lsad => (v - wm / tm * t).abs(),
                        MatchType::Ssd => (v - t).powi(2),
                        MatchType::Zssd => (v - wm - t + tm).powi(2),
                        MatchType::Lssd => (v - wm / tm * t).powi(2),
                        _ => unreachable!("not timed"),
                    };
                }
            }
            out[ox + h * oy] = acc;
        }
    }
    out
}

/// The `k` nearest of `train` (features of `len` elements, feature-major) to every query by
/// (distance, index): `[k, nq]` indices and distances.
fn neighbours_host(q: &[f32], t: &[f32], len: usize, k: usize, hamming: bool) -> (Vec<u32>, Vec<f64>) {
    let (nq, nt) = (q.len() / len, t.len() / len);
    let (mut idx, mut dist) = (Vec::new(), Vec::new());
    for i in 0..nq {
        let mut d: Vec<(f64, u32)> = (0..nt)
            .map(|j| {
                let sum: f64 = (0..len)
                    .map(|l| {
                        let (a, b) = (q[i * len + l], t[j * len + l]);
                        if hamming {
                            f64::from((a.to_bits() ^ b.to_bits()).count_ones())
                        } else {
                            f64::from(a - b).powi(2)
                        }
                    })
                    .sum();
                (sum, j as u32)
            })
            .collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &(dd, j) in &d[..k] {
            idx.push(j);
            dist.push(dd);
        }
    }
    (idx, dist)
}

/// Descriptors as `u32` words of random bits.
fn words(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 32) as u32
        })
        .collect()
}

pub fn run(h: &mut Harness) {
    h.guarded("matching", |h| {
        // Small checks against the host.
        let small = [37usize, 29];
        let sdata = values(small[0] * small[1], 31, 0.0, 255.0).iter().map(|v| v.floor()).collect::<Vec<_>>();
        let simg = upload(h, &sdata, small);
        let stpl = values(7 * 5, 32, 1.0, 255.0).iter().map(|v| v.floor()).collect::<Vec<_>>();
        let template = upload(h, &stpl, [7, 5]);
        for (name, metric) in [("sad", MatchType::Sad), ("zsad", MatchType::Zsad), ("lsad", MatchType::Lsad), ("ssd", MatchType::Ssd), ("zssd", MatchType::Zssd), ("lssd", MatchType::Lssd)] {
            check64(&format!("matching/match_template_{name}"), &host(&simg.match_template(&template, metric)), &template_host(&sdata, small, &stpl, [7, 5], metric), 1e-4, 1e-3);
        }
        {
            let (nq, nt, len) = (45usize, 130usize, 20usize);
            let (q, t) = (values(nq * len, 41, 0.0, 8.0).iter().map(|v| v.floor()).collect::<Vec<_>>(), values(nt * len, 42, 0.0, 8.0).iter().map(|v| v.floor()).collect::<Vec<_>>());
            let (qa, ta) = (upload(h, &q, [len, nq]), upload(h, &t, [len, nt]));
            let (idx, dist) = nearest_neighbour(&qa, &ta, 0, 3, DistType::Ssd).unwrap();
            let (wi, wd) = neighbours_host(&q, &t, len, 3, false);
            assert_eq!(idx.to_vec().unwrap(), wi, "matching/nearest_neighbour indices");
            check64("matching/nearest_neighbour", &host(&dist), &wd, 1e-5, 1e-5);
            let (qw, tw) = (words(nq * 8, 43), words(nt * 8, 44));
            let to_f = |w: &[u32]| w.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>();
            let (wi, wd) = neighbours_host(&to_f(&qw), &to_f(&tw), 8, 2, true);
            let (qa, ta) = (Array::<u32>::from_slice_on(&h.ctx, &qw, [8, nq]).unwrap(), Array::<u32>::from_slice_on(&h.ctx, &tw, [8, nt]).unwrap());
            let (idx, dist) = hamming_matcher(&qa, &ta, 0, 2).unwrap();
            assert_eq!(idx.to_vec().unwrap(), wi, "matching/hamming_matcher indices");
            let got: Vec<f32> = dist.to_vec().unwrap().iter().map(|&d| d as f32).collect();
            check64("matching/hamming_matcher", &got, &wd, 0.0, 0.0);
        }

        for dims in sizes(h) {
            let data = values(dims[0] * dims[1], 7, 0.0, 255.0);
            let img = upload(h, &data, dims);
            let tag = format!("{}x{}", dims[1], dims[0]);
            for k in [3usize, 7, 16, 32] {
                let template = upload(h, &values(k * k, 8, 0.0, 255.0), [k, k]);
                let work = Work::Items((dims[0] * dims[1]) as f64, "Mpix/s");
                for (name, metric) in [("sad", MatchType::Sad), ("ssd", MatchType::Ssd)] {
                    h.case(&format!("matching/match_template_{name}_{k}x{k}"), &tag, work, &mut || img.match_template(&template, metric).eval().unwrap());
                }
                if k == 16 {
                    for (name, metric) in [("zsad", MatchType::Zsad), ("lssd", MatchType::Lssd), ("zncc", MatchType::Zncc)] {
                        h.case(&format!("matching/match_template_{name}_{k}x{k}"), &tag, work, &mut || img.match_template(&template, metric).eval().unwrap());
                    }
                }
            }
        }

        // Descriptor sets.
        let counts: Vec<usize> = if h.ctx.is_cpu() { vec![300] } else { pick(h, vec![1000, 10_000], vec![1000, 10_000, 50_000]) };
        for &n in &counts {
            for len in [64usize, 128] {
                let (q, t) = (values(n * len, 51, 0.0, 1.0), values(n * len, 52, 0.0, 1.0));
                let (qa, ta) = (upload(h, &q, [len, n]), upload(h, &t, [len, n]));
                let flops = Work::Flops(2.0 * (n as f64) * (n as f64) * len as f64);
                for k in [1usize, 2, 8, 16, 64] {
                    if k > 8 && (len != 64 || n > 10_000) {
                        continue;
                    }
                    h.case(&format!("matching/nearest_neighbour_ssd_k{k}_len{len}"), &format!("{n}x{n}"), flops, &mut || {
                        let (i, d) = nearest_neighbour(&qa, &ta, 0, k, DistType::Ssd).unwrap();
                        eval_all(&[&d]).unwrap();
                        let _ = i;
                    });
                }
                h.case(&format!("matching/nearest_neighbour_sad_k1_len{len}"), &format!("{n}x{n}"), flops, &mut || {
                    nearest_neighbour(&qa, &ta, 0, 1, DistType::Sad).unwrap().1.eval().unwrap();
                });
            }
            let (q, t) = (words(n * 8, 53), words(n * 8, 54));
            let (qa, ta) = (Array::<u32>::from_slice_on(&h.ctx, &q, [8, n]).unwrap(), Array::<u32>::from_slice_on(&h.ctx, &t, [8, n]).unwrap());
            qa.eval().unwrap();
            ta.eval().unwrap();
            let pairs = Work::Items((n as f64) * (n as f64), "Mpair/s");
            for k in [1usize, 2] {
                h.case(&format!("matching/hamming_matcher_256bit_k{k}"), &format!("{n}x{n}"), pairs, &mut || {
                    hamming_matcher(&qa, &ta, 0, k).unwrap().1.eval().unwrap();
                });
            }
        }
    });
}
