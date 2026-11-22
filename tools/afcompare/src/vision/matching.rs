//! `match_template`, `nearest_neighbour` and `hamming_matcher` against ArrayFire's `matchTemplate`,
//! `nearestNeighbour` and `hammingMatcher`: the windows start at the pixel (zero past the image),
//! the nearest neighbours of ArrayFire's CUDA backend break ties by the smaller train index as
//! luma does, so indices and distances are compared exactly.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::attempt;
use crate::lumax;
use crate::util::*;

fn as_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Descriptors of `len` random 32-bit words.
fn bits(n: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| (r.next_u64() >> 32) as u32).collect()
}

/// The `[len, n]` matrix of columns as `[n, len]` (the features as rows).
fn transposed(data: &[f32], len: usize, n: usize) -> Vec<f32> {
    (0..n * len).map(|p| data[(p % n) * len + p / n]).collect()
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision/matching", |b| {
        let ctx = b.ctx.clone();
        let sync = || {
            ctx.sync().unwrap();
        };
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let data = uniform(h * w, 601, 0.0, 1.0);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &[h as u64, w as u64]));
            for k in [3usize, 7, 16, 32] {
                let td = uniform(k * k, 602, 0.0, 1.0);
                let (tl, ta) = (lumax::arr(&ctx, &td, [k, k]), afx::arr(&td, &[k as u64, k as u64]));
                for (name, lm, am) in [("sad", luma::MatchType::Sad, af::MatchType::SAD), ("ssd", luma::MatchType::Ssd, af::MatchType::SSD), ("zsad", luma::MatchType::Zsad, af::MatchType::ZSAD), ("lssd", luma::MatchType::Lssd, af::MatchType::LSSD)] {
                    if (name == "zsad" || name == "lssd") && k != 16 {
                        continue;
                    }
                    let v = compare(&lumax::host(&xl.match_template(&tl, lm)), &afx::host(&af::match_template(&xa, &ta, am)), 2e-4, 1e-3);
                    let mut l = || {
                        xl.match_template(&tl, lm).eval().unwrap();
                        sync();
                    };
                    let mut a = || {
                        af::match_template(&xa, &ta, am).eval();
                        afx::sync();
                    };
                    b.run(&format!("43 match_template {name} {k}x{k}"), &size, v, "window starts at the pixel, zero past the image", &mut l, &mut a);
                }
            }
        }

        // Descriptor sets of f32 features (a feature is a column) and of 256-bit words.
        let counts: &[usize] = if b.quick { &[1000, 10_000] } else { &[1000, 10_000, 50_000] };
        for &n in counts {
            for len in [64usize, 128] {
                let (qd, td) = (uniform(n * len, 611, 0.0, 1.0), uniform(n * len, 612, 0.0, 1.0));
                for dist_dim in [0usize, 1] {
                    // `dist_dim` 1: the features are the rows of `[n, len]`.
                    let dims = if dist_dim == 0 { [len as u64, n as u64] } else { [n as u64, len as u64] };
                    let (qd, td) = if dist_dim == 0 { (qd.clone(), td.clone()) } else { (transposed(&qd, len, n), transposed(&td, len, n)) };
                    if dist_dim == 1 && (len != 64 || n != 10_000) {
                        continue;
                    }
                    let (ql, tl) = (lumax::arr(&ctx, &qd, [dims[0] as usize, dims[1] as usize]), lumax::arr(&ctx, &td, [dims[0] as usize, dims[1] as usize]));
                    let (qa, ta) = (afx::arr(&qd, &dims), afx::arr(&td, &dims));
                    for k in [1usize, 2, 8, 16] {
                        if (n > 10_000 && k > 1) || (dist_dim == 1 && k > 1) || (k > 2 && len != 64) {
                            continue;
                        }
                        let size = format!("{n}x{n} len {len}");
                        let name = format!("44 nearest_neighbour ssd k={k} dim {dist_dim}");
                        let lf = || luma::nearest_neighbour(&ql, &tl, dist_dim, k, luma::DistType::Ssd).unwrap();
                        let mut l = || {
                            let (i, d) = lf();
                            i.eval().unwrap();
                            d.eval().unwrap();
                            sync();
                        };
                        let Some(_) = attempt(|| af::nearest_neighbour(&qa, &ta, dist_dim as i64, k as u32, af::MatchType::SSD)) else {
                            b.run_one(&name, &size, Some(&mut l), None, "ArrayFire fails (memory)");
                            continue;
                        };
                        let (li, ld) = lf();
                        let (ai, ad) = af::nearest_neighbour(&qa, &ta, dist_dim as i64, k as u32, af::MatchType::SSD);
                        let v = compare(&lumax::host(&ld), &afx::host(&ad), 1e-4, 1e-4).and(exact(&as_f32(&lumax::host(&li)), &as_f32(&afx::host(&ai))));
                        let mut a = || {
                            let r = af::nearest_neighbour(&qa, &ta, dist_dim as i64, k as u32, af::MatchType::SSD);
                            r.1.eval();
                            afx::sync();
                        };
                        b.run(&name, &size, v, "ties: smaller train index first (as ArrayFire CUDA)", &mut l, &mut a);
                    }
                }
            }
            // 256-bit descriptors, the Hamming matcher.
            let (qd, td) = (bits(n * 8, 621), bits(n * 8, 622));
            let (ql, tl) = (lumax::arr(&ctx, &qd, [8, n]), lumax::arr(&ctx, &td, [8, n]));
            let (qa, ta) = (afx::arr(&qd, &[8, n as u64]), afx::arr(&td, &[8, n as u64]));
            for k in [1usize, 2] {
                let size = format!("{n}x{n} 256 bit");
                let name = format!("45 hamming_matcher k={k}");
                let lf = || luma::hamming_matcher(&ql, &tl, 0, k).unwrap();
                let mut l = || {
                    let (i, d) = lf();
                    i.eval().unwrap();
                    d.eval().unwrap();
                    sync();
                };
                let Some(_) = attempt(|| af::hamming_matcher(&qa, &ta, 0, k as u32)) else {
                    b.run_one(&name, &size, Some(&mut l), None, "ArrayFire fails (memory)");
                    continue;
                };
                let (li, ld) = lf();
                let (ai, ad) = af::hamming_matcher(&qa, &ta, 0, k as u32);
                let v = exact(&as_f32(&lumax::host(&ld)), &as_f32(&afx::host(&ad))).and(exact(&as_f32(&lumax::host(&li)), &as_f32(&afx::host(&ai))));
                let mut a = || {
                    let r = af::hamming_matcher(&qa, &ta, 0, k as u32);
                    r.1.eval();
                    afx::sync();
                };
                b.run(&name, &size, v, "ties: smaller train index first (as ArrayFire CUDA)", &mut l, &mut a);
            }
        }
    });
}
