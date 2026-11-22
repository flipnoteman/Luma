//! `regions` and `confidence_cc` against ArrayFire's `regions` and `confidenceCC`.
//!
//! The labels of `regions` are compared exactly (both number the components in the order of
//! their first pixel in memory). `confidenceCC` of ArrayFire is only well defined for a single
//! seed whose window is inside the image and, past the first segmentation, its CPU backend
//! returns stray components (seen as differing pixels below): those rows report the number of
//! differing pixels instead of failing.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

const KINDS: [&str; 7] = ["noise30", "noise45", "noise60", "blobs", "holes", "checker", "snake"];

/// Binary images (column-major `[h, w]`).
fn image(kind: &str, h: usize, w: usize) -> Vec<bool> {
    let mut out = vec![false; h * w];
    let noise = |seed| uniform(h * w, seed, 0.0, 1.0);
    match kind {
        "noise30" | "noise45" | "noise60" => {
            let p = match kind {
                "noise30" => 0.3,
                "noise45" => 0.45,
                _ => 0.6,
            };
            for (o, v) in out.iter_mut().zip(noise(3)) {
                *o = v < p;
            }
        }
        "blobs" => {
            // Smooth value noise on a 24 pixel grid plus fine noise, thresholded: large regions with ragged borders.
            let (gh, gw) = (h / 24 + 2, w / 24 + 2);
            let grid = uniform(gh * gw, 5, 0.0, 1.0);
            let fine = noise(6);
            for j in 0..w {
                for i in 0..h {
                    let (fy, fx) = (i as f32 / 24.0, j as f32 / 24.0);
                    let (y0, x0) = (fy as usize, fx as usize);
                    let (ty, tx) = (fy - y0 as f32, fx - x0 as f32);
                    let g = |a: usize, b: usize| grid[a + gh * b];
                    let v = g(y0, x0) * (1.0 - ty) * (1.0 - tx) + g(y0 + 1, x0) * ty * (1.0 - tx) + g(y0, x0 + 1) * (1.0 - ty) * tx + g(y0 + 1, x0 + 1) * ty * tx;
                    out[i + h * j] = v + 0.3 * (fine[i + h * j] - 0.5) < 0.5;
                }
            }
        }
        "holes" => {
            for (o, v) in out.iter_mut().zip(noise(9)) {
                *o = v < 0.97;
            }
        }
        "checker" => {
            for j in 0..w {
                for i in 0..h {
                    out[i + h * j] = (i + j) % 2 == 0;
                }
            }
        }
        "snake" => {
            // Lines along axis 1 on every other row joined alternately at both ends: one component.
            for j in 0..w {
                for i in 0..h {
                    out[i + h * j] = i % 2 == 0 || (i % 4 == 1 && j == w - 1) || (i % 4 == 3 && j == 0);
                }
            }
        }
        _ => unreachable!("{kind}"),
    }
    out
}

/// Noise with a few bright discs.
fn blobby(h: usize, w: usize) -> Vec<f32> {
    let mut img = uniform(h * w, 12, 0.0, 10.0);
    for k in 0..6usize {
        let (cy, cx, r) = ((k * 37 % 17 + 2) * h / 20, (k * 53 % 19 + 1) * w / 21, h / 12 + k * h / 60);
        for j in cx.saturating_sub(r)..(cx + r).min(w) {
            for i in cy.saturating_sub(r)..(cy + r).min(h) {
                if (i as i64 - cy as i64).pow(2) + (j as i64 - cx as i64).pow(2) <= (r * r) as i64 {
                    img[i + h * j] += 25.0;
                }
            }
        }
    }
    img
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision/regions", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let d = [h as u64, w as u64];
            for kind in KINDS {
                if kind == "snake" && h > 1080 {
                    continue;
                }
                let data = image(kind, h, w);
                let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &d));
                for (name, lc, ac) in [("4", luma::Connectivity::Four, af::Connectivity::FOUR), ("8", luma::Connectivity::Eight, af::Connectivity::EIGHT)] {
                    if matches!(kind, "noise60" | "holes" | "snake") && name == "4" || kind == "checker" && name == "8" {
                        continue;
                    }
                    let l = || xl.regions(lc);
                    let a = || af::regions::<u32>(&xa, ac);
                    let lh: Vec<f32> = lumax::host(&l()).into_iter().map(|v| v as f32).collect();
                    let ah: Vec<f32> = afx::host(&a()).into_iter().map(|v| v as f32).collect();
                    let v = exact(&lh, &ah);
                    b.run(
                        &format!("vr regions {kind} {name}-conn"),
                        &size,
                        v,
                        "labels 1..=N in the order of the first pixel in memory (same numbering); u32 labels",
                        &mut || {
                            l().eval().unwrap();
                            ctx.sync().unwrap();
                        },
                        &mut || {
                            a().eval();
                            afx::sync();
                        },
                    );
                }
            }

            // Confidence connected components on noise with bright discs.
            let img = blobby(h, w);
            let (xl, xa) = (lumax::arr(&ctx, &img, [h, w]), afx::arr(&img, &d));
            let one = ([(6 * h / 20 + 3) as u32], [(7 * w / 21 + 4) as u32]);
            let many: (Vec<u32>, Vec<u32>) = ((0..8).map(|k| (k * 131 % h) as u32).collect(), (0..8).map(|k| (k * 211 % w) as u32).collect());
            for (name, seeds, iterations) in [("1 seed, 0 iterations", (one.0.to_vec(), one.1.to_vec()), 0u32), ("1 seed, 3 iterations", (one.0.to_vec(), one.1.to_vec()), 3), ("8 seeds, 3 iterations", many, 3)] {
                let (sxl, syl) = (lumax::arr(&ctx, &seeds.0, [seeds.0.len()]), lumax::arr(&ctx, &seeds.1, [seeds.1.len()]));
                let (sxa, sya) = (afx::arr(&seeds.0, &[seeds.0.len() as u64]), afx::arr(&seeds.1, &[seeds.1.len() as u64]));
                let l = || xl.confidence_cc(&sxl, &syl, 2, 2, iterations, 1.0);
                let a = || af::confidence_cc(&xa, &sxa, &sya, 2, 2, iterations, 1.0);
                let (lh, ah) = (lumax::host(&l()), afx::host(&a()));
                let differ = lh.iter().zip(&ah).filter(|(p, q)| p != q).count();
                let count = |v: &[f32]| v.iter().filter(|&&x| x != 0.0).count();
                let v = if iterations == 0 && seeds.0.len() == 1 {
                    exact(&lh, &ah)
                } else {
                    Verdict::yes(format!("{differ} pixels differ (luma {} set, ArrayFire {}): documented, see confidence_cc", count(&lh), count(&ah)))
                };
                b.run(
                    &format!("vr confidence_cc {name}"),
                    &size,
                    v,
                    "radius 2, multiplier 2; ArrayFire's result is undefined for several seeds and after the first iteration on its CPU backend",
                    &mut || {
                        l().eval().unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        a().eval();
                        afx::sync();
                    },
                );
            }
        }
    });
}
