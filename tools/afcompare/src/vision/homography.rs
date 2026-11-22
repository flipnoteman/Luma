//! `homography` (RANSAC and LMedS) against ArrayFire's, on point sets with 20 % outliers.
//!
//! The random samples differ, so the matrices are compared with the transform that generated the
//! points, not with each other: both have to recover it, and find at least 90 % of the true inliers.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

const TRUTH: [f64; 9] = [1.1, 0.1, 5.0, -0.05, 0.95, -3.0, 0.0004, 0.0002, 1.0];

/// Source points, their images under `TRUTH` and every fifth one replaced by noise.
fn points(n: usize, seed: u64) -> [Vec<f32>; 4] {
    let r = uniform(4 * n, seed, 0.0, 1.0);
    let (mut xs, mut ys, mut xd, mut yd) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let (x, y) = (f64::from(r[i]) * 600.0, f64::from(r[n + i]) * 400.0);
        let w = TRUTH[6] * x + TRUTH[7] * y + TRUTH[8];
        xs.push(x as f32);
        ys.push(y as f32);
        if i % 5 == 0 {
            xd.push(r[2 * n + i] * 600.0);
            yd.push(r[3 * n + i] * 400.0);
        } else {
            xd.push(((TRUTH[0] * x + TRUTH[1] * y + TRUTH[2]) / w) as f32);
            yd.push(((TRUTH[3] * x + TRUTH[4] * y + TRUTH[5]) / w) as f32);
        }
    }
    [xs, ys, xd, yd]
}

/// The largest error of the homography (nine values in memory order, any scale) at the true
/// inliers, in pixels.
fn reprojection_error(h: &[f32], p: &[Vec<f32>; 4]) -> f64 {
    let h: Vec<f64> = h.iter().map(|&v| f64::from(v)).collect();
    let mut worst = 0.0f64;
    for i in 0..p[0].len() {
        if i % 5 == 0 {
            continue;
        }
        let (x, y) = (f64::from(p[0][i]), f64::from(p[1][i]));
        let w = h[6] * x + h[7] * y + h[8];
        let (u, v) = ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w);
        worst = worst.max(((u - f64::from(p[2][i])).powi(2) + (v - f64::from(p[3][i])).powi(2)).sqrt());
    }
    worst
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision-homography", |b| {
        let ctx = b.ctx.clone();
        for n in [1000usize, 10000] {
            let p = points(n, 411);
            let size = format!("{n} points");
            let truth_inliers = n - n.div_ceil(5);
            let up = |v: &[f32]| lumax::arr(&ctx, v, n);
            let (xs, ys, xd, yd) = (up(&p[0]), up(&p[1]), up(&p[2]), up(&p[3]));
            let aup = |v: &[f32]| afx::arr(v, &[n as u64]);
            let (axs, ays, axd, ayd) = (aup(&p[0]), aup(&p[1]), aup(&p[2]), aup(&p[3]));
            for (name, lm, am) in [("homography RANSAC", luma::HomographyMethod::Ransac, af::HomographyType::RANSAC), ("homography LMedS", luma::HomographyMethod::Lmeds, af::HomographyType::LMEDS)] {
                let (lh, li) = luma::homography(&xs, &ys, &xd, &yd, lm, 3.0, 1000);
                let (lh, li) = (lumax::host(&lh), lumax::host(&li)[0] as usize);
                let (ah, ai) = af::homography::<f32>(&axs, &ays, &axd, &ayd, am, 3.0, 1000);
                let ah = afx::host(&ah);
                let (le, ae) = (reprojection_error(&lh, &p), reprojection_error(&ah, &p));
                let good = |inl: usize| inl * 10 >= truth_inliers * 9 && inl <= n;
                let ok = le < 0.5 && ae < 0.5 && good(li) && good(ai as usize);
                let text = format!("luma {li} inliers, max error {le:.2e} px; ArrayFire {ai} inliers, max error {ae:.2e} px (true inliers {truth_inliers})");
                let v = if ok { Verdict::yes(text) } else { Verdict::no(format!("NO: {text}")) };
                b.run(
                    &format!("15 {name}"),
                    &size,
                    v,
                    "1000 iterations, inlier threshold 3; both read the inlier count back",
                    &mut || {
                        let (h, i) = luma::homography(&xs, &ys, &xd, &yd, lm, 3.0, 1000);
                        luma::eval_all(&[&h]).unwrap();
                        let _ = i.item().unwrap();
                    },
                    &mut || {
                        let (h, _) = af::homography::<f32>(&axs, &ays, &axd, &ayd, am, 3.0, 1000);
                        h.eval();
                        afx::sync();
                    },
                );
            }
        }
    });
}
