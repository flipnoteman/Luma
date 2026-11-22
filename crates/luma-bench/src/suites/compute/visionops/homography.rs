//! `homography` (RANSAC, LMedS) on point sets with 20 % outliers: the transform that made the
//! points must be recovered before the case is timed.

use luma::*;

use super::super::upload;
use crate::harness::*;

const TRUTH: [f64; 9] = [1.1, 0.1, 5.0, -0.05, 0.95, -3.0, 0.0004, 0.0002, 1.0];

pub fn run(h: &mut Harness) {
    h.guarded("visionops/homography", |h| {
        let counts: &[usize] = if h.ctx.is_cpu() { &[500] } else { &[1000, 10000] };
        for &n in counts {
            let r = values(4 * n, 41, 0.0, 1.0);
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
            let (xs, ys, xd, yd) = (upload(h, &xs, n), upload(h, &ys, n), upload(h, &xd, n), upload(h, &yd, n));
            for (name, method) in [("ransac", HomographyMethod::Ransac), ("lmeds", HomographyMethod::Lmeds)] {
                let (m, inliers) = homography(&xs, &ys, &xd, &yd, method, 3.0, 1000);
                let m = m.to_vec().unwrap();
                for k in 0..9 {
                    let tol = if k >= 6 { 1e-4 } else { 2e-2 * TRUTH[k].abs().max(1.0) };
                    assert!((f64::from(m[k]) - TRUTH[k]).abs() <= tol, "visionops/homography_{name}: h[{k}] = {} want {}", m[k], TRUTH[k]);
                }
                assert!(inliers.item().unwrap() as usize >= n * 4 / 5 * 9 / 10, "visionops/homography_{name}: inliers");
                h.case(&format!("visionops/homography_{name}"), &format!("{n} points"), Work::Items(1000.0 * n as f64, "Mpair-iter/s"), &mut || {
                    let (m, i) = homography(&xs, &ys, &xd, &yd, method, 3.0, 1000);
                    m.eval().unwrap();
                    let _ = i.item().unwrap();
                });
            }
        }
    });
}
