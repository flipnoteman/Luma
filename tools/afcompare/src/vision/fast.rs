//! `fast` against ArrayFire's `fast`, at 1080p and 4K on a textured image.
//!
//! The point lists are compared as sets of `(x, y)` with their scores (the order of the points
//! differs by design, see `Array::fast`): the verdict says how many of ArrayFire's points luma
//! also found with the same score and how many luma found that ArrayFire did not.

use std::collections::HashMap;

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

/// Value noise on a grid of 8 pixels plus per-pixel noise, 0..255: an image with corners everywhere.
pub fn textured(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let cell = 8usize;
    let (gh, gw) = (h / cell + 2, w / cell + 2);
    let grid = uniform(gh * gw, seed, 0.0, 255.0);
    let noise = uniform(h * w, seed + 1, -10.0, 10.0);
    let mut out = vec![0.0f32; h * w];
    for j in 0..w {
        let (gj, fj) = (j / cell, (j % cell) as f32 / cell as f32);
        for i in 0..h {
            let (gi, fi) = (i / cell, (i % cell) as f32 / cell as f32);
            let at = |a: usize, b: usize| grid[a + gh * b];
            let top = at(gi, gj) * (1.0 - fi) + at(gi + 1, gj) * fi;
            let bottom = at(gi, gj + 1) * (1.0 - fi) + at(gi + 1, gj + 1) * fi;
            out[i + h * j] = (top * (1.0 - fj) + bottom * fj + noise[i + h * j]).clamp(0.0, 255.0);
        }
    }
    out
}

fn points_of(x: &[f32], y: &[f32], s: &[f32]) -> HashMap<(u32, u32), f32> {
    (0..x.len()).map(|i| ((x[i] as u32, y[i] as u32), s[i])).collect()
}

/// ArrayFire's points as a map from `(x, y)` to the score.
fn af_points(f: &af::Features) -> HashMap<(u32, u32), f32> {
    let n = f.num_features() as usize;
    let (x, y, s) = (afx::host(&f.xpos()), afx::host(&f.ypos()), afx::host(&f.score()));
    points_of(&x[..n], &y[..n], &s[..n])
}

fn luma_points(k: &Option<luma::Keypoints>) -> HashMap<(u32, u32), f32> {
    match k {
        None => HashMap::new(),
        Some(k) => points_of(&lumax::host(&k.x), &lumax::host(&k.y), &lumax::host(&k.score)),
    }
}

/// How the two point sets relate: ArrayFire's points luma has with the same score / has with another
/// one / lacks, and luma's extra points.
pub fn relate(luma: &HashMap<(u32, u32), f32>, af: &HashMap<(u32, u32), f32>, min_found: f64) -> Verdict {
    let (mut same, mut other, mut missing) = (0usize, 0usize, 0usize);
    for (p, s) in af {
        match luma.get(p) {
            Some(l) if (l - s).abs() <= 1e-3 * s.abs().max(1.0) => same += 1,
            Some(_) => other += 1,
            None => missing += 1,
        }
    }
    let extra = luma.keys().filter(|p| !af.contains_key(p)).count();
    let found = same as f64 / af.len().max(1) as f64;
    let text = format!("luma {} points, ArrayFire {}: {} same score, {} other score, {} missing, {} extra", luma.len(), af.len(), same, other, missing, extra);
    if found >= min_found {
        Verdict::yes(text)
    } else {
        Verdict::no(format!("NO: {text}"))
    }
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision-fast", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let data = textured(h, w, 401);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &[h as u64, w as u64]));
            for (name, arc, nm) in [("fast 9", 9u32, true), ("fast 12", 12, true), ("fast 9 without non-max", 9, false)] {
                // Compared without the limit on the points.
                let l = luma_points(&xl.fast(20.0, arc, nm, 1.0, 3).unwrap());
                let a = af_points(&af::fast(&xa, 20.0, arc, nm, 1.0, 3));
                let v = relate(&l, &a, 0.99);
                b.run(
                    &format!("15 {name}"),
                    &size,
                    v,
                    "threshold 20, edge 3, at most 20% of the pixels; both read the point count back",
                    &mut || {
                        let _ = xl.fast(20.0, arc, nm, 0.2, 3).unwrap();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let f = af::fast(&xa, 20.0, arc, nm, 0.2, 3);
                        let _ = f.num_features();
                        afx::sync();
                    },
                );
            }
        }
    });
}
