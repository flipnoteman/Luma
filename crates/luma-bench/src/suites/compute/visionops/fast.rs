//! `Array::fast` at 1080p and 4K on a textured image (value noise with per-pixel noise, 0..255),
//! checked against a host implementation of the definition on a small image first.

use luma::*;

use super::super::{pick, upload};
use crate::harness::*;

/// Value noise on a grid of `cell` pixels plus per-pixel noise: an image with corners everywhere.
pub fn textured(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let cell = 8usize;
    let (gh, gw) = (h / cell + 2, w / cell + 2);
    let grid = values(gh * gw, seed, 0.0, 255.0);
    let noise = values(h * w, seed + 1, -10.0, 10.0);
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

const RING: [(i64, i64); 16] = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3), (0, -3), (1, -3), (2, -2), (3, -1)];

/// The corners `(x, y, score)` (x along axis 1) of the definition, sorted.
fn fast_host(data: &[f32], h: usize, w: usize, thr: f32, arc: usize, non_max: bool) -> Vec<(u32, u32, f32)> {
    let mut score = vec![0.0f32; h * w];
    for j in 3..w - 3 {
        for i in 3..h - 3 {
            let p = data[i + h * j];
            let ring: Vec<f32> = RING.iter().map(|&(d0, d1)| data[(i as i64 + d0) as usize + h * (j as i64 + d1) as usize]).collect();
            let bright: Vec<bool> = ring.iter().map(|&r| r > p + thr).collect();
            let dark: Vec<bool> = ring.iter().map(|&r| r < p - thr).collect();
            let sb: f32 = (0..16).filter(|&k| bright[k]).map(|k| ring[k] - (p + thr)).sum();
            let sd: f32 = (0..16).filter(|&k| dark[k]).map(|k| (p - thr) - ring[k]).sum();
            let run = |f: &[bool]| (0..16).any(|s| (0..arc).all(|d| f[(s + d) % 16]));
            let mut out = 0.0f32;
            if run(&bright) {
                out = sb;
            }
            if run(&dark) {
                out = out.max(sd);
            }
            score[i + h * j] = out;
        }
    }
    let mut out = Vec::new();
    for j in 0..w {
        for i in 0..h {
            let s = score[i + h * j];
            if s <= 0.0 {
                continue;
            }
            let mut keep = true;
            if non_max {
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        let (a, b) = (i as i64 + di, j as i64 + dj);
                        if (di, dj) != (0, 0) && a >= 0 && b >= 0 && (a as usize) < h && (b as usize) < w && score[a as usize + h * b as usize] >= s {
                            keep = false;
                        }
                    }
                }
            }
            if keep {
                out.push((j as u32, i as u32, s));
            }
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn points(k: &Option<Keypoints>) -> Vec<(u32, u32, f32)> {
    let Some(k) = k else { return Vec::new() };
    let (x, y, s) = (k.x.to_vec().unwrap(), k.y.to_vec().unwrap(), k.score.to_vec().unwrap());
    let mut v: Vec<_> = (0..x.len()).map(|i| (x[i] as u32, y[i] as u32, s[i])).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

pub fn run(h: &mut Harness) {
    h.guarded("visionops/fast", |h| {
        // The host definition on a small image, for every arc length that is timed.
        let small = [61usize, 77];
        let sdata = textured(small[0], small[1], 5);
        let simg = upload(h, &sdata, small);
        for (arc, nm) in [(9u32, true), (12, true), (9, false)] {
            let got = points(&simg.fast(20.0, arc, nm, 1.0, 3).unwrap());
            assert_eq!(got, fast_host(&sdata, small[0], small[1], 20.0, arc as usize, nm), "visionops/fast: arc {arc} non_max {nm}");
        }
        let sizes: Vec<[usize; 2]> = if h.ctx.is_cpu() { vec![[270, 480]] } else { pick(h, vec![[1080, 1920]], vec![[1080, 1920], [2160, 3840]]) };
        for dims in sizes {
            let img = upload(h, &textured(dims[0], dims[1], 11), dims);
            let tag = format!("{}x{}", dims[1], dims[0]);
            let work = Work::Items((dims[0] * dims[1]) as f64, "Mpix/s");
            for (name, arc, nm) in [("fast_9", 9u32, true), ("fast_12", 12, true), ("fast_9_all", 9, false)] {
                h.case(&format!("visionops/{name}"), &tag, work, &mut || {
                    let _ = img.fast(20.0, arc, nm, 0.2, 3).unwrap();
                });
            }
        }
    });
}
