//! `orb`, `sift` and `gloh` at 1080p and 4K on a natural-looking image (blurred noise with
//! structure, a few thousand keypoints). Each is checked on a small image first: the descriptors
//! of ORB are recomputed on the host from the keypoints it returned, the ones of SIFT and GLOH
//! have to be integers in range and the points inside the image (the complete comparison with a
//! host implementation is in `tests/all/vision_descriptors.rs`).

use luma::*;

use super::super::{pick, upload};
use crate::harness::*;

/// `[H, W]` sizes: 1080p and 4K (small under `--quick` and on the CPU backend).
fn sizes(h: &Harness) -> Vec<[usize; 2]> {
    if h.ctx.is_cpu() {
        vec![[270, 480]]
    } else {
        pick(h, vec![[1080, 1920]], vec![[1080, 1920], [2160, 3840]])
    }
}

fn label(dims: [usize; 2]) -> String {
    format!("{}x{}", dims[1], dims[0])
}

/// Value noise at four scales plus rectangles, in `0..=255`.
fn natural(h: usize, w: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32
    };
    let mut img = vec![0f32; h * w];
    for &(cell, amp) in &[(2usize, 0.25f32), (5, 0.5), (12, 1.0), (30, 1.5)] {
        let (gh, gw) = (h / cell + 3, w / cell + 3);
        let g: Vec<f32> = (0..gh * gw).map(|_| rnd()).collect();
        for j in 0..w {
            for i in 0..h {
                let (fy, fx) = (i as f32 / cell as f32, j as f32 / cell as f32);
                let (y0, x0) = (fy as usize, fx as usize);
                let sm = |t: f32| t * t * (3.0 - 2.0 * t);
                let (ty, tx) = (sm(fy - y0 as f32), sm(fx - x0 as f32));
                let v = g[y0 + gh * x0] * (1.0 - ty) * (1.0 - tx) + g[y0 + 1 + gh * x0] * ty * (1.0 - tx) + g[y0 + gh * (x0 + 1)] * (1.0 - ty) * tx + g[y0 + 1 + gh * (x0 + 1)] * ty * tx;
                img[i + h * j] += amp * v;
            }
        }
    }
    for _ in 0..h * w / 3000 {
        let (r0, c0) = ((rnd() * h as f32) as usize, (rnd() * w as f32) as usize);
        let (dh, dw) = (3 + (rnd() * 20.0) as usize, 3 + (rnd() * 20.0) as usize);
        let v = rnd() * 1.5 - 0.5;
        for j in c0..(c0 + dw).min(w) {
            for i in r0..(r0 + dh).min(h) {
                img[i + h * j] += v;
            }
        }
    }
    let (mn, mx) = img.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    img.iter().map(|v| ((v - mn) / (mx - mn) * 255.0).round()).collect()
}

/// The BRIEF pattern of ORB, read from the source file that holds it.
fn pattern() -> Vec<i32> {
    let text = include_str!("../../../../../../src/vision/descriptors/orb_pattern.rs");
    let start = text.find("PATTERN: [i32; 1024] = [").unwrap();
    let body = &text[start + 24..];
    text_numbers(&body[..body.find("];").unwrap()])
}

fn text_numbers(s: &str) -> Vec<i32> {
    s.split(',').filter_map(|t| t.trim().parse().ok()).collect()
}

/// ORB: the level-0 descriptors recomputed from the returned keypoints.
fn check_orb(data: &[f32], [h, w]: [usize; 2], points: &Keypoints, desc: &Array<u32>) {
    let (x, y, o, z) = (points.x.to_vec().unwrap(), points.y.to_vec().unwrap(), points.orientation.to_vec().unwrap(), points.size.to_vec().unwrap());
    let d = desc.to_vec().unwrap();
    let pat = pattern();
    assert_eq!(pat.len(), 1024);
    let mut checked = 0;
    for i in 0..x.len() {
        assert!(x[i] >= 0.0 && x[i] < w as f32 && y[i] >= 0.0 && y[i] < h as f32, "orb: keypoint outside the image");
        if z[i] != 31.0 {
            continue;
        }
        let (col, row) = (x[i] as i64, y[i] as i64);
        let (sn, cs) = (o[i].sin(), o[i].cos());
        let mut bad = 0;
        for t in 0..256 {
            let px = |px: f32, py: f32| {
                let c = col + (px * cs - py * sn).round() as i64;
                let r = row + (px * sn + py * cs).round() as i64;
                data[r as usize + h * c as usize]
            };
            let bit = px(pat[4 * t] as f32, pat[4 * t + 1] as f32) < px(pat[4 * t + 2] as f32, pat[4 * t + 3] as f32);
            if bit != ((d[8 * i + t / 32] >> (t % 32)) & 1 == 1) {
                bad += 1;
            }
        }
        assert!(bad <= 4, "orb: {bad} wrong descriptor bits of keypoint {i}");
        checked += 1;
    }
    assert!(checked > 0, "orb: no keypoint of level 0 to check");
}

fn check_sift(name: &str, [h, w]: [usize; 2], points: &Keypoints, desc: &Array<f32>, len: usize) {
    let (x, y) = (points.x.to_vec().unwrap(), points.y.to_vec().unwrap());
    let d = desc.to_vec().unwrap();
    assert_eq!(d.len(), len * x.len(), "{name}: descriptor shape");
    assert!(x.iter().all(|&v| v >= 0.0 && v < w as f32) && y.iter().all(|&v| v >= 0.0 && v < h as f32), "{name}: keypoint outside the image");
    assert!(d.iter().all(|&v| (0.0..=255.0).contains(&v) && v.fract() == 0.0), "{name}: descriptor entry out of range");
    for i in 0..x.len() {
        assert!(d[len * i..len * (i + 1)].iter().any(|&v| v > 0.0), "{name}: empty descriptor");
    }
}

pub fn run(h: &mut Harness) {
    h.guarded("visionops/descriptors", |h| {
        // Checks on a small image.
        let small = [120usize, 160];
        let sdata = natural(small[0], small[1], 5);
        let simg = upload(h, &sdata, small);
        let (p, d) = simg.orb(20.0, 300, 1.5, 3, false).unwrap().expect("orb found nothing on the small image");
        check_orb(&sdata, small, &p, &d);
        let (p, d) = simg.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap().expect("sift found nothing on the small image");
        check_sift("sift", small, &p, &d, 128);
        let (p, d) = simg.gloh(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap().expect("gloh found nothing on the small image");
        check_sift("gloh", small, &p, &d, 272);

        for dims in sizes(h) {
            let tag = label(dims);
            let img = upload(h, &natural(dims[0], dims[1], 11), dims);
            let pixels = (dims[0] * dims[1]) as f64;
            let work = Work::Items(pixels, "Mpix/s");
            let ctx = h.ctx.clone();
            h.case("visionops/descriptors/orb", &tag, work, &mut || {
                img.orb(20.0, 3000, 1.5, 4, false).unwrap();
                ctx.sync().unwrap();
            });
            h.case("visionops/descriptors/orb_blur", &tag, work, &mut || {
                img.orb(20.0, 3000, 1.5, 4, true).unwrap();
                ctx.sync().unwrap();
            });
            h.case("visionops/descriptors/sift", &tag, work, &mut || {
                img.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap();
                ctx.sync().unwrap();
            });
            h.case("visionops/descriptors/sift_nodouble", &tag, work, &mut || {
                img.sift(3, 0.04, 10.0, 1.6, false, 1.0 / 256.0, 0.05).unwrap();
                ctx.sync().unwrap();
            });
            h.case("visionops/descriptors/gloh", &tag, work, &mut || {
                img.gloh(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap();
                ctx.sync().unwrap();
            });
        }
    });
}
