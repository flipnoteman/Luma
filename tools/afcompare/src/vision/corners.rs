//! `harris`, `susan` and `dog` of luma against ArrayFire's functions of the same name at 1080p and
//! 4K.
//!
//! The detectors are compared as sets of `(x, y)` with their scores: both libraries find the same
//! corners (a strict maximum of the response among its eight neighbours) except where two
//! neighbouring responses are equal up to float rounding, so the verdict allows a few such
//! differences (0.1 % of the corners, at least 3) and compares the scores of the common ones.
//! ArrayFire's `susan` returns `x` as the row index, luma's as the column like every detector:
//! the comparison swaps them. The order of the corners is not compared (ArrayFire's CUDA backend
//! returns them in no fixed order, luma's in tile order).

use std::collections::HashMap;

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

/// Blocks of different brightness plus noise (about as many corners as a photograph has).
fn scene(h: usize, w: usize, seed: u64) -> Vec<f32> {
    scene_noise(h, w, seed, 12.0)
}

/// [`scene`] with noise of amplitude `noise` (0: only the corners of the blocks).
fn scene_noise(h: usize, w: usize, seed: u64, noise: f32) -> Vec<f32> {
    let noise = uniform(h * w, seed, 0.0, noise.max(1e-6));
    let blocks = uniform(256, seed + 1, 0.0, 255.0);
    (0..h * w)
        .map(|k| {
            let (i, j) = (k % h, k / h);
            blocks[((i / 17) * 13 + (j / 23) * 5) % 256] + noise[k]
        })
        .collect()
}

fn luma_points(k: &Option<luma::Keypoints>) -> HashMap<(u32, u32), f32> {
    match k {
        None => HashMap::new(),
        Some(k) => {
            let (x, y, s) = (lumax::host(&k.x), lumax::host(&k.y), lumax::host(&k.score));
            (0..x.len()).map(|i| ((x[i] as u32, y[i] as u32), s[i])).collect()
        }
    }
}

fn af_points(f: &af::Features, swap: bool) -> HashMap<(u32, u32), f32> {
    let (x, y, s) = (afx::host(&f.xpos()), afx::host(&f.ypos()), afx::host(&f.score()));
    (0..x.len()).map(|i| (if swap { (y[i] as u32, x[i] as u32) } else { (x[i] as u32, y[i] as u32) }, s[i])).collect()
}

/// The verdict of two corner sets.
/// `floor` is the score below which a relative difference means nothing (SUSAN's response is a
/// difference of nearly equal numbers, so its rounding error is relative to `geom_thr`).
fn sets(luma: &HashMap<(u32, u32), f32>, af: &HashMap<(u32, u32), f32>, floor: f32) -> Verdict {
    let only_luma = luma.keys().filter(|p| !af.contains_key(p)).count();
    let only_af = af.keys().filter(|p| !luma.contains_key(p)).count();
    let tol = (af.len() / 1000).max(3);
    let mut worst = 0.0f64;
    for (p, &s) in luma {
        if let Some(&a) = af.get(p) {
            worst = worst.max((f64::from(s) - f64::from(a)).abs() / f64::from(a).abs().max(f64::from(floor)));
        }
    }
    let text = format!("{} corners (ArrayFire {}), {only_luma} only in luma, {only_af} only in ArrayFire, scores of the common ones within {worst:.1e} relative (to the larger of the score and the floor)", luma.len(), af.len());
    if only_luma <= tol && only_af <= tol && worst <= 1e-3 {
        Verdict::yes(text)
    } else {
        Verdict::no(format!("NO: {text}"))
    }
}

type Lk = Option<luma::Keypoints>;

fn lr<'a>(ctx: &'a luma::Context, f: impl Fn() -> Lk + 'a) -> impl FnMut() + 'a {
    move || {
        if let Some(k) = f() {
            luma::eval_all(&[&k.x, &k.y, &k.score, &k.orientation, &k.size]).unwrap();
        }
        ctx.sync().unwrap();
    }
}

fn ar<'a>(f: impl Fn() -> af::Features + 'a) -> impl FnMut() + 'a {
    move || {
        let k = f();
        k.xpos().eval();
        k.ypos().eval();
        k.score().eval();
        afx::sync();
    }
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision corners", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let data = scene(h, w, 401);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &[h as u64, w as u64]));

            // Harris: the strongest 500, all above a threshold, a box window.
            for (name, max, minr, sigma, block, note) in [
                ("harris 500 (sigma 1)", 500usize, 1e5f32, 1.0f32, 0usize, "Gaussian window of 5 cells; the strongest 500, descending; same corners as ArrayFire, order of ties may differ"),
                ("harris all >= 1e5 (sigma 1)", 0, 1e5, 1.0, 0, "every corner of response >= 1e5; ArrayFire lists them by column, luma by tile"),
                ("harris box 3", 500, 1e5, 0.1, 3, "window of 3 equal weights (block_size 3, sigma < 0.5)"),
                ("harris sigma 3", 500, 1e5, 3.0, 0, "Gaussian window of 17 cells"),
            ] {
                let l = || xl.harris(max, minr, sigma, block as usize, 0.04).unwrap();
                let a = || af::harris(&xa, max as u32, minr, sigma, block as u32, 0.04);
                let v = sets(&luma_points(&l()), &af_points(&a(), false), 1e-30);
                b.run(&format!("14 {name}"), &size, v, note, &mut lr(&ctx, l), &mut ar(a));
            }
            // The same without noise: few corners, so the ranking has little to sort.
            {
                let clean = scene_noise(h, w, 402, 0.0);
                let (cl, ca) = (lumax::arr(&ctx, &clean, [h, w]), afx::arr(&clean, &[h as u64, w as u64]));
                let l = || cl.harris(500, 1e5, 1.0, 0, 0.04).unwrap();
                let a = || af::harris(&ca, 500, 1e5, 1.0, 0, 0.04);
                let v = sets(&luma_points(&l()), &af_points(&a(), false), 1e-30);
                b.run("14 harris 500, no noise", &size, v, "image of blocks without noise (few corners); the strongest 500", &mut lr(&ctx, l), &mut ar(a));
            }
            // SUSAN.
            for (name, r, t, g, edge) in [("susan r3", 3usize, 32.0f32, 10.0f32, 3usize), ("susan r5", 5, 32.0, 30.0, 5)] {
                let l = || xl.susan(r, t, g, 0.05, edge).unwrap();
                let a = || af::susan(&xa, r as u32, t, g, 0.05, edge as u32);
                let v = sets(&luma_points(&l()), &af_points(&a(), true), g * 1e-3);
                b.run(&format!("14 {name}"), &size, v, "x and y compared swapped (ArrayFire's x is the row)", &mut lr(&ctx, l), &mut ar(a));
            }
            // Difference of Gaussians.
            for (r1, r2) in [(1usize, 2usize), (3, 5), (8, 4)] {
                let l = || xl.dog(r1, r2);
                let a = || af::dog(&xa, r1 as i32, r2 as i32);
                let v = compare(&lumax::host(&l()), &afx::host(&a()), 1e-4, 2e-3);
                b.run(
                    &format!("14 dog {r1}-{r2}"),
                    &size,
                    v,
                    "zero border, same size",
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
