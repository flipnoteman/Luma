//! `canny`, `anisotropic_diffusion` and `mean_shift` against ArrayFire's functions of the same
//! name, at 1080p and 4K.
//!
//! ArrayFire's `canny` returns a `b8` array; the edge maps are compared pixel by pixel (they can
//! differ where gradient magnitudes tie in exact arithmetic or where Otsu's between-class
//! variance has a plateau, see the documentation of `Array::canny`). ArrayFire's diffusion
//! updates in place, tile by tile, luma reads one image and writes the other: values are compared
//! with a tolerance. Mean shift is compared exactly. The `fftype` / `diffusion_kind` of the C
//! function are passed through the Rust wrapper's `FluxFn` / `DiffusionEq` enums, whose names
//! are swapped (`FluxFn::GRADIENT` = 1 = quadratic flux, `FluxFn::MCDE` = 2 = exponential flux,
//! `DiffusionEq::QUADRATIC` = 1 = gradient equation, `DiffusionEq::EXPONENTIAL` = 2 = MCDE).

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

/// An image of 0..255 with smooth structure, blocks and noise (whole numbers, like 8-bit data).
fn scene(h: usize, w: usize, planes: usize, seed: u64) -> Vec<f32> {
    let noise = uniform(h * w * planes, seed, 0.0, 12.0);
    (0..h * w * planes)
        .map(|k| {
            let (i, j, p) = ((k % h) as f32, ((k / h) % w) as f32, (k / (h * w)) as f32);
            let a = ((i * 0.021 + p).sin() + (j * 0.017 - p).cos() + ((i + j) * 0.009).sin()) * 70.0 + 128.0;
            let b = if ((i as usize / 130 + j as usize / 170 + p as usize) % 2) == 0 { 60.0 } else { 0.0 };
            (a * 0.6 + b + noise[k]).floor()
        })
        .collect()
}

/// Edge maps against each other: the pixels that differ, in numbers and as a share of ArrayFire's edge pixels.
fn compare_edges(luma: &[bool], af: &[bool]) -> Verdict {
    if luma.len() != af.len() {
        return Verdict::no(format!("length: luma {} vs ArrayFire {}", luma.len(), af.len()));
    }
    let edges = af.iter().filter(|&&b| b).count();
    let bad = luma.iter().zip(af).filter(|(a, b)| a != b).count();
    let text = format!("{} edge px (AF {}), {} differ ({:.3}%)", luma.iter().filter(|&&b| b).count(), edges, bad, 100.0 * bad as f64 / edges.max(1) as f64);
    if bad * 100 <= edges.max(100) {
        Verdict::yes(text)
    } else {
        Verdict::no(format!("NO: {text}"))
    }
}

/// One diffusion step against ArrayFire's: compared on the leading columns ArrayFire's CUDA kernel
/// updated (it leaves the others unchanged, a bug that depends on the image size); the verdict
/// says how many that is and how well luma agrees there.
fn diffusion_verdict(luma: &[f32], af: &[f32], input: &[f32], h: usize) -> Verdict {
    let w = input.len() / h;
    let updated = (0..w).rev().find(|&j| (0..h).any(|i| af[i + h * j] != input[i + h * j])).map_or(0, |j| j + 1);
    if updated == 0 {
        return Verdict::no("ArrayFire changed nothing");
    }
    let n = h * updated;
    let v = compare(&luma[..n], &af[..n], 1e-3, 0.05);
    Verdict { ok: v.ok, text: format!("one step, first {updated} of {w} columns (ArrayFire's CUDA kernel updates only those): {}", v.text) }
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision-edges", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let d = [h as u64, w as u64];
            let data = scene(h, w, 1, 401);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &d));

            // Canny.
            for (name, otsu, low, high, fast, note) in [
                ("canny manual 0.1/0.3", false, 0.1f32, 0.3f32, false, "thresholds are ratios of the largest suppressed magnitude"),
                ("canny manual fast", false, 0.1, 0.3, true, "|gx| + |gy| magnitude"),
                ("canny otsu 0.3", true, 0.3, 0.9, false, "Otsu on floor(max) bins, low = 0.3 high"),
            ] {
                let mode = if otsu { luma::CannyThreshold::AutoOtsu } else { luma::CannyThreshold::Manual };
                let amode = if otsu { af::CannyThresholdType::OTSU } else { af::CannyThresholdType::MANUAL };
                let l = || xl.canny(mode, low, high, 3, fast);
                let a = || af::canny(&xa, amode, low, high, 3, fast);
                let v = compare_edges(&lumax::host(&l()), &afx::host(&a()));
                b.run(
                    &format!("13 {name}"),
                    &size,
                    v,
                    note,
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

            // Anisotropic diffusion: 5 iterations.
            for (name, flux, eq, aflux, aeq) in [
                ("anisotropic exp grad x5", luma::FluxFunction::Exponential, luma::DiffusionEq::Gradient, af::FluxFn::MCDE, af::DiffusionEq::QUADRATIC),
                ("anisotropic mcde x5", luma::FluxFunction::Exponential, luma::DiffusionEq::Mcde, af::FluxFn::MCDE, af::DiffusionEq::EXPONENTIAL),
            ] {
                let l = || xl.anisotropic_diffusion(0.1, 5.0, 5, flux, eq);
                let a = || af::anisotropic_diffusion(&xa, 0.1, 5.0, 5, aflux, aeq);
                // ArrayFire's CUDA kernel leaves the columns past a size-dependent limit (72 of 128,
                // 96 of 300, ...) unchanged, and later steps feed on the half-updated image, so the
                // verdict is taken on one step (equal normalisation) over the columns it updated.
                let v = diffusion_verdict(&lumax::host(&xl.anisotropic_diffusion(0.1, 5.0, 1, flux, eq)), &afx::host(&af::anisotropic_diffusion(&xa, 0.1, 5.0, 1, aflux, aeq)), &data, h);
                b.run(
                    &format!("13 {name}"),
                    &size,
                    v,
                    "timed: 5 steps; luma ping-pongs two images, ArrayFire updates in place and its CUDA kernel skips the trailing columns (see the verdict)",
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

            // Mean shift.
            let l = || xl.mean_shift(2.0, 30.0, 5, false);
            let a = || af::mean_shift(&xa, 2.0, 30.0, 5, false);
            let v = exact(&lumax::host(&l()), &afx::host(&a()));
            b.run(
                "13 mean_shift s=2 c=30 x5",
                &size,
                v,
                "radius 3, whole-number means like ArrayFire",
                &mut || {
                    l().eval().unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    a().eval();
                    afx::sync();
                },
            );
            let rgb = scene(h, w, 3, 402);
            let (rl, ra) = (lumax::arr(&ctx, &rgb, [h, w, 3]), afx::arr(&rgb, &[h as u64, w as u64, 3]));
            let l = || rl.mean_shift(2.0, 40.0, 5, true);
            let a = || af::mean_shift(&ra, 2.0, 40.0, 5, true);
            let v = exact(&lumax::host(&l()), &afx::host(&a()));
            b.run(
                "13 mean_shift colour s=2 c=40 x5",
                &size,
                v,
                "joint distance of the three channels",
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
    });
}
