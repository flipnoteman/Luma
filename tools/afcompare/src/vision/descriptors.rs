//! `orb`, `sift` and `gloh` against ArrayFire's functions of the same name, at 1080p and 4K on a
//! natural-looking image (blurred noise with structure).
//!
//! The results are compared as sets of keypoints (the order of the points differs, see the
//! documentation of luma's functions): the share of ArrayFire's points that luma has too (same
//! position, size and orientation), for ORB the Hamming distance of their descriptors, for SIFT and
//! GLOH the largest and mean difference of their (integer) descriptor entries. The luma side
//! reports the orientation of SIFT and GLOH in radians, ArrayFire's in degrees.

use std::collections::HashMap;
use std::ffi::c_void;

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

unsafe extern "C" {
    fn af_sift(feat: *mut *mut c_void, desc: *mut *mut c_void, input: *mut c_void, n_layers: u32, contrast: f32, edge: f32, sigma: f32, dbl: bool, scale: f32, ratio: f32) -> i32;
    fn af_gloh(feat: *mut *mut c_void, desc: *mut *mut c_void, input: *mut c_void, n_layers: u32, contrast: f32, edge: f32, sigma: f32, dbl: bool, scale: f32, ratio: f32) -> i32;
    fn af_get_features_num(n: *mut u32, f: *mut c_void) -> i32;
    fn af_get_features_xpos(a: *mut *mut c_void, f: *mut c_void) -> i32;
    fn af_get_features_ypos(a: *mut *mut c_void, f: *mut c_void) -> i32;
    fn af_get_features_score(a: *mut *mut c_void, f: *mut c_void) -> i32;
    fn af_get_features_orientation(a: *mut *mut c_void, f: *mut c_void) -> i32;
    fn af_get_features_size(a: *mut *mut c_void, f: *mut c_void) -> i32;
    fn af_release_features(f: *mut c_void) -> i32;
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

/// A keypoint set read back: position, size, orientation in degrees, descriptor entries.
struct Points {
    x: Vec<f32>,
    y: Vec<f32>,
    size: Vec<f32>,
    ori_deg: Vec<f32>,
    desc: Vec<f32>,
    len: usize,
}

impl Points {
    fn count(&self) -> usize {
        self.x.len()
    }
}

fn luma_points(k: &luma::Keypoints, desc: Vec<f32>, len: usize, radians: bool) -> Points {
    let conv = |v: Vec<f32>| if radians { v.iter().map(|a| a.to_degrees()).collect() } else { v };
    Points { x: lumax::host(&k.x), y: lumax::host(&k.y), size: lumax::host(&k.size), ori_deg: conv(lumax::host(&k.orientation)), desc, len }
}

fn af_orb_points(img: &af::Array<f32>, thr: f32, max_feat: u32, scl: f32, levels: u32, blur: bool) -> Points {
    let (f, d) = af::orb(img, thr, max_feat, scl, levels, blur);
    let n = f.num_features() as usize;
    let g = |a: af::Array<f32>| {
        let mut v = vec![0f32; n];
        a.host(&mut v);
        v
    };
    let desc = afx::host(&d);
    Points { x: g(f.xpos()), y: g(f.ypos()), size: g(f.size()), ori_deg: g(f.orientation()).iter().map(|a| a.to_degrees()).collect(), desc, len: 8 }
}

/// ArrayFire's SIFT or GLOH through the C API (the Rust crate does not wrap them).
fn af_sift_points(img: &af::Array<f32>, gloh: bool, layers: u32, contrast: f32, edge: f32, sigma: f32, double: bool, scale: f32, ratio: f32) -> Points {
    let (mut f, mut d): (*mut c_void, *mut c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
    let err = unsafe {
        if gloh {
            af_gloh(&mut f, &mut d, img.get() as *mut c_void, layers, contrast, edge, sigma, double, scale, ratio)
        } else {
            af_sift(&mut f, &mut d, img.get() as *mut c_void, layers, contrast, edge, sigma, double, scale, ratio)
        }
    };
    assert_eq!(err, 0, "af sift/gloh error {err}");
    let mut n = 0u32;
    unsafe { af_get_features_num(&mut n, f) };
    let n = n as usize;
    let col = |get: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> i32| {
        let mut a: *mut c_void = std::ptr::null_mut();
        unsafe { get(&mut a, f) };
        let arr: af::Array<f32> = (a as af::af_array).into();
        let mut v = vec![0f32; n];
        if n > 0 {
            arr.host(&mut v);
        }
        v
    };
    let (x, y, ori, size) = (col(af_get_features_xpos), col(af_get_features_ypos), col(af_get_features_orientation), col(af_get_features_size));
    let _score = col(af_get_features_score);
    let desc_arr: af::Array<f32> = (d as af::af_array).into();
    let len = if gloh { 272 } else { 128 };
    let mut desc = vec![0f32; if n > 0 { desc_arr.elements() } else { 0 }];
    if n > 0 {
        desc_arr.host(&mut desc);
    }
    unsafe { af_release_features(f) };
    Points { x, y, size, ori_deg: ori, desc, len }
}

/// Compares `got` (luma) with `want` (ArrayFire): the share of `want` found in `got`, and the
/// distances of the matched descriptors (Hamming for ORB, entries for SIFT and GLOH).
fn compare_points(got: &Points, want: &Points, orb: bool, min_share: f64) -> Verdict {
    if want.count() == 0 || got.count() == 0 {
        return Verdict::no(format!("{} luma points, {} ArrayFire points", got.count(), want.count()));
    }
    let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for i in 0..got.count() {
        grid.entry((got.x[i].round() as i32, got.y[i].round() as i32)).or_default().push(i);
    }
    let mut used = vec![false; got.count()];
    let (mut matched, mut worst, mut sum) = (0usize, 0f64, 0f64);
    for i in 0..want.count() {
        let (cx, cy) = (want.x[i].round() as i32, want.y[i].round() as i32);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let d = (got.ori_deg[j] - want.ori_deg[i]).abs();
                    let ori = if orb { d } else { d.min(360.0 - d) };
                    if !used[j] && (got.x[j] - want.x[i]).abs() < if orb { 0.5 } else { 0.05 } && (got.y[j] - want.y[i]).abs() < if orb { 0.5 } else { 0.05 } && (got.size[j] - want.size[i]).abs() < 1e-2 * want.size[i] && ori < if orb { 0.5 } else { 1.0 } {
                        found = Some(j);
                        break 'search;
                    }
                }
            }
        }
        if let Some(j) = found {
            used[j] = true;
            matched += 1;
            if orb {
                let (a, b) = (&got.desc[8 * j..8 * j + 8], &want.desc[8 * i..8 * i + 8]);
                let bits: u32 = a.iter().zip(b).map(|(p, q)| (p.to_bits() ^ q.to_bits()).count_ones()).sum();
                worst = worst.max(f64::from(bits));
                sum += f64::from(bits);
            } else {
                let len = want.len;
                let (a, b) = (&got.desc[len * j..len * (j + 1)], &want.desc[len * i..len * (i + 1)]);
                let diff = a.iter().zip(b).map(|(p, q)| f64::from((p - q).abs())).fold(0.0, f64::max);
                worst = worst.max(diff);
                sum += diff;
            }
        }
    }
    let share = matched as f64 / want.count() as f64;
    let what = if orb { "Hamming bits" } else { "max descriptor entry difference" };
    let text = format!("luma {} / AF {} points, {:.1}% of AF's found in luma, {what}: mean {:.2}, worst {worst}", got.count(), want.count(), share * 100.0, sum / matched.max(1) as f64);
    if share >= min_share { Verdict::yes(text) } else { Verdict::no(text) }
}

pub fn rows(b: &mut Bench) {
    b.guarded("vision descriptors", |b| {
        let ctx = b.ctx.clone();
        let sizes: &[(usize, usize)] = if b.quick { &[(1080, 1920)] } else { &[(1080, 1920), (2160, 3840)] };
        for &(h, w) in sizes {
            let size = format!("{w}x{h}");
            let data = natural(h, w, 11);
            let (xl, xa) = (lumax::arr(&ctx, &data, [h, w]), afx::arr(&data, &[h as u64, w as u64]));
            // ORB.
            for blur in [false, true] {
                let l = || {
                    xl.orb(20.0, 3000, 1.5, 4, blur).unwrap();
                };
                let got = xl.orb(20.0, 3000, 1.5, 4, blur).unwrap().map(|(k, d)| luma_points(&k, lumax::host(&d).iter().map(|&v| f32::from_bits(v)).collect(), 8, false)).expect("luma orb found nothing");
                let want = af_orb_points(&xa, 20.0, 3000, 1.5, 4, blur);
                let verdict = compare_points(&got, &want, true, 0.97);
                let name = if blur { "vision orb blur" } else { "vision orb" };
                b.run(
                    name,
                    &size,
                    verdict,
                    "fast_thr 20, max_feat 3000, scale 1.5, 4 levels; sets compared (ArrayFire's order depends on its CUDA blocks)",
                    &mut || {
                        l();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let _ = af_orb_points_timed(&xa, blur);
                        afx::sync();
                    },
                );
            }
            // SIFT and GLOH (default parameters).
            for gloh in [false, true] {
                let len = if gloh { 272 } else { 128 };
                let call = || if gloh { xl.gloh(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap() } else { xl.sift(3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05).unwrap() };
                let got = call().map(|(k, d)| luma_points(&k, lumax::host(&d), len, true)).expect("luma sift found nothing");
                let want = af_sift_points(&xa, gloh, 3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05);
                let verdict = compare_points(&got, &want, false, 0.9);
                b.run(
                    if gloh { "vision gloh" } else { "vision sift" },
                    &size,
                    verdict,
                    "default parameters (double_input); sets compared, orientation in degrees",
                    &mut || {
                        call();
                        ctx.sync().unwrap();
                    },
                    &mut || {
                        let _ = af_sift_points_timed(&xa, gloh);
                        afx::sync();
                    },
                );
            }
        }
    });
}

/// ArrayFire's ORB without reading the results back.
fn af_orb_points_timed(img: &af::Array<f32>, blur: bool) -> usize {
    let (f, d) = af::orb(img, 20.0, 3000, 1.5, 4, blur);
    d.eval();
    f.num_features() as usize
}

/// ArrayFire's SIFT or GLOH without reading the results back.
fn af_sift_points_timed(img: &af::Array<f32>, gloh: bool) -> u32 {
    let (mut f, mut d): (*mut c_void, *mut c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
    let err = unsafe {
        if gloh {
            af_gloh(&mut f, &mut d, img.get() as *mut c_void, 3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)
        } else {
            af_sift(&mut f, &mut d, img.get() as *mut c_void, 3, 0.04, 10.0, 1.6, true, 1.0 / 256.0, 0.05)
        }
    };
    assert_eq!(err, 0, "af sift/gloh error {err}");
    let mut n = 0u32;
    unsafe { af_get_features_num(&mut n, f) };
    let desc: af::Array<f32> = (d as af::af_array).into();
    drop(desc);
    unsafe { af_release_features(f) };
    n
}
