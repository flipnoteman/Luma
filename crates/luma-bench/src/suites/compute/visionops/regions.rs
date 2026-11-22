//! Connected-component labelling (`regions`) on images of different structure and
//! `confidence_cc` with a few seeds, at 1080p and 4K. Each kind is checked against a host
//! computation on a small image before it is timed.

use luma::*;

use crate::harness::*;
use crate::suites::compute::{check64, pick, upload};

/// `[H, W]` sizes (smaller on the CPU backend).
fn sizes(h: &Harness) -> Vec<[usize; 2]> {
    if h.ctx.is_cpu() {
        vec![[270, 480]]
    } else {
        pick(h, vec![[1080, 1920]], vec![[1080, 1920], [2160, 3840]])
    }
}

fn pix(dims: [usize; 2]) -> Work {
    Work::Items((dims[0] * dims[1]) as f64, "Mpix/s")
}

fn label(dims: [usize; 2]) -> String {
    format!("{}x{}", dims[1], dims[0])
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, 0.0, 1.0)
}

/// Binary images (column-major `[h, w]`): `kind` is one of the names in [`KINDS`].
fn image(kind: &str, h: usize, w: usize) -> Vec<bool> {
    let mut out = vec![false; h * w];
    match kind {
        "noise30" | "noise45" | "noise60" => {
            let p = match kind {
                "noise30" => 0.3,
                "noise45" => 0.45,
                _ => 0.6,
            };
            for (o, v) in out.iter_mut().zip(noise(h * w, 3)) {
                *o = v < p;
            }
        }
        "blobs" => {
            // Smooth value noise on a 24 pixel grid plus fine noise, thresholded: large regions with ragged borders.
            let (gh, gw) = (h / 24 + 2, w / 24 + 2);
            let grid = noise(gh * gw, 5);
            let fine = noise(h * w, 6);
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
            for (o, v) in out.iter_mut().zip(noise(h * w, 9)) {
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
            // Lines along axis 1 on every other row joined alternately at both ends: one component of h * w / 2 pixels.
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

const KINDS: [&str; 7] = ["noise30", "noise45", "noise60", "blobs", "holes", "checker", "snake"];

/// Labels by flood fill from the first unlabelled pixel in memory order.
fn labels_host(data: &[bool], h: usize, w: usize, eight: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    let mut next = 0.0;
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !data[start] || out[start] != 0.0 {
            continue;
        }
        next += 1.0;
        out[start] = next;
        stack.push(start);
        while let Some(q) = stack.pop() {
            let (i, j) = ((q % h) as i64, (q / h) as i64);
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    if (di == 0 && dj == 0) || (!eight && di != 0 && dj != 0) {
                        continue;
                    }
                    let (a, b) = (i + di, j + dj);
                    if a < 0 || b < 0 || a >= h as i64 || b >= w as i64 {
                        continue;
                    }
                    let n = a as usize + h * b as usize;
                    if data[n] && out[n] == 0.0 {
                        out[n] = next;
                        stack.push(n);
                    }
                }
            }
        }
    }
    out
}

fn seg_host(img: &[f32], h: usize, w: usize, seed: (usize, usize), radius: usize, mult: f32, iterations: usize) -> Vec<f64> {
    // Windows fully inside the image: the interval from the seed window, one pass of the loop.
    let (mn, mx) = img.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let j: Vec<f64> = img.iter().map(|&v| f64::from((v - mn) * (1.0 / (mx - mn)) + 1.0)).collect();
    let stats = |px: &mut dyn Iterator<Item = f64>| {
        let (mut n, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for v in px {
            n += 1.0;
            s1 += v - 1.5;
            s2 += (v - 1.5) * (v - 1.5);
        }
        let mean = 1.5 + s1 / n;
        let sd = ((s2 - s1 * s1 / n) / (n - 1.0)).max(0.0).sqrt();
        (mean - f64::from(mult) * sd, mean + f64::from(mult) * sd)
    };
    let seed_v = j[seed.0 + h * seed.1];
    let mut win = (seed.1 - radius..=seed.1 + radius).flat_map(|b| (seed.0 - radius..=seed.0 + radius).map(move |a| (a, b))).map(|(a, b)| j[a + h * b]);
    let (mut lo, mut hi) = stats(&mut win);
    let grow = |lo: f64, hi: f64| {
        let mut seg = vec![false; h * w];
        let mut stack = vec![seed.0 + h * seed.1];
        seg[stack[0]] = true;
        while let Some(q) = stack.pop() {
            let (i, jj) = ((q % h) as i64, (q / h) as i64);
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    let (a, b) = (i + di, jj + dj);
                    if a < 0 || b < 0 || a >= h as i64 || b >= w as i64 {
                        continue;
                    }
                    let n = a as usize + h * b as usize;
                    if !seg[n] && j[n] >= lo && j[n] <= hi {
                        seg[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        seg
    };
    lo = lo.min(seed_v);
    hi = hi.max(seed_v);
    let mut seg = grow(lo, hi);
    for _ in 0..iterations {
        let mut it = seg.iter().enumerate().filter(|(_, b)| **b).map(|(q, _)| j[q]);
        let (l, u) = stats(&mut it);
        seg = grow(l.min(seed_v), u.max(seed_v));
    }
    seg.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()
}

/// Noise with a few bright discs (axis 0 first).
fn blobby(h: usize, w: usize) -> Vec<f32> {
    let mut img = values(h * w, 12, 0.0, 10.0);
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

pub fn run(h: &mut Harness) {
    h.guarded("visionops/regions", |h| {
        // Small checks against the host.
        let (sh, sw) = (61usize, 83usize);
        for kind in KINDS {
            let data = image(kind, sh, sw);
            let img = Array::from_slice_on(&h.ctx, &data, [sh, sw]).unwrap();
            for (conn, eight) in [(Connectivity::Four, false), (Connectivity::Eight, true)] {
                let got = img.regions(conn).to_f32().to_vec().unwrap();
                check64(&format!("regions/{kind}"), &got, &labels_host(&data, sh, sw, eight), 0.0, 0.0);
            }
        }
        let simg = blobby(sh, sw);
        let a = upload(h, &simg, [sh, sw]);
        let got = a.confidence_cc_at(&[30], &[40], 2, 2, 2, 1.0).unwrap().to_vec().unwrap();
        check64("confidence_cc", &got, &seg_host(&simg, sh, sw, (30, 40), 2, 2.0, 2), 0.0, 0.0);

        for dims in sizes(h) {
            let [ih, iw] = dims;
            let tag = label(dims);
            for kind in KINDS {
                if kind == "snake" && ih * iw > 3_000_000 {
                    continue;
                }
                let data = image(kind, ih, iw);
                let img = Array::from_slice_on(&h.ctx, &data, [ih, iw]).unwrap();
                img.eval().unwrap();
                for (conn, name) in [(Connectivity::Four, "4"), (Connectivity::Eight, "8")] {
                    if matches!(kind, "noise60" | "holes" | "snake") && name == "4" || matches!(kind, "checker") && name == "8" {
                        continue;
                    }
                    h.case(&format!("regions/{kind}_{name}"), &tag, pix(dims), &mut || img.regions(conn).eval().unwrap());
                }
            }
            let img = upload(h, &blobby(ih, iw), dims);
            let (cx, cy) = (6 * ih / 20 + 3, 7 * iw / 21 + 4);
            let one = (vec![cx as u32], vec![cy as u32]);
            for (name, iterations) in [("0", 0u32), ("3", 3)] {
                h.case(&format!("confidence_cc/1seed_{name}it"), &tag, pix(dims), &mut || img.confidence_cc_at(&one.0, &one.1, 2, 2, iterations, 1.0).unwrap().eval().unwrap());
            }
            let sx: Vec<u32> = (0..8).map(|k| (k * 131 % ih) as u32).collect();
            let sy: Vec<u32> = (0..8).map(|k| (k * 211 % iw) as u32).collect();
            h.case("confidence_cc/8seeds_3it", &tag, pix(dims), &mut || img.confidence_cc_at(&sx, &sy, 2, 2, 3, 1.0).unwrap().eval().unwrap());
        }
    });
}
