//! Compact integer arrays: luma's `i16` / `u16` / `u8` arrays against ArrayFire's `s16` / `u16` /
//! `u8` ones, the same data on both sides: host upload, a cast fused into a sum, per-column
//! statistics of a trace matrix and byte histograms. The matrix of `traces` rows of `samples`
//! columns is `[samples, traces]` in both libraries, its columns reduced along axis 1.

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::lumax;
use crate::util::*;

fn noise(n: usize, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.next_u64() >> 24).collect()
}

/// `samples x traces` of 12-bit noise around a level per column.
fn traces(samples: usize, count: usize, seed: u64) -> Vec<i16> {
    let raw = noise(samples * count, seed);
    (0..samples * count).map(|i| (((raw[i] % 4096) as i64) - 2048 + ((i % samples) as i64 * 7 % 600) - 300) as i16).collect()
}

pub fn compact(b: &mut Bench) {
    b.guarded("compact", |b| {
        let ctx = b.ctx.clone();
        let (samples, count) = (1024usize, 32768usize);
        let n = samples * count;
        let size = format!("{samples} x {count} ({}M elements)", n >> 20);
        let hi = traces(samples, count, 11);

        // ---- host upload: both sides allocate, copy and evaluate ----
        let dims = [samples as u64, count as u64];
        let up_l = |data: &[i16]| {
            let a = luma::Array::from_slice_on(&ctx, data, [samples, count]).unwrap();
            a.eval().unwrap();
            ctx.sync().unwrap();
        };
        let up_a = |data: &[i16]| {
            let a = af::Array::new(data, afx::dims(&dims));
            a.eval();
            afx::sync();
        };
        let (l, a) = (luma::Array::from_slice_on(&ctx, &hi, [samples, count]).unwrap(), af::Array::new(&hi, afx::dims(&dims)));
        let same = lumax::host(&l) == afx::host(&a);
        b.run(
            "16 upload s16 (host to device)",
            &size,
            if same { Verdict::yes("yes (identical)") } else { Verdict::no("NO: read back differently") },
            "allocation, copy and evaluation; the host slice is the same",
            &mut || up_l(&hi),
            &mut || up_a(&hi),
        );
        drop((l, a));

        let (xl, xa) = (lumax::arr(&ctx, &hi, [samples, count]), afx::arr(&hi, &dims));

        // ---- cast + sum ----
        let want: i64 = hi.iter().map(|&v| i64::from(v)).sum();
        let (ls, asum) = (xl.sum().item().unwrap(), f64::from(af::sum_all(&xa).0));
        let v = if i64::from(ls) == want && (asum - want as f64).abs() <= 1.0 { Verdict::yes(format!("yes (sum {want})")) } else { Verdict::no(format!("NO: sum luma {ls} ArrayFire {asum} host {want}")) };
        b.run(
            "17 sum of s16 (read as 2 bytes, 32-bit sum)",
            &size,
            v,
            "luma: Array<i16>::sum (i32); ArrayFire: sum_all of the s16 array",
            &mut || {
                xl.sum().item().unwrap();
            },
            &mut || {
                af::sum_all(&xa);
            },
        );
        let lf = xl.to_f32().sum().item().unwrap();
        let af_f = f64::from(af::sum_all(&xa.cast::<f32>()).0);
        let v = if (f64::from(lf) - want as f64).abs() <= 1e-3 * (want as f64).abs().max(1e4) && (af_f - want as f64).abs() <= 1e-3 * (want as f64).abs().max(1e4) {
            Verdict::yes(format!("yes (sum {want})"))
        } else {
            Verdict::no(format!("NO: f32 sum luma {lf} ArrayFire {af_f} host {want}"))
        };
        b.run(
            "18 cast s16 to f32 + sum",
            &size,
            v,
            "luma: the cast is fused into the reduction; ArrayFire: cast (a 4-byte array) then sum_all",
            &mut || {
                xl.to_f32().sum().item().unwrap();
            },
            &mut || {
                af::sum_all(&xa.cast::<f32>());
            },
        );

        // ---- per-column mean and variance over the traces ----
        let (mut sum, mut sq) = (vec![0f64; samples], vec![0f64; samples]);
        for (i, &v) in hi.iter().enumerate() {
            sum[i % samples] += f64::from(v);
            sq[i % samples] += f64::from(v) * f64::from(v);
        }
        let mean: Vec<f64> = sum.iter().map(|&s| s / count as f64).collect();
        let var: Vec<f64> = (0..samples).map(|c| sq[c] / count as f64 - mean[c] * mean[c]).collect();
        let stats_l = |x: &luma::Array<i16>| (x.mean_axis(1), x.var_axis(1));
        let stats_a = |x: &af::Array<i16>| (af::mean(x, 1), af::var_v2(x, af::VarianceBias::POPULATION, 1));
        let ((lm, lv), (am, av)) = (stats_l(&xl), stats_a(&xa));
        let (lmh, lvh, amh, avh) = (lumax::host(&lm), lumax::host(&lv), afx::host(&am), afx::host(&av));
        let v = compare(&lmh, &amh, 1e-4, 1e-3)
            .and(compare(&lvh, &avh, 1e-3, 1e-1))
            .and(compare_ref("luma mean", &lmh, &mean, 1e-4, 1e-3))
            .and(compare_ref("luma var", &lvh, &var, 1e-3, 1e-1));
        b.run(
            "19 per-column mean + var of s16",
            &size,
            v,
            "population variance over the traces (axis 1); both results evaluated together",
            &mut || {
                let (m, v) = stats_l(&xl);
                luma::eval_all(&[&m, &v]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (m, v) = stats_a(&xa);
                af::eval_multiple(vec![&m, &v]);
                afx::sync();
            },
        );
        // The same on f32 for scale: the 4-byte arrays of both libraries.
        let hf: Vec<f32> = hi.iter().map(|&v| f32::from(v)).collect();
        let (fl, fa) = (lumax::arr(&ctx, &hf, [samples, count]), afx::arr(&hf, &dims));
        b.run(
            "20 per-column mean + var of f32 (same data)",
            &size,
            Verdict::yes("same data as 19"),
            "the 4-byte baseline of 19",
            &mut || {
                let (m, v) = (fl.mean_axis(1), fl.var_axis(1));
                luma::eval_all(&[&m, &v]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (m, v) = (af::mean(&fa, 1), af::var_v2(&fa, af::VarianceBias::POPULATION, 1));
                af::eval_multiple(vec![&m, &v]);
                afx::sync();
            },
        );
        drop((fl, fa, xl, xa));

        // ---- histograms of bytes and u16 ----
        let m = 40_000_000usize;
        let bytes: Vec<u8> = noise(m, 12).iter().map(|&v| (v as u8) & ((v >> 8) as u8)).collect();
        let mut want = vec![0u32; 256];
        bytes.iter().for_each(|&v| want[v as usize] += 1);
        let (bl, ba) = (lumax::arr(&ctx, &bytes, m), afx::arr(&bytes, &[m as u64]));
        let (lh, ah) = (lumax::host(&bl.histogram()), afx::host(&af::histogram(&ba, 256, 0.0, 256.0)));
        let v = if lh == want && ah == want { Verdict::yes("yes (exact)") } else { Verdict::no("NO: histogram differs from the host count") };
        b.run(
            "21 histogram of u8 (256 bins)",
            &format!("{}M bytes", m >> 20),
            v,
            "luma: Array<u8>::histogram; ArrayFire: histogram of the u8 array over [0, 256)",
            &mut || {
                bl.histogram().eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::histogram(&ba, 256, 0.0, 256.0).eval();
                afx::sync();
            },
        );
        drop((bl, ba));
        let halves: Vec<u16> = noise(m, 13).iter().map(|&v| (v as u16) & 0x3fff).collect();
        let mut want = vec![0u32; 16384];
        halves.iter().for_each(|&v| want[v as usize] += 1);
        let (hl, ha) = (lumax::arr(&ctx, &halves, m), afx::arr(&halves, &[m as u64]));
        let (lh, ah) = (lumax::host(&hl.bincount(16384)), afx::host(&af::histogram(&ha, 16384, 0.0, 16384.0)));
        let v = if lh == want && ah == want { Verdict::yes("yes (exact)") } else { Verdict::no("NO: histogram differs from the host count") };
        b.run(
            "22 histogram of u16 (16384 bins)",
            &format!("{}M values", m >> 20),
            v,
            "luma: Array<u16>::bincount; ArrayFire: histogram of the u16 array over [0, 16384)",
            &mut || {
                hl.bincount(16384).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::histogram(&ha, 16384, 0.0, 16384.0).eval();
                afx::sync();
            },
        );
    });
}
