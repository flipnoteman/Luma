//! Runs of keys and sets: `scan_by_key`, the reductions by key, `set_union`, `set_intersect`,
//! `set_unique` (luma `cumsum_by_key`, `*_by_key`, `set_*`; ArrayFire `scanByKey`, `*ByKey`,
//! `setUnion`, `setIntersect`, `setUnique`).

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::human;
use crate::lumax;
use crate::util::*;

/// `n` keys in runs of 1 to `2 * mean_run - 1` equal keys, ascending (consecutive runs differ).
fn run_keys(n: usize, mean_run: u32, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    let mut keys = Vec::with_capacity(n);
    let mut key = 0u32;
    while keys.len() < n {
        key += 1;
        let len = 1 + ((r.next_u64() >> 32) % u64::from(2 * mean_run - 1)) as usize;
        keys.extend(std::iter::repeat_n(key, len.min(n - keys.len())));
    }
    keys
}

fn as_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

pub fn bykey(b: &mut Bench) {
    b.guarded("bykey", |b| {
        let ctx = b.ctx.clone();
        let n = 10_000_000usize;
        let keys = run_keys(n, 100, 71);
        let vals = signs(n, 72);
        let (kl, ka) = (lumax::arr(&ctx, &keys, n), afx::arr(&keys, &[n as u64]));
        let (vl, va) = (lumax::arr(&ctx, &vals, n), afx::arr(&vals, &[n as u64]));
        // scan by key.
        let lo = lumax::host(&vl.cumsum_by_key(&kl, 0));
        let ao = afx::host(&af::scan_by_key(&ka, &va, 0, af::BinaryOp::ADD, true));
        let mut acc = 0.0f32;
        let want: Vec<f32> = (0..n)
            .map(|i| {
                acc = if i == 0 || keys[i] != keys[i - 1] { vals[i] } else { acc + vals[i] };
                acc
            })
            .collect();
        let v = exact(&lo, &ao).and(exact(&lo, &want));
        drop((lo, ao, want));
        b.run(
            "14 scan_by_key (add) f32, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys; +-1 values (exact sums); luma cumsum_by_key",
            &mut || {
                vl.cumsum_by_key(&kl, 0).eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::scan_by_key(&ka, &va, 0, af::BinaryOp::ADD, true).eval();
                afx::sync();
            },
        );
        // Reductions by key: the sizes of the results depend on the data, both sides synchronize.
        let (lk, lv) = vl.sum_by_key(&kl).unwrap();
        let (ak, av) = af::sum_by_key(&ka, &va, 0);
        let (lk, lv, ak, av) = (as_f32(&lumax::host(&lk)), lumax::host(&lv), as_f32(&afx::host(&ak)), afx::host(&av));
        let v = exact(&lk, &ak).and(exact(&lv, &av));
        b.run(
            "14 sum_by_key f32, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys; results stay on the device, the run count is read back by both",
            &mut || {
                let (k, s) = vl.sum_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                drop(k);
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::sum_by_key(&ka, &va, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        let (lk, lv) = vl.max_by_key(&kl).unwrap();
        let (ak, av) = af::max_by_key(&ka, &va, 0);
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&ak))).and(exact(&lumax::host(&lv), &afx::host(&av)));
        b.run(
            "14 max_by_key f32, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys",
            &mut || {
                let (_k, s) = vl.max_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::max_by_key(&ka, &va, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        let (lk, lv) = vl.count_by_key(&kl).unwrap();
        let (ak, av) = af::count_by_key(&ka, &va, 0);
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&ak))).and(exact(&as_f32(&lumax::host(&lv)), &afx::host(&av).iter().map(|x| x.to_bits() as f32).collect::<Vec<_>>()));
        b.run(
            "14 count_by_key (non-zero values), u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys",
            &mut || {
                let (_k, s) = vl.count_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::count_by_key(&ka, &va, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        // The other reductions; a value of 0 or 1 for any / all.
        let (lk, lv) = vl.product_by_key(&kl).unwrap();
        let (ak, av) = af::product_by_key(&ka, &va, 0);
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&ak))).and(exact(&lumax::host(&lv), &afx::host(&av)));
        b.run(
            "14 product_by_key f32, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys; +-1 values (exact products); luma: scan_by_key with * read at the run ends",
            &mut || {
                let (_k, s) = vl.product_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::product_by_key(&ka, &va, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        let (lk, lv) = vl.min_by_key(&kl).unwrap();
        let (ak, av) = af::min_by_key(&ka, &va, 0);
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&ak))).and(exact(&lumax::host(&lv), &afx::host(&av)));
        b.run(
            "14 min_by_key f32, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys",
            &mut || {
                let (_k, s) = vl.min_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::min_by_key(&ka, &va, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        let bits: Vec<f32> = vals.iter().enumerate().map(|(i, &x)| if keys[i] % 3 == 0 { 1.0 } else if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let (bl, ba) = (lumax::arr(&ctx, &bits, n), afx::arr(&bits, &[n as u64]));
        let (lk, lv) = bl.any_true_by_key(&kl).unwrap();
        let lv: Vec<f32> = lumax::host(&lv).iter().map(|&x| f32::from(u8::from(x))).collect();
        let want: Vec<f32> = keys
            .chunk_by(|a, b| a == b)
            .scan(0usize, |at, run| {
                let part = &bits[*at..*at + run.len()];
                *at += run.len();
                Some(f32::from(u8::from(part.iter().any(|&x| x != 0.0))))
            })
            .collect();
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&af::any_true_by_key(&ka, &ba, 0).0))).and(exact(&lv, &want));
        b.run(
            "14 any_true_by_key f32 0/1 values, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys; luma checked against a host reference (ArrayFire's result type is not read back)",
            &mut || {
                let (_k, s) = bl.any_true_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::any_true_by_key(&ka, &ba, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        let (lk, lv) = bl.all_true_by_key(&kl).unwrap();
        let lv: Vec<f32> = lumax::host(&lv).iter().map(|&x| f32::from(u8::from(x))).collect();
        let want: Vec<f32> = keys
            .chunk_by(|a, b| a == b)
            .scan(0usize, |at, run| {
                let part = &bits[*at..*at + run.len()];
                *at += run.len();
                Some(f32::from(u8::from(part.iter().all(|&x| x != 0.0))))
            })
            .collect();
        let v = exact(&as_f32(&lumax::host(&lk)), &as_f32(&afx::host(&af::all_true_by_key(&ka, &ba, 0).0))).and(exact(&lv, &want));
        b.run(
            "14 all_true_by_key f32 0/1 values, u32 keys",
            &human(n),
            v,
            "runs of about 100 equal keys; luma checked against a host reference (ArrayFire's result type is not read back)",
            &mut || {
                let (_k, s) = bl.all_true_by_key(&kl).unwrap();
                luma::eval_all(&[&s]).unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                let (k, s) = af::all_true_by_key(&ka, &ba, 0);
                k.eval();
                s.eval();
                afx::sync();
            },
        );
        drop((kl, ka, vl, va));
    });
}

/// `set_unique`, `set_union` and `set_intersect` (group `sets`).
pub fn sets(b: &mut Bench) {
    b.guarded("sets", |b| {
        let ctx = b.ctx.clone();
        // Sets of u32 below 2^24 (exact in f32 for the comparison).
        let m = 4_000_000usize;
        let sa = indices(m, 81, 1 << 23);
        let sb = indices(m, 82, 1 << 23);
        let (xl, xa) = (lumax::arr(&ctx, &sa, m), afx::arr(&sa, &[m as u64]));
        let (yl, ya) = (lumax::arr(&ctx, &sb, m), afx::arr(&sb, &[m as u64]));
        let lo = as_f32(&lumax::host(&xl.set_unique(false).unwrap()));
        let ao = as_f32(&afx::host(&af::set_unique(&xa, false)));
        let v = exact(&lo, &ao);
        b.run(
            "15 set_unique u32 (unsorted input)",
            &human(m),
            v,
            "",
            &mut || {
                let r = xl.set_unique(false).unwrap();
                r.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::set_unique(&xa, false).eval();
                afx::sync();
            },
        );
        let lo = as_f32(&lumax::host(&xl.set_union(&yl, false).unwrap()));
        let ao = as_f32(&afx::host(&af::set_union(&xa, &ya, false)));
        let v = exact(&lo, &ao);
        b.run(
            "15 set_union u32 (unsorted inputs)",
            &human(m),
            v,
            "two sets of the same size, duplicates inside each",
            &mut || {
                let r = xl.set_union(&yl, false).unwrap();
                r.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::set_union(&xa, &ya, false).eval();
                afx::sync();
            },
        );
        let lo = as_f32(&lumax::host(&xl.set_intersect(&yl, false).unwrap().expect("a non-empty intersection")));
        let ao = as_f32(&afx::host(&af::set_intersect(&xa, &ya, false)));
        let v = exact(&lo, &ao);
        b.run(
            "15 set_intersect u32 (unsorted inputs)",
            &human(m),
            v,
            "two sets of the same size, duplicates inside each",
            &mut || {
                let r = xl.set_intersect(&yl, false).unwrap().expect("non-empty");
                r.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::set_intersect(&xa, &ya, false).eval();
                afx::sync();
            },
        );
        // Already sorted and distinct.
        let mut ua = sa.clone();
        ua.sort_unstable();
        ua.dedup();
        let mut ub = sb.clone();
        ub.sort_unstable();
        ub.dedup();
        let (ul, ual) = (lumax::arr(&ctx, &ua, ua.len()), afx::arr(&ua, &[ua.len() as u64]));
        let (vl2, val) = (lumax::arr(&ctx, &ub, ub.len()), afx::arr(&ub, &[ub.len() as u64]));
        let lo = as_f32(&lumax::host(&ul.set_union(&vl2, true).unwrap()));
        let ao = as_f32(&afx::host(&af::set_union(&ual, &val, true)));
        let v = exact(&lo, &ao);
        b.run(
            "15 set_union u32 (sorted, distinct: is_unique)",
            &human(ua.len()),
            v,
            "",
            &mut || {
                let r = ul.set_union(&vl2, true).unwrap();
                r.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::set_union(&ual, &val, true).eval();
                afx::sync();
            },
        );
        let lo = as_f32(&lumax::host(&ul.set_intersect(&vl2, true).unwrap().expect("non-empty")));
        let ao = as_f32(&afx::host(&af::set_intersect(&ual, &val, true)));
        let v = exact(&lo, &ao);
        b.run(
            "15 set_intersect u32 (sorted, distinct: is_unique)",
            &human(ua.len()),
            v,
            "",
            &mut || {
                let r = ul.set_intersect(&vl2, true).unwrap().expect("non-empty");
                r.eval().unwrap();
                ctx.sync().unwrap();
            },
            &mut || {
                af::set_intersect(&ual, &val, true).eval();
                afx::sync();
            },
        );
    });
}
