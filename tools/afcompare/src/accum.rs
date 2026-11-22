//! Incremental accumulation against ArrayFire's best equivalent.
//!
//! luma folds `B` batches into an accumulator as they arrive (the batches are never kept
//! together); ArrayFire has no incremental statistics, so its side is what a program that keeps
//! the data would do: the statistic computed once from the concatenated array (the
//! concatenation itself is outside the timed region, which favours ArrayFire: a program
//! really holding the batches would pay for it, and for the memory of all of them).

use arrayfire as af;
use luma::{Accumulator, Covariance, Df32, Fold, Histogram, TopK};

use crate::afx;
use crate::bench::Bench;
use crate::cases::{A, L};
use crate::lumax;
use crate::util::*;

fn wide(a: &luma::Array<Df32>) -> Vec<f32> {
    Accumulator::<Df32>::to_f64(a).unwrap().into_iter().map(|v| v as f32).collect()
}

pub fn accumulate(b: &mut Bench) {
    b.guarded("accum", |b| {
        meanvar(b);
        covariance(b);
        histogram(b);
        topk(b);
    });
}

fn meanvar(b: &mut Bench) {
    let ctx = b.ctx.clone();
    const F: usize = 4096;
    const R: usize = 256;
    let count = if b.quick { 16 } else { 64 };
    // Per-feature offsets and spreads.
    let host: Vec<Vec<f32>> = (0..count)
        .map(|i| uniform(F * R, 1000 + i as u64, -1.0, 1.0).iter().enumerate().map(|(k, x)| 10.0 * (k % F) as f32 + (1.0 + (k % 7) as f32) * x).collect())
        .collect();
    let lb: Vec<L> = host.iter().map(|d| lumax::arr(&ctx, d, [F, R])).collect();
    let ab: Vec<A> = host.iter().map(|d| afx::arr(d, &[F as u64, R as u64])).collect();
    let all = {
        let joined = af::join_many(1, ab.iter().collect());
        joined.eval();
        afx::sync();
        joined
    };
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).build().unwrap();
    let luma_run = |acc: &Accumulator<Df32>| {
        acc.reset();
        for batch in &lb {
            acc.push(batch).unwrap();
        }
        let (m, v) = (acc.mean().unwrap(), acc.var().unwrap());
        luma::eval_all(&[&m, &v]).unwrap();
    };
    let af_run = || {
        let (m, v) = (af::mean(&all, 1), af::var_v2(&all, af::VarianceBias::POPULATION, 1));
        af::eval_multiple(vec![&m, &v]);
    };
    luma_run(&acc);
    let (lm, lv) = (wide(&acc.mean().unwrap()), wide(&acc.var().unwrap()));
    let (am, av) = (afx::host(&af::mean(&all, 1)), afx::host(&af::var_v2(&all, af::VarianceBias::POPULATION, 1)));
    // An f64 reference for a sample of the features.
    let sample: Vec<usize> = (0..F).step_by(131).collect();
    let (mut wm, mut wv) = (Vec::new(), Vec::new());
    for &f in &sample {
        let col: Vec<f64> = host.iter().flat_map(|m| (0..R).map(move |r| f64::from(m[r * F + f]))).collect();
        let n = col.len() as f64;
        let mu = col.iter().sum::<f64>() / n;
        wm.push(mu);
        wv.push(col.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n);
    }
    let pick = |v: &[f32]| sample.iter().map(|&f| v[f]).collect::<Vec<f32>>();
    let verdict = compare(&lm, &am, 2e-6, 1e-4)
        .and(compare(&lv, &av, 5e-3, 0.0))
        .and(compare_ref("luma", &pick(&lm), &wm, 2e-6, 1e-4))
        .and(compare_ref("luma var", &pick(&lv), &wv, 5e-3, 0.0));
    let size = format!("{count} batches of 256 x 4096 (per feature)");
    b.run(
        "A1 accumulate per-feature mean + var",
        &size,
        verdict,
        "luma: incremental pushes (Df32 state, never holds the batches); ArrayFire: mean + var of the concatenated array, computed once (concatenation not timed)",
        &mut || {
            luma_run(&acc);
            ctx.sync().unwrap();
        },
        &mut || {
            af_run();
            afx::sync();
        },
    );
}

fn covariance(b: &mut Bench) {
    let ctx = b.ctx.clone();
    const F: usize = 512;
    const R: usize = 1024;
    let count = if b.quick { 8 } else { 32 };
    let host: Vec<Vec<f32>> = (0..count).map(|i| uniform(F * R, 2000 + i as u64, 0.0, 1.0)).collect();
    let lb: Vec<L> = host.iter().map(|d| lumax::arr(&ctx, d, [F, R])).collect();
    let ab: Vec<A> = host.iter().map(|d| afx::arr(d, &[F as u64, R as u64])).collect();
    let all = af::join_many(1, ab.iter().collect());
    all.eval();
    afx::sync();
    let n = (count * R) as f32;
    let acc = Covariance::<Df32>::new();
    let luma_run = |acc: &Covariance<Df32>| {
        acc.reset();
        for batch in &lb {
            acc.push(batch).unwrap();
        }
        acc.cov().unwrap().eval().unwrap();
    };
    let af_cov = || {
        let d = af::sub(&all, &af::mean(&all, 1), true);
        af::div(&af::matmul(&d, &d, af::MatProp::NONE, af::MatProp::TRANS), &n, true)
    };
    luma_run(&acc);
    let lc = wide(&acc.cov().unwrap());
    let ac = afx::host(&af_cov());
    let verdict = compare(&lc, &ac, 5e-3, 2e-5);
    let size = format!("{count} batches of 1024 x 512 (512 x 512 matrix)");
    b.run(
        "A2 accumulate feature covariance matrix",
        &size,
        verdict,
        "luma: incremental pushes (full-f32 matmul of the centered batch, Df32 state); ArrayFire: center + SGEMM of the concatenated array (concatenation not timed)",
        &mut || {
            luma_run(&acc);
            ctx.sync().unwrap();
        },
        &mut || {
            af_cov().eval();
            afx::sync();
        },
    );
}

fn histogram(b: &mut Bench) {
    let ctx = b.ctx.clone();
    const N: usize = 1 << 20;
    let count = if b.quick { 8 } else { 32 };
    let host: Vec<Vec<f32>> = (0..count).map(|i| uniform(N, 3000 + i as u64, 0.0, 1.0)).collect();
    let lb: Vec<L> = host.iter().map(|d| lumax::arr(&ctx, d, N)).collect();
    let ab: Vec<A> = host.iter().map(|d| afx::arr(d, &[N as u64])).collect();
    let all = af::join_many(0, ab.iter().collect());
    all.eval();
    afx::sync();
    let hist = Histogram::<Df32>::new(64, 0.0, 1.0).unwrap();
    let luma_run = |hist: &Histogram<Df32>| {
        hist.reset();
        for batch in &lb {
            hist.push(batch).unwrap();
        }
        hist.counts().unwrap().eval().unwrap();
    };
    luma_run(&hist);
    let lh = wide(&hist.counts().unwrap());
    let ah: Vec<f32> = afx::host(&af::histogram(&all, 64, 0.0, 1.0)).into_iter().map(|c| c as f32).collect();
    // Bin edges are rounded differently in the two: a few values near an edge may move.
    let verdict = compare(&lh, &ah, 0.0, 64.0);
    let size = format!("{count} batches of 1M f32 in [0,1), 64 bins");
    b.run(
        "A3 accumulate histogram",
        &size,
        verdict,
        "luma: incremental pushes (u32 scatter-add per push, Df32 counts); ArrayFire: histogram of the concatenated array (concatenation not timed)",
        &mut || {
            luma_run(&hist);
            ctx.sync().unwrap();
        },
        &mut || {
            af::histogram(&all, 64, 0.0, 1.0).eval();
            afx::sync();
        },
    );
}

fn topk(b: &mut Bench) {
    let ctx = b.ctx.clone();
    const N: usize = 1 << 20;
    const K: u32 = 100;
    let count = if b.quick { 8 } else { 32 };
    let host: Vec<Vec<f32>> = (0..count).map(|i| uniform(N, 4000 + i as u64, 0.0, 1.0)).collect();
    let lb: Vec<L> = host.iter().map(|d| lumax::arr(&ctx, d, N)).collect();
    let ab: Vec<A> = host.iter().map(|d| afx::arr(d, &[N as u64])).collect();
    let all = af::join_many(0, ab.iter().collect());
    all.eval();
    afx::sync();
    let top = TopK::<f32, Df32>::largest(K as usize);
    let luma_run = |top: &TopK<f32, Df32>| {
        top.reset();
        for batch in &lb {
            top.push(batch).unwrap();
        }
        top.values().unwrap().eval().unwrap();
    };
    luma_run(&top);
    let lv: Vec<f32> = top.values().unwrap().to_vec().unwrap();
    let (av, _) = af::topk(&all, K, 0, af::TopkFn::MAX);
    let mut av = afx::host(&av);
    av.sort_by(|a, b| b.total_cmp(a));
    let verdict = exact(&lv, &av);
    let size = format!("{count} batches of 1M f32, k = {K}");
    b.run(
        "A4 accumulate top-k with indices",
        &size,
        verdict,
        "luma: incremental pushes (batch top-k merged with the state's, global indices kept); ArrayFire: topk of the concatenated array (values only here; concatenation not timed)",
        &mut || {
            luma_run(&top);
            ctx.sync().unwrap();
        },
        &mut || {
            let (v, i) = af::topk(&all, K, 0, af::TopkFn::MAX);
            af::eval_multiple(vec![&v]);
            i.eval();
            afx::sync();
        },
    );
}
