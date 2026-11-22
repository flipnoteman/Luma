//! The `accum` suite: incremental accumulation (`Accumulator`, `Covariance`, `Histogram`,
//! `Histogram2d`, `TopK`) of data that arrives in batches, against computing the same statistic
//! once from the concatenated data.
//!
//! A case that pushes `B` batches says so in its size (`64x4096x256`: 64 batches of 256 rows of
//! 4096 features). The `_concat` cases hold the same data as one array and compute the result
//! once (the `_concat_build` ones include building that array from the batches): what a program
//! that keeps everything would do, at the memory price of all the data.

use luma::*;

use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("meanvar", meanvar_cases);
    h.guarded("cov", cov_cases);
    h.guarded("hist", hist_cases);
    h.guarded("topk", topk_cases);
    h.guarded("group", group_cases);
}

/// A few distinct device batches of `shape`, cycled through by the pushes.
fn batches(h: &Harness, count: usize, shape: [usize; 2], seed: u64, lo: f32, hi: f32) -> Vec<Array<f32>> {
    (0..count)
        .map(|i| {
            let data = values(shape[0] * shape[1], seed + i as u64, lo, hi);
            Array::from_slice_on(&h.ctx, &data, shape).unwrap()
        })
        .collect()
}

fn up(h: &Harness, data: &[f32], shape: impl Into<Shape>) -> Array<f32> {
    Array::from_slice_on(&h.ctx, data, shape).unwrap()
}

fn host(a: &Array<Df32>) -> Vec<f64> {
    Accumulator::<Df32>::to_f64(a).unwrap()
}

fn pushes(h: &Harness) -> usize {
    if h.quick { 16 } else { 64 }
}

// -------------------------------------------------------------------------------------------------
// Per-feature mean and variance
// -------------------------------------------------------------------------------------------------

fn meanvar_cases(h: &mut Harness) {
    const F: usize = 4096;
    const R: usize = 256;
    let b = pushes(h);
    let distinct = b.min(16);
    // Features with different offsets and spreads, so that the check means something.
    let data: Vec<Array<f32>> = (0..distinct)
        .map(|i| {
            let v = values(F * R, 300 + i as u64, -1.0, 1.0);
            let shaped: Vec<f32> = v.iter().enumerate().map(|(k, x)| 10.0 * (k % F) as f32 + (1.0 + (k % 7) as f32) * x).collect();
            up(h, &shaped, [F, R])
        })
        .collect();
    let size = format!("{b}x{F}x{R}");
    let bytes = (b * F * R * 4) as f64;

    // Verification: the accumulator against host two-pass statistics of the first batches.
    {
        let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).build().unwrap();
        let rows: Vec<Vec<f32>> = data.iter().take(3).map(|a| a.to_vec().unwrap()).collect();
        for a in data.iter().take(3) {
            acc.push(a).unwrap();
        }
        let (mean, var) = (host(&acc.mean().unwrap()), host(&acc.var().unwrap()));
        for f in (0..F).step_by(397) {
            let col: Vec<f64> = rows.iter().flat_map(|m| (0..R).map(move |r| m[r * F + f] as f64)).collect();
            let n = col.len() as f64;
            let mu = col.iter().sum::<f64>() / n;
            let v = col.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n;
            check_scalar("accum mean", mean[f], mu, 1e-5);
            check_scalar("accum var", var[f], v, 1e-4);
        }
    }

    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).build().unwrap();
    h.case("meanvar/feat_push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for i in 0..b {
            acc.push(&data[i % distinct]).unwrap();
        }
        acc.mean().unwrap().eval().unwrap();
        acc.var().unwrap().eval().unwrap();
    });

    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(4).sum().min_max().build().unwrap();
    h.case("meanvar/feat_moments4_minmax_push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for i in 0..b {
            acc.push(&data[i % distinct]).unwrap();
        }
        acc.kurtosis().unwrap().eval().unwrap();
        acc.max().unwrap().eval().unwrap();
    });

    if h.want("meanvar/feat_concat") {
        let parts: Vec<Array<f32>> = (0..b).map(|i| data[i % distinct].clone()).collect();
        let all = concat_slice(1, &parts);
        all.eval().unwrap();
        h.case("meanvar/feat_concat", &size, Work::Bytes(bytes), &mut || {
            all.mean_axis(1).eval().unwrap();
            all.var_axis(1).eval().unwrap();
        });
        h.case("meanvar/feat_concat_build", &size, Work::Bytes(bytes), &mut || {
            let all = concat_slice(1, &parts);
            all.mean_axis(1).eval().unwrap();
            all.var_axis(1).eval().unwrap();
        });
    }

    // The whole batch as one stream.
    let acc = Accumulator::<Df32>::builder(Fold::all()).moments(2).sum().build().unwrap();
    h.case("meanvar/whole_push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for i in 0..b {
            acc.push(&data[i % distinct]).unwrap();
        }
        acc.mean().unwrap().eval().unwrap();
        acc.var().unwrap().eval().unwrap();
    });

    // A stream of many small pushes: the push is mostly launch and host cost.
    let small: Vec<Array<f32>> = (0..8).map(|i| up(h, &values(256 * 64, 400 + i, -1.0, 1.0), [256, 64])).collect();
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).build().unwrap();
    h.case("meanvar/small_push", "1000x256x64", Work::Items(1000.0, "Mpush/s"), &mut || {
        acc.reset();
        for i in 0..1000 {
            acc.push(&small[i % 8]).unwrap();
        }
        acc.var().unwrap().eval().unwrap();
    });

    // Ten thousand batches of [256 rows, 4096 features] (full runs on a GPU only).
    if !h.quick && !h.ctx.is_cpu() {
        let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).build().unwrap();
        let n = 10_000;
        h.case("meanvar/feat_push_10k", &format!("{n}x{F}x{R}"), Work::Bytes((n * F * R * 4) as f64), &mut || {
            acc.reset();
            for i in 0..n {
                acc.push(&data[i % distinct]).unwrap();
            }
            acc.mean().unwrap().eval().unwrap();
            acc.var().unwrap().eval().unwrap();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// Feature covariance
// -------------------------------------------------------------------------------------------------

fn cov_cases(h: &mut Harness) {
    const F: usize = 512;
    const R: usize = 1024;
    let b = if h.quick { 8 } else { 32 };
    let distinct = b.min(8);
    let data = batches(h, distinct, [F, R], 500, 0.0, 1.0);
    let size = format!("{b}x{F}x{R}");
    let bytes = (b * F * R * 4) as f64;
    let parts: Vec<Array<f32>> = (0..b).map(|i| data[i % distinct].clone()).collect();

    // Verification against the concatenated data.
    {
        let acc = Covariance::<Df32>::new();
        for p in &parts[..3] {
            acc.push(p).unwrap();
        }
        let all = concat_slice(1, &parts[..3]);
        let n = (3 * R) as f32;
        let d = &all - &all.mean_axis(1);
        let want = matmul_f32(&d.transpose(), &d) / n;
        let got: Vec<f32> = acc.cov().unwrap().to_f32().to_vec().unwrap();
        check("accum cov", &got, &want.to_vec().unwrap(), 2e-3, 2e-5);
    }

    let acc = Covariance::<Df32>::new();
    h.case("cov/push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for p in &parts {
            acc.push(p).unwrap();
        }
        acc.cov().unwrap().eval().unwrap();
    });

    if h.want("cov/concat") {
        let all = concat_slice(1, &parts);
        all.eval().unwrap();
        let n = (b * R) as f32;
        h.case("cov/concat", &size, Work::Bytes(bytes), &mut || {
            let d = &all - &all.mean_axis(1);
            (matmul_f32(&d.transpose(), &d) / n).eval().unwrap();
        });
        h.case("cov/concat_build", &size, Work::Bytes(bytes), &mut || {
            let all = concat_slice(1, &parts);
            let d = &all - &all.mean_axis(1);
            (matmul_f32(&d.transpose(), &d) / n).eval().unwrap();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// Histograms
// -------------------------------------------------------------------------------------------------

fn hist_cases(h: &mut Harness) {
    const N: usize = 1 << 20;
    let b = if h.quick { 8 } else { 32 };
    let distinct = b.min(8);
    let xs = batches(h, distinct, [N, 1], 600, -0.1, 1.1);
    let ys: Vec<Array<f32>> = xs.iter().map(|x| (x * 0.5 + 0.25).sin().abs()).collect();
    let size = format!("{b}x{N}");
    let bytes = (b * N * 4) as f64;
    let xparts: Vec<Array<f32>> = (0..b).map(|i| xs[i % distinct].clone()).collect();
    let yparts: Vec<Array<f32>> = (0..b).map(|i| ys[i % distinct].clone()).collect();

    // Verification: exact counts of the first two batches.
    {
        let hist = Histogram::<Df32>::new(64, 0.0, 1.0).unwrap();
        hist.push(&xs[0]).unwrap();
        hist.push(&xs[1]).unwrap();
        let mut want = vec![0.0f64; 64];
        for batch in [&xs[0], &xs[1]] {
            for v in batch.to_vec().unwrap() {
                if (0.0..=1.0).contains(&v) {
                    want[((v * 64.0) as usize).min(63)] += 1.0;
                }
            }
        }
        assert_eq!(host(&hist.counts().unwrap()), want, "accum hist");
    }

    for bins in [64usize, 4096] {
        let hist = Histogram::<Df32>::new(bins, 0.0, 1.0).unwrap();
        h.case(&format!("hist/{bins}bins_push"), &size, Work::Bytes(bytes), &mut || {
            hist.reset();
            for p in &xparts {
                hist.push(p).unwrap();
            }
            hist.counts().unwrap().eval().unwrap();
        });
    }
    if h.want("hist/64bins_concat") {
        let all = concat_slice(0, &xparts);
        all.eval().unwrap();
        h.case("hist/64bins_concat", &size, Work::Bytes(bytes), &mut || all.histogram(64, 0.0, 1.0).eval().unwrap());
        h.case("hist/64bins_concat_build", &size, Work::Bytes(bytes), &mut || concat_slice(0, &xparts).histogram(64, 0.0, 1.0).eval().unwrap());
    }

    let joint = Histogram2d::<Df32>::new((32, 32), (0.0, 1.0), (0.0, 1.0)).unwrap();
    h.case("hist/2d_32x32_push", &size, Work::Bytes(2.0 * bytes), &mut || {
        joint.reset();
        for (x, y) in xparts.iter().zip(&yparts) {
            joint.push(x, y).unwrap();
        }
        joint.mutual_information().unwrap().eval().unwrap();
    });
    if h.want("hist/2d_32x32_concat") {
        let (ax, ay) = (concat_slice(0, &xparts), concat_slice(0, &yparts));
        ax.eval().unwrap();
        ay.eval().unwrap();
        h.case("hist/2d_32x32_concat", &size, Work::Bytes(2.0 * bytes), &mut || {
            let bx = (&ax * 32.0).to_u32().minimum(31);
            let by = (&ay * 32.0).to_u32().minimum(31);
            bincount(&(&bx + &(&by * 32u32)), 1024).eval().unwrap();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// Top-k
// -------------------------------------------------------------------------------------------------

fn topk_cases(h: &mut Harness) {
    const N: usize = 1 << 20;
    const K: usize = 100;
    let b = if h.quick { 8 } else { 32 };
    let distinct = b.min(8);
    let xs = batches(h, distinct, [N, 1], 700, 0.0, 1.0);
    let parts: Vec<Array<f32>> = (0..b).map(|i| xs[i % distinct].clone()).collect();
    let size = format!("{b}x{N}");
    let bytes = (b * N * 4) as f64;
    {
        let top = TopK::<f32, Df32>::largest(K);
        top.push(&xs[0]).unwrap();
        top.push(&xs[1]).unwrap();
        let all = concat_slice(0, &[xs[0].clone(), xs[1].clone()]);
        let (values, idx) = all.topk(K);
        check("accum topk", &top.values().unwrap().to_vec().unwrap(), &values.to_vec().unwrap(), 0.0, 0.0);
        let want: Vec<f64> = idx.to_vec().unwrap().into_iter().map(f64::from).collect();
        assert_eq!(TopK::<f32, Df32>::to_f64(&top.indices().unwrap()).unwrap(), want, "accum topk indices");
    }
    let top = TopK::<f32, Df32>::largest(K);
    h.case("topk/k100_push", &size, Work::Bytes(bytes), &mut || {
        top.reset();
        for p in &parts {
            top.push(p).unwrap();
        }
        top.values().unwrap().eval().unwrap();
    });
    if h.want("topk/k100_concat") {
        let all = concat_slice(0, &parts);
        all.eval().unwrap();
        h.case("topk/k100_concat", &size, Work::Bytes(bytes), &mut || {
            let (v, i) = all.topk(K);
            v.eval().unwrap();
            i.eval().unwrap();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// Per-group states
// -------------------------------------------------------------------------------------------------

fn group_cases(h: &mut Harness) {
    const F: usize = 64;
    const R: usize = 4096;
    const G: usize = 10;
    let b = if h.quick { 8 } else { 32 };
    let distinct = b.min(8);
    let data = batches(h, distinct, [F, R], 800, -1.0, 1.0);
    let labels: Vec<Array<u32>> = (0..distinct).map(|i| Array::from_slice_on(&h.ctx, &ints(R, 900 + i as u64, G as u32), R).unwrap()).collect();
    let size = format!("{b}x{F}x{R}");
    let bytes = (b * F * R * 4) as f64;
    {
        let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).groups(G).build().unwrap();
        acc.push_grouped(&data[0], &labels[0]).unwrap();
        let (mean, count) = (host(&acc.mean().unwrap()), host(&acc.count().unwrap()));
        let (x, l) = (data[0].to_vec().unwrap(), labels[0].to_vec().unwrap());
        for g in 0..G {
            let rows: Vec<usize> = (0..R).filter(|&r| l[r] == g as u32).collect();
            let want = rows.iter().map(|&r| x[r * F + 3] as f64).sum::<f64>() / rows.len() as f64;
            check_scalar("accum group mean", mean[3 + F * g], want, 1e-4);
            assert_eq!(count[3 + F * g], rows.len() as f64);
        }
    }
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).groups(G).build().unwrap();
    h.case("group/meanvar_push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for i in 0..b {
            acc.push_grouped(&data[i % distinct], &labels[i % distinct]).unwrap();
        }
        acc.mean().unwrap().eval().unwrap();
        acc.var().unwrap().eval().unwrap();
    });
    // Masked pushes of a plain accumulator: the elements below zero.
    let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).min_max().build().unwrap();
    let masks: Vec<Array<bool>> = data.iter().map(|d| d.gt(0.0)).collect();
    h.case("group/masked_meanvar_push", &size, Work::Bytes(bytes), &mut || {
        acc.reset();
        for i in 0..b {
            acc.push_where(&data[i % distinct], &masks[i % distinct]).unwrap();
        }
        acc.var().unwrap().eval().unwrap();
        acc.max().unwrap().eval().unwrap();
    });
}
