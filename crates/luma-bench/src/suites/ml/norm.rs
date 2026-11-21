//! Row-wise operations of transformers and classifiers: softmax, logsumexp, cross-entropy,
//! layernorm, rmsnorm, argmax. Shapes are `[cols, rows]` (rows are contiguous lines, reduced
//! along axis 0) unless the case says strided (reduced along axis 1).

use super::common::*;
use crate::harness::*;
use luma::*;

/// Checks the sampled rows of `got` (row-major, `out_cols` wide per row) against `f(row)`.
#[allow(clippy::too_many_arguments)]
fn check_rows(name: &str, got: &[f32], x: &[f32], cols: usize, out_cols: usize, rows: usize, tol: (f32, f32), f: impl Fn(&[f32]) -> Vec<f64>) {
    let picked = sample(rows, 8);
    let want: Vec<f64> = picked.iter().flat_map(|&r| f(&x[r * cols..][..cols])).collect();
    check64(name, &gather_rows(got, out_cols, &picked), &want, tol.0, tol.1);
}

/// Column `j` of the row-major `rows x lines` matrix `x` (a line of an axis-1 reduction).
fn column(x: &[f32], lines: usize, j: usize) -> Vec<f32> {
    x.chunks(lines).map(|r| r[j]).collect()
}

/// Rows of a case: `quick` / `full` rows, fewer on the CPU.
fn rows_of(h: &Harness, quick: usize, full: usize) -> usize {
    size(h, quick, full, (full / 4).max(1))
}

fn layout(cols: usize, rows: usize) -> String {
    format!("{cols}x{rows}")
}

fn softmax(h: &mut Harness, cols: usize, full_rows: usize) {
    let rows = rows_of(h, (full_rows / 4).max(1), full_rows);
    let case = "norm/softmax";
    h.guarded(&format!("{case} {cols}"), |h| {
        let xs = values(cols * rows, 21, -8.0, 8.0);
        let x = arr(h, &xs, [cols, rows]);
        let run = || {
            let e = (&x - &x.max_axis(0)).exp();
            &e / &e.sum_axis(0)
        };
        check_rows(case, &run().to_vec().unwrap(), &xs, cols, cols, rows, (1e-3, 1e-6), |r| softmax_row(r).0);
        h.case(case, &layout(cols, rows), Work::Bytes(8.0 * (cols * rows) as f64), &mut || run().eval().unwrap());
    });
}

fn logsumexp(h: &mut Harness, cols: usize, full_rows: usize) {
    let rows = rows_of(h, (full_rows / 4).max(1), full_rows);
    let case = "norm/logsumexp";
    h.guarded(&format!("{case} {cols}"), |h| {
        let xs = values(cols * rows, 22, -8.0, 8.0);
        let x = arr(h, &xs, [cols, rows]);
        let run = || {
            let m = x.max_axis(0);
            &m + &(&x - &m).exp().sum_axis(0).ln()
        };
        check_rows(case, &run().to_vec().unwrap(), &xs, cols, 1, rows, (1e-5, 1e-5), |r| vec![softmax_row(r).1]);
        h.case(case, &layout(cols, rows), Work::Bytes(4.0 * (cols * rows) as f64), &mut || run().eval().unwrap());
    });
}

/// Softmax over axis 1: the lines are strided (a column of a row-major matrix).
fn softmax_strided(h: &mut Harness, lines: usize, n: usize) {
    let n = if h.ctx.is_cpu() { n / 4 } else { n };
    let case = "norm/softmax_axis1";
    h.guarded(case, |h| {
        let xs = values(lines * n, 23, -8.0, 8.0);
        let x = arr(h, &xs, [lines, n]);
        let run = || {
            let e = (&x - &x.max_axis(1)).exp();
            &e / &e.sum_axis(1)
        };
        let got = run().to_vec().unwrap();
        for j in sample(lines, 6) {
            let want = softmax_row(&column(&xs, lines, j)).0;
            check64(case, &column(&got, lines, j), &want, 1e-3, 1e-6);
        }
        h.case(case, &format!("{lines}x{n}"), Work::Bytes(8.0 * (lines * n) as f64), &mut || run().eval().unwrap());
    });
}

fn layernorm(h: &mut Harness, cols: usize, full_rows: usize, one_pass: bool) {
    let rows = rows_of(h, (full_rows / 4).max(1), full_rows);
    let case = if one_pass { "norm/layernorm_1pass" } else { "norm/layernorm" };
    h.guarded(&format!("{case} {cols}"), |h| {
        let eps = 1e-5f32;
        // Activations with a mean well off zero, as after a residual stream.
        let xs = values(cols * rows, 24, 0.0, 4.0);
        let gs = values(cols, 25, 0.5, 1.5);
        let bs = values(cols, 26, -0.5, 0.5);
        let (x, gamma, beta) = (arr(h, &xs, [cols, rows]), arr(h, &gs, cols), arr(h, &bs, cols));
        let n = cols as f32;
        let run = || {
            let (mean, var) = if one_pass {
                let [s, ss] = x.reduce_axis(0, [Reduction::Sum, Reduction::SumOfSquares]);
                let mean = &s / n;
                let var = &ss / n - &mean * &mean;
                (mean, var)
            } else {
                (x.mean_axis(0), x.var_axis(0))
            };
            &((&x - &mean) / &(var + eps).sqrt()) * &gamma + &beta
        };
        check_rows(case, &run().to_vec().unwrap(), &xs, cols, cols, rows, (2e-3, 2e-3), |r| {
            let mean = r.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64;
            let var = r.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / cols as f64;
            r.iter().enumerate().map(|(i, &v)| (f64::from(v) - mean) / (var + f64::from(eps)).sqrt() * f64::from(gs[i]) + f64::from(bs[i])).collect()
        });
        h.case(case, &layout(cols, rows), Work::Bytes(8.0 * (cols * rows) as f64), &mut || run().eval().unwrap());
    });
}

fn rmsnorm(h: &mut Harness, cols: usize, full_rows: usize) {
    let rows = rows_of(h, (full_rows / 4).max(1), full_rows);
    let case = "norm/rmsnorm";
    h.guarded(&format!("{case} {cols}"), |h| {
        let eps = 1e-6f32;
        let xs = values(cols * rows, 27, -2.0, 2.0);
        let gs = values(cols, 28, 0.5, 1.5);
        let (x, gamma) = (arr(h, &xs, [cols, rows]), arr(h, &gs, cols));
        let run = || {
            let ss = (&x * &x).sum_axis(0);
            let scale = 1.0 / (&ss / cols as f32 + eps).sqrt();
            &(&x * &scale) * &gamma
        };
        check_rows(case, &run().to_vec().unwrap(), &xs, cols, cols, rows, (2e-3, 2e-3), |r| {
            let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / cols as f64;
            r.iter().enumerate().map(|(i, &v)| f64::from(v) / (ms + f64::from(eps)).sqrt() * f64::from(gs[i])).collect()
        });
        h.case(case, &layout(cols, rows), Work::Bytes(8.0 * (cols * rows) as f64), &mut || run().eval().unwrap());
    });
}

fn argmax(h: &mut Harness, cols: usize, full_rows: usize) {
    let rows = rows_of(h, (full_rows / 4).max(1), full_rows);
    let case = "norm/argmax";
    h.guarded(&format!("{case} {cols}"), |h| {
        let xs = values(cols * rows, 29, -1.0, 1.0);
        let x = arr(h, &xs, [cols, rows]);
        let run = || x.argmax_axis(0);
        let got: Vec<f32> = run().to_vec().unwrap().iter().map(|&i| i as f32).collect();
        // The first of the equal maxima, as NumPy.
        check_rows(case, &got, &xs, cols, 1, rows, (0.0, 0.0), |r| {
            let best = r.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            vec![best as f64]
        });
        h.case(case, &layout(cols, rows), Work::Bytes(4.0 * (cols * rows) as f64), &mut || run().eval().unwrap());
    });
}

fn argmax_strided(h: &mut Harness, lines: usize, n: usize) {
    let n = if h.ctx.is_cpu() { n / 4 } else { n };
    let case = "norm/argmax_axis1";
    h.guarded(case, |h| {
        let xs = values(lines * n, 30, -1.0, 1.0);
        let x = arr(h, &xs, [lines, n]);
        let run = || x.argmax_axis(1);
        let got = run().to_vec().unwrap();
        for j in sample(lines, 6) {
            let col = column(&xs, lines, j);
            let best = col.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            assert_eq!(got[j] as usize, best, "{case}: column {j}");
        }
        h.case(case, &format!("{lines}x{n}"), Work::Bytes(4.0 * (lines * n) as f64), &mut || run().eval().unwrap());
    });
}

/// Softmax cross-entropy of logits `[classes, batch]` against labels: the mean loss as a host
/// value (a reduction and a readback, as a training loop logs it).
fn cross_entropy(h: &mut Harness, classes: usize, full_batch: usize) {
    let batch = rows_of(h, (full_batch / 4).max(1), full_batch);
    let case = "norm/cross_entropy";
    h.guarded(&format!("{case} {classes}"), |h| {
        let xs = values(classes * batch, 31, -4.0, 4.0);
        let labels = ints(batch, 32, classes as u32);
        let (x, y) = (arr(h, &xs, [classes, batch]), arr_u32(h, &labels, [1, batch]));
        let offsets = range(h, [1, batch], 1) * classes as u32;
        let run = || {
            let m = x.max_axis(0);
            let lse = &m + &(&x - &m).exp().sum_axis(0).ln();
            // The logit of the label: a gather at `label + classes * row`.
            let target = x.take(&(&y + &offsets));
            (&lse - &target).mean().item().unwrap()
        };
        let want: f64 = (0..batch).map(|r| softmax_row(&xs[r * classes..][..classes]).1 - f64::from(xs[r * classes + labels[r] as usize])).sum::<f64>() / batch as f64;
        check_scalar(case, f64::from(run()), want, 1e-4);
        h.case(case, &layout(classes, batch), Work::Bytes(4.0 * (classes * batch) as f64), &mut || {
            run();
        });
    });
}

/// The logsumexp of a whole array in double-float arithmetic (about 48 mantissa bits): `exp` and
/// `ln` of `Df32`. Axis reductions are not available for `Df32`, so this is a flat reduction.
fn logsumexp_df32(h: &mut Harness, n: usize) {
    let n = size(h, n / 4, n, n / 4);
    let case = "norm/logsumexp_df32";
    h.guarded(case, |h| {
        let xs = values(n, 33, -8.0, 8.0);
        let x = arr(h, &xs, n).to_df32();
        let run = || {
            let m = x.max();
            &(&x - &m).exp().sum().ln() + &m
        };
        let got = run().item().unwrap();
        let want = softmax_row(&xs).1;
        check_scalar(case, f64::from(got.hi) + f64::from(got.lo), want, 1e-9);
        h.case(case, &format!("{n}"), Work::Bytes(8.0 * n as f64), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    // Short rows (a class dimension) through long ones (a vocabulary).
    for (cols, rows) in [(64, 16384), (1024, 4096), (4096, 4096), (32768, 256), (10, 524288)] {
        softmax(h, cols, rows);
    }
    softmax_strided(h, 4096, 1024);
    for (cols, rows) in [(1024, 4096), (32768, 256)] {
        logsumexp(h, cols, rows);
    }
    for (cols, rows) in [(768, 8192), (4096, 1024), (64, 65536)] {
        layernorm(h, cols, rows, false);
    }
    layernorm(h, 768, 8192, true);
    for (cols, rows) in [(768, 8192), (4096, 1024)] {
        rmsnorm(h, cols, rows);
    }
    for (cols, rows) in [(1000, 4096), (10, 524288)] {
        argmax(h, cols, rows);
    }
    argmax_strided(h, 4096, 1024);
    cross_entropy(h, 1000, 4096);
    cross_entropy(h, 32000, 512);
    logsumexp_df32(h, 1 << 20);
}
