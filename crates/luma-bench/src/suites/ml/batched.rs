//! Many small matrix products: `matmul_batched` (one launch for the batch, `[K, M, B]` times
//! `[N, K, B]`, or `[N, K]` shared by every product), against the ways of doing it without: a
//! broadcast multiply fused into a reduction (`[K, M, 1, B] * [K, 1, N, B]` summed over axis 0,
//! one kernel, nothing stored) and a loop of `matmul`, one launch each.

use super::common::*;
use crate::harness::*;
use luma::*;

/// `batch` products of `m x k` by `k x n`, as one fused broadcast reduction. `a` is stored
/// batch-major `m x k` per batch, `bt` batch-major `n x k` per batch (the transposed right
/// operand, so that the reduced axis is contiguous in both); the result is `[1, m, n, batch]`.
fn broadcast(h: &mut Harness, batch: usize, m: usize) {
    let batch = size(h, batch / 4, batch, batch / 4);
    let (k, n) = (m, m);
    let flops = 2.0 * (batch * m * n * k) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "batched/broadcast_reduce";
    let size = format!("{batch}x{m}x{k}x{n}");
    h.guarded(&format!("{case} {size}"), |h| {
        let a = values(batch * m * k, 81, -1.0, 1.0);
        let bt = values(batch * n * k, 82, -1.0, 1.0);
        let (aa, bb) = (arr(h, &a, [k, m, 1, batch]), arr(h, &bt, [k, 1, n, batch]));
        let run = || (&aa * &bb).sum_axis(0);
        let got = run().to_vec().unwrap();
        for b in sample(batch, 6) {
            for (mi, ni) in [(0, 0), (m - 1, n - 1), (m / 2, n / 3)] {
                let want: f64 = (0..k).map(|ki| f64::from(a[(b * m + mi) * k + ki]) * f64::from(bt[(b * n + ni) * k + ki])).sum();
                check_scalar(case, f64::from(got[mi + m * (ni + n * b)]), want, 1e-3);
            }
        }
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

/// The same products as a loop of `matmul` over pre-split operands, evaluated in one batch.
fn looped(h: &mut Harness, batch: usize, m: usize) {
    let batch = size(h, batch / 2, batch, batch / 2);
    let (k, n) = (m, m);
    let flops = 2.0 * (batch * m * n * k) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = "batched/matmul_loop";
    let size = format!("{batch}x{m}x{k}x{n}");
    h.guarded(&format!("{case} {size}"), |h| {
        let a = values(batch * m * k, 83, -1.0, 1.0);
        let b = values(batch * k * n, 84, -1.0, 1.0);
        let aa: Vec<Array<f32>> = (0..batch).map(|i| arr(h, &a[i * m * k..][..m * k], [k, m])).collect();
        let bb: Vec<Array<f32>> = (0..batch).map(|i| arr(h, &b[i * k * n..][..k * n], [n, k])).collect();
        let run = || -> Vec<Array<f32>> { aa.iter().zip(&bb).map(|(x, w)| matmul(x, w)).collect() };
        let outs = run();
        for i in sample(batch, 4) {
            let want = gemm(&a[i * m * k..][..m * k], &b[i * k * n..][..k * n], m, k, n);
            check(case, &outs[i].to_vec().unwrap(), &want, 1e-3, 1e-3);
        }
        h.case(case, &size, Work::Flops(flops), &mut || {
            let outs = run();
            let refs: Vec<&Array<f32>> = outs.iter().collect();
            eval_all(&refs).unwrap();
        });
    });
}

/// The same products as one `matmul_batched`. `shared`: the right operand is one `k x n` matrix
/// for every product (the weights of a layer), else a matrix per product.
fn batched_matmul(h: &mut Harness, batch: usize, m: usize, shared: bool) {
    let batch = size(h, batch / 4, batch, batch / 4);
    let (k, n) = (m, m);
    let flops = 2.0 * (batch * m * n * k) as f64;
    if !fits(h, flops) {
        return;
    }
    let case = if shared { "batched/matmul_batched_shared" } else { "batched/matmul_batched" };
    let size = format!("{batch}x{m}x{k}x{n}");
    h.guarded(&format!("{case} {size}"), |h| {
        let a = values(batch * m * k, 85, -1.0, 1.0);
        let b_batch = if shared { 1 } else { batch };
        let b = values(b_batch * k * n, 86, -1.0, 1.0);
        let aa = arr(h, &a, [k, m, batch]);
        let bb = arr(h, &b, [n, k, b_batch]);
        let run = || aa.matmul_batched(&bb);
        let got = run().to_vec().unwrap();
        for i in sample(batch, 6) {
            let w = &b[if shared { 0 } else { i * k * n }..][..k * n];
            let want = gemm(&a[i * m * k..][..m * k], w, m, k, n);
            check(case, &got[i * m * n..][..m * n], &want, 1e-3, 1e-3);
        }
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    batched_matmul(h, 16384, 16, false);
    batched_matmul(h, 4096, 32, false);
    batched_matmul(h, 1024, 64, false);
    batched_matmul(h, 256, 32, false);
    batched_matmul(h, 64, 64, false);
    batched_matmul(h, 256, 128, false);
    batched_matmul(h, 4096, 32, true);
    batched_matmul(h, 1024, 64, true);
    broadcast(h, 16384, 16);
    broadcast(h, 4096, 32);
    broadcast(h, 1024, 64);
    looped(h, 256, 32);
    looped(h, 64, 64);
}
