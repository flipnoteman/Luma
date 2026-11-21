//! Scaled dot-product attention: `softmax(Q K^T / sqrt(d)) V` for one head over several
//! sequence lengths (plain and causal), many heads, and a multi-head transformer block (the
//! projections, head slices as lazy views, concatenation, output projection, residual).
//!
//! Row-major `T x d` matrices are luma arrays `[d, T]`; the score matrix is `[keys, queries]`,
//! so softmax runs along axis 0.

use super::common::*;
use crate::harness::*;
use luma::*;

/// `softmax(Q K^T * scale (+ causal mask)) V` for `q, k, v` of shape `[d, T]`.
fn attention(q: &Array<f32>, k: &Array<f32>, v: &Array<f32>, scale: f32, causal: bool) -> Array<f32> {
    let s = &matmul(q, &k.transpose()) * scale;
    let s = if causal { select(&s.index(0).gt(s.index(1)), f32::NEG_INFINITY, &s) } else { s };
    let e = (&s - &s.max_axis(0)).exp();
    let p = &e / &e.sum_axis(0);
    matmul(&p, v)
}

/// Host attention output rows `rows` (queries), for row-major `T x d` `[q, k, v]`.
fn host_attention([q, k, v]: [&[f32]; 3], t: usize, d: usize, scale: f64, causal: bool, rows: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * d);
    for &r in rows {
        let keys = if causal { r + 1 } else { t };
        let scores: Vec<f32> = (0..keys).map(|j| (scale * (0..d).map(|c| f64::from(q[r * d + c]) * f64::from(k[j * d + c])).sum::<f64>()) as f32).collect();
        let (probs, _) = softmax_row(&scores);
        out.extend((0..d).map(|c| probs.iter().enumerate().map(|(j, p)| p * f64::from(v[j * d + c])).sum::<f64>()));
    }
    out
}

fn flops_of(t: usize, d: usize) -> f64 {
    4.0 * (t * t) as f64 * d as f64
}

/// One head, `T` tokens, head width `d`.
fn single(h: &mut Harness, t: usize, d: usize, causal: bool) {
    let flops = flops_of(t, d);
    if !fits(h, flops) {
        return;
    }
    let case = if causal { "attention/causal" } else { "attention/single" };
    let size = format!("T{t} d{d}");
    h.guarded(&format!("{case} {size}"), |h| {
        let (qs, ks, vs) = (values(t * d, 41, -1.0, 1.0), values(t * d, 42, -1.0, 1.0), values(t * d, 43, -1.0, 1.0));
        let (q, k, v) = (arr(h, &qs, [d, t]), arr(h, &ks, [d, t]), arr(h, &vs, [d, t]));
        let scale = 1.0 / (d as f32).sqrt();
        let rows = sample(t, 8);
        let want = host_attention([&qs, &ks, &vs], t, d, f64::from(scale), causal, &rows);
        let run = || attention(&q, &k, &v, scale, causal);
        check64(case, &gather_rows(&run().to_vec().unwrap(), d, &rows), &want, 2e-3, 2e-3);
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

/// `heads` independent heads, one attention each, recorded together and evaluated in one batch.
fn heads(h: &mut Harness, heads: usize, t: usize, d: usize) {
    let flops = heads as f64 * flops_of(t, d);
    if !fits(h, flops) {
        return;
    }
    let case = "attention/heads";
    let size = format!("H{heads} T{t} d{d}");
    h.guarded(&format!("{case} {size}"), |h| {
        let per = t * d;
        let (qs, ks, vs) = (values(heads * per, 44, -1.0, 1.0), values(heads * per, 45, -1.0, 1.0), values(heads * per, 46, -1.0, 1.0));
        // One array of all heads, `[d, T, H]`, sliced per head: the slice is a lazy view.
        let (q, k, v) = (arr(h, &qs, [d, t, heads]), arr(h, &ks, [d, t, heads]), arr(h, &vs, [d, t, heads]));
        let scale = 1.0 / (d as f32).sqrt();
        let run = || -> Vec<Array<f32>> { (0..heads).map(|i| attention(&q.slice(2, i..i + 1), &k.slice(2, i..i + 1), &v.slice(2, i..i + 1), scale, false)).collect() };
        let outs = run();
        let rows = sample(t, 4);
        for i in [0, heads - 1] {
            let (a, b, c) = (&qs[i * per..][..per], &ks[i * per..][..per], &vs[i * per..][..per]);
            let want = host_attention([a, b, c], t, d, f64::from(scale), false, &rows);
            check64(case, &gather_rows(&outs[i].to_vec().unwrap(), d, &rows), &want, 2e-3, 2e-3);
        }
        h.case(case, &size, Work::Flops(flops), &mut || {
            let outs = run();
            let refs: Vec<&Array<f32>> = outs.iter().collect();
            eval_all(&refs).unwrap();
        });
    });
}

/// A multi-head self-attention block with residual: `x + concat_h(attention_h) Wo`.
fn block(h: &mut Harness, t: usize, dm: usize, nheads: usize) {
    let dh = dm / nheads;
    let flops = 8.0 * t as f64 * (dm * dm) as f64 + nheads as f64 * flops_of(t, dh);
    if !fits(h, flops) {
        return;
    }
    let case = "attention/block";
    let size = format!("T{t} dm{dm} H{nheads}");
    h.guarded(case, |h| {
        let xs = values(t * dm, 47, -1.0, 1.0);
        let s = 1.0 / (dm as f32).sqrt();
        let ws: Vec<Vec<f32>> = (0..4).map(|i| values(dm * dm, 48 + i, -s, s)).collect();
        let x = arr(h, &xs, [dm, t]);
        let w: Vec<Array<f32>> = ws.iter().map(|w| arr(h, w, [dm, dm])).collect();
        let scale = 1.0 / (dh as f32).sqrt();
        let run = || {
            let (q, k, v) = (matmul(&x, &w[0]), matmul(&x, &w[1]), matmul(&x, &w[2]));
            let outs: Vec<Array<f32>> = (0..nheads)
                .map(|i| {
                    let r = i * dh..(i + 1) * dh;
                    attention(&q.slice(0, r.clone()), &k.slice(0, r.clone()), &v.slice(0, r), scale, true)
                })
                .collect();
            let cat = concat_slice(0, &outs);
            &matmul(&cat, &w[3]) + &x
        };
        // Host: the same block, head by head.
        let (q, k, v) = (gemm(&xs, &ws[0], t, dm, dm), gemm(&xs, &ws[1], t, dm, dm), gemm(&xs, &ws[2], t, dm, dm));
        let all: Vec<usize> = (0..t).collect();
        let mut cat = vec![0.0f32; t * dm];
        for i in 0..nheads {
            let head = |m: &[f32]| -> Vec<f32> { (0..t).flat_map(|r| m[r * dm + i * dh..][..dh].iter().copied()).collect() };
            let o = host_attention([&head(&q), &head(&k), &head(&v)], t, dh, f64::from(scale), true, &all);
            for r in 0..t {
                for c in 0..dh {
                    cat[r * dm + i * dh + c] = o[r * dh + c] as f32;
                }
            }
        }
        let want: Vec<f32> = gemm(&cat, &ws[3], t, dm, dm).iter().zip(&xs).map(|(a, b)| a + b).collect();
        check(case, &run().to_vec().unwrap(), &want, 5e-3, 5e-3);
        h.case(case, &size, Work::Flops(flops), &mut || run().eval().unwrap());
    });
}

pub fn run(h: &mut Harness) {
    for t in [128, 512, 2048, 4096] {
        single(h, t, 64, false);
    }
    single(h, 1024, 128, false);
    for t in [512, 2048] {
        single(h, t, 64, true);
    }
    heads(h, 8, 512, 64);
    heads(h, 16, 128, 64);
    block(h, 256, 512, 8);
    block(h, 1024, 768, 12);
}
