//! Case 13: one SGD step (forward, softmax cross-entropy, backward, update) of a 784-512-512-10
//! MLP with batch 256, written the way each library is normally used.
//!
//! Matrices are row-major on the host; a `rows x cols` matrix is `[cols, rows]` in both
//! libraries. With `A_af` the array of a row-major matrix, the row-major product
//! `op(A) * op(B)` is `matmul(B_af, A_af)` with the transposition flags swapped in position
//! (`rmm` below).

use arrayfire as af;

use crate::afx;
use crate::bench::Bench;
use crate::cases::{A, L};
use crate::lumax;
use crate::util::*;

const BATCH: usize = 256;
const D: [usize; 4] = [784, 512, 512, 10];
const LR: f32 = 0.5;

/// Parameter shapes as `[cols, rows]`: `w1, b1, w2, b2, w3, b3`.
fn shapes() -> [Vec<usize>; 6] {
    [vec![D[1], D[0]], vec![D[1]], vec![D[2], D[1]], vec![D[2]], vec![D[3], D[2]], vec![D[3]]]
}

struct Host {
    params: Vec<Vec<f32>>,
    x: Vec<f32>,
    onehot: Vec<f32>,
}

fn host_data() -> Host {
    let params = shapes()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n: usize = s.iter().product();
            if i % 2 == 0 {
                let r = (2.0 / s[1] as f32).sqrt();
                uniform(n, 200 + i as u64, -r, r)
            } else {
                uniform(n, 200 + i as u64, -0.1, 0.1)
            }
        })
        .collect();
    let x = uniform(BATCH * D[0], 210, 0.0, 1.0);
    let labels = indices(BATCH, 211, D[3] as u32);
    let mut onehot = vec![0.0f32; D[3] * BATCH];
    for (b, &l) in labels.iter().enumerate() {
        onehot[l as usize + D[3] * b] = 1.0;
    }
    Host { params, x, onehot }
}

type Mm = fn(&L, &L) -> L;

/// One step on luma: the six new parameters and the loss.
fn step_luma(mm: Mm, p: &[L], x: &L, y: &L) -> Vec<L> {
    let z1 = &mm(x, &p[0]) + &p[1];
    let h1 = z1.maximum(0.0f32);
    let z2 = &mm(&h1, &p[2]) + &p[3];
    let h2 = z2.maximum(0.0f32);
    let logits = &mm(&h2, &p[4]) + &p[5];
    let m = logits.max_axis(0);
    let e = (&logits - &m).exp();
    let s = e.sum_axis(0);
    let inv = 1.0 / BATCH as f32;
    let loss = &((&m + &s.ln()).sum() - (&logits * y).sum()) * inv;
    let dl = &(&(&e / &s) - y) * inv;
    let dw3 = mm(&h2.transpose(), &dl);
    let db3 = dl.sum_axis(1);
    let dz2 = luma::select(&z2.gt(0.0f32), mm(&dl, &p[4].transpose()), 0.0f32);
    let dw2 = mm(&h1.transpose(), &dz2);
    let db2 = dz2.sum_axis(1);
    let dz1 = luma::select(&z1.gt(0.0f32), mm(&dz2, &p[2].transpose()), 0.0f32);
    let dw1 = mm(&x.transpose(), &dz1);
    let db1 = dz1.sum_axis(1);
    let g = [dw1, db1, dw2, db2, dw3, db3];
    let mut out: Vec<L> = p.iter().zip(&g).map(|(p, g)| p - &(g * LR)).collect();
    out.push(loss);
    out
}

/// `op(a) * op(b)` of row-major matrices held as ArrayFire arrays.
fn rmm(a: &A, ta: bool, b: &A, tb: bool) -> A {
    let prop = |t: bool| if t { af::MatProp::TRANS } else { af::MatProp::NONE };
    af::matmul(b, a, prop(tb), prop(ta))
}

fn step_af(p: &[A], x: &A, y: &A) -> Vec<A> {
    let add = |a: &A, b: &A| af::add(a, b, true);
    let relu = |a: &A| af::maxof(a, &af::constant(0.0f32, a.dims()), false);
    let z1 = add(&rmm(x, false, &p[0], false), &p[1]);
    let h1 = relu(&z1);
    let z2 = add(&rmm(&h1, false, &p[2], false), &p[3]);
    let h2 = relu(&z2);
    let logits = add(&rmm(&h2, false, &p[4], false), &p[5]);
    let m = af::max(&logits, 0);
    let e = af::exp(&af::sub(&logits, &m, true));
    let s = af::sum(&e, 0);
    let inv = 1.0 / BATCH as f32;
    let lse = af::sum(&af::flat(&af::add(&m, &af::log(&s), false)), 0);
    let picked = af::sum(&af::flat(&(&logits * y)), 0);
    let loss = &(&lse - &picked) * inv;
    let dl = &af::sub(&af::div(&e, &s, true), y, false) * inv;
    let dw3 = rmm(&h2, true, &dl, false);
    let db3 = af::sum(&dl, 1);
    let zeros = |a: &A| af::constant(0.0f32, a.dims());
    let dz2 = {
        let g = rmm(&dl, false, &p[4], true);
        af::select(&g, &af::gt(&z2, &0.0f32, false), &zeros(&g))
    };
    let dw2 = rmm(&h1, true, &dz2, false);
    let db2 = af::sum(&dz2, 1);
    let dz1 = {
        let g = rmm(&dz2, false, &p[2], true);
        af::select(&g, &af::gt(&z1, &0.0f32, false), &zeros(&g))
    };
    let dw1 = rmm(x, true, &dz1, false);
    let db1 = af::sum(&dz1, 1);
    let g = [dw1, db1, dw2, db2, dw3, db3];
    let mut out: Vec<A> = p.iter().zip(&g).map(|(p, g)| p - &(g * LR)).collect();
    out.push(loss);
    out
}

pub fn mlp(b: &mut Bench) {
    b.guarded("mlp", |b| {
        let ctx = b.ctx.clone();
        let h = host_data();
        let lshape = |s: &[usize]| luma::Shape::from(if s.len() == 1 { [s[0], 1] } else { [s[0], s[1]] });
        let pl: Vec<L> = h.params.iter().zip(shapes()).map(|(v, s)| lumax::arr(&ctx, v, lshape(&s))).collect();
        let xl = lumax::arr(&ctx, &h.x, [D[0], BATCH]);
        let yl = lumax::arr(&ctx, &h.onehot, [D[3], BATCH]);
        let dims = |s: &Vec<usize>| if s.len() == 1 { vec![s[0] as u64, 1] } else { s.iter().map(|&d| d as u64).collect() };
        let pa: Vec<A> = h.params.iter().zip(shapes()).map(|(v, s)| afx::arr(v, &dims(&s))).collect();
        let xa = afx::arr(&h.x, &[D[0] as u64, BATCH as u64]);
        let ya = afx::arr(&h.onehot, &[D[3] as u64, BATCH as u64]);
        let (tf, _) = crate::cases::probe_af_tf32();

        let mut f32_step: Vec<Vec<f32>> = Vec::new();
        for tf32 in [false, true] {
            let mm: Mm = if tf32 { luma::matmul_tf32 } else { luma::matmul_f32 };
            let mode = if tf32 { tf.unwrap_or(afx::CUBLAS_DEFAULT_MATH) } else { afx::CUBLAS_DEFAULT_MATH };
            afx::set_math_mode(mode).expect("set cuBLAS math mode");
            let lo = step_luma(mm, &pl, &xl, &yl);
            let ao = step_af(&pa, &xa, &ya);
            let flat_l: Vec<Vec<f32>> = lo.iter().map(lumax::host).collect();
            let flat_a: Vec<Vec<f32>> = ao.iter().map(afx::host).collect();
            if !tf32 {
                f32_step = flat_a.clone();
            }
            // TF32 rounds the operands to 10 bits; a ReLU input within that error of zero may fall on either
            // side in the two libraries, which moves single gradient entries by up to ~1e-3 * lr.
            let (abs, loss_rel) = if tf32 { (5e-4, 2e-3) } else { (1e-5, 1e-4) };
            let names = ["w1", "b1", "w2", "b2", "w3", "b3"];
            let mut v = compare(&flat_l[6], &flat_a[6], loss_rel, 0.0);
            let mut moved = 0.0f64;
            for i in 0..6 {
                let vi = compare(&flat_l[i], &flat_a[i], 1e-3, abs);
                if !vi.ok {
                    v = v.and(Verdict::no(format!("{}: {}", names[i], vi.text)));
                } else if i == 0 {
                    v = v.and(vi);
                }
                let delta: Vec<f32> = flat_l[i].iter().zip(&h.params[i]).map(|(a, b)| a - b).collect();
                moved = moved.max(rms(&delta));
            }
            if !(moved > 1e-4) {
                v = v.and(Verdict::no(format!("NO: the step barely moved the parameters (rms {moved:.1e})")));
            } else {
                v = v.and(Verdict::yes(format!("all 6 params + loss compared; luma loss {:.4}, AF {:.4}; rms step {moved:.1e}", flat_l[6][0], flat_a[6][0])));
                if tf32 {
                    let dev = |x: &[Vec<f32>]| (0..6).map(|i| x[i].iter().zip(&f32_step[i]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
                    v = v.and(Verdict::yes(format!("max deviation of the TF32 step from the AF f32 step: luma {:.1e}, AF {:.1e}", dev(&flat_l), dev(&flat_a))));
                }
            }
            let note = if tf32 && tf.is_none() {
                "DIFFERENT PRECISION: luma TF32 vs ArrayFire f32 (no TF32 mode in ArrayFire)"
            } else if tf32 {
                "both TF32 where eligible (luma: dims that are multiples of 4 and >= 32; the 10-class products stay f32)"
            } else {
                "both f32 (luma matmul_f32, AF sgemm)"
            };
            let name = if tf32 { "13 MLP SGD step, TF32" } else { "13 MLP SGD step, f32" };
            b.run(
                name,
                "784-512-512-10, batch 256",
                v,
                note,
                &mut || {
                    let out = step_luma(mm, &pl, &xl, &yl);
                    let refs: Vec<&L> = out.iter().collect();
                    luma::eval_all(&refs).unwrap();
                    ctx.sync().unwrap();
                },
                &mut || {
                    let out = step_af(&pa, &xa, &ya);
                    // eval_multiple needs equal sizes; the loss and the parameters differ.
                    out.iter().for_each(|a| a.eval());
                    afx::sync();
                },
            );
        }
        afx::set_math_mode(afx::CUBLAS_DEFAULT_MATH).expect("reset cuBLAS math mode");
    });
}
