//! `svd`, `singular_values`, `pinv`, `rank` and `cond` against `f64` host references: sizes around
//! the workgroup-kernel and global-memory thresholds, tall / wide / square matrices, batches,
//! rank-deficient, ill-conditioned and repeated-value spectra, zero, identity, `NaN` and badly
//! scaled matrices. Checks: backward error `||A - U S Vt||_F / ||A||_F`, orthogonality of `U` and
//! `V`, order and accuracy of the singular values against a host one-sided Jacobi in `f64`.

#![allow(clippy::too_many_arguments)]

use crate::common::values;
use crate::dense_host::*;
use luma::{Array, Context};

const SIZES: [usize; 13] = [1, 2, 3, 4, 5, 7, 16, 31, 32, 33, 64, 100, 257];

pub fn mat(ctx: &Context, data: &[f32], cols: usize, rows: usize, batch: usize) -> Array<f32> {
    Array::from_slice_on(ctx, data, [cols, rows, batch]).unwrap()
}

/// Singular values (descending) of the `m x n` row-major `a` by one-sided Jacobi in `f64`.
pub fn host_singular_values(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let (rows, cols, mut w) = if m >= n {
        // Columns of `a` as vectors.
        let mut w = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                w[j * m + i] = a[i * n + j];
            }
        }
        (m, n, w)
    } else {
        // The columns of `aᵀ` are the rows of `a`.
        (n, m, a.to_vec())
    };
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let (mut app, mut aqq, mut apq) = (0.0, 0.0, 0.0);
                for i in 0..rows {
                    let (x, y) = (w[p * rows + i], w[q * rows + i]);
                    app += x * x;
                    aqq += y * y;
                    apq += x * y;
                }
                if apq.abs() <= 1e-15 * (app * aqq).sqrt() || apq == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (aqq - app) / (2.0 * apq);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..rows {
                    let (x, y) = (w[p * rows + i], w[q * rows + i]);
                    w[p * rows + i] = c * x - s * y;
                    w[q * rows + i] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut s: Vec<f64> = (0..cols).map(|j| (0..rows).map(|i| w[j * rows + i] * w[j * rows + i]).sum::<f64>().sqrt()).collect();
    s.sort_by(|x, y| y.total_cmp(x));
    s
}

/// A random orthogonal `n x n` matrix (row-major, `f64`) by Gram-Schmidt on random columns.
pub fn orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let r = values(n * n, seed, -1.0, 1.0);
    let mut q = vec![0.0f64; n * n];
    for j in 0..n {
        let mut v: Vec<f64> = (0..n).map(|i| f64::from(r[i * n + j])).collect();
        for _ in 0..2 {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| v[i] * q[i * n + k]).sum();
                for i in 0..n {
                    v[i] -= d * q[i * n + k];
                }
            }
        }
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..n {
            q[i * n + j] = v[i] / nv;
        }
    }
    q
}

/// `U diag(spec) Vᵀ` for random orthogonal `U` (`m x m`) and `V` (`n x n`), rounded to `f32`.
pub fn with_spectrum(m: usize, n: usize, spec: &[f64], seed: u64) -> Vec<f32> {
    let (u, v) = (orthogonal(m, seed), orthogonal(n, seed + 1000));
    let k = spec.len();
    assert!(k <= m.min(n));
    let mut a = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = (0..k).map(|p| u[i * m + p] * spec[p] * v[j * n + p]).sum::<f64>() as f32;
        }
    }
    a
}

pub fn random(m: usize, n: usize, seed: u64) -> Vec<f32> {
    values(m * n, seed, -1.0, 1.0)
}

/// What a decomposition of one `m x n` matrix got wrong.
#[derive(Debug)]
pub struct Errors {
    pub recon: f64,
    pub orth_u: f64,
    pub orth_v: f64,
    /// Largest `|s - s_ref|` over `s_ref[0]`.
    pub values: f64,
    /// Largest `|s - s_ref| / s_ref` over the values above `0.05 s_ref[0]`.
    pub rel_values: f64,
    pub sorted: bool,
}

/// `||Mᵀ M - I||_F` of the first `used` of the `cols` columns of the `rows x cols` row-major `m`.
fn orth_cols(m: &[f64], rows: usize, cols: usize, used: usize) -> f64 {
    let mut e = 0.0;
    for p in 0..used {
        for q in 0..used {
            let d: f64 = (0..rows).map(|i| m[i * cols + p] * m[i * cols + q]).sum();
            let want = if p == q { 1.0 } else { 0.0 };
            e += (d - want) * (d - want);
        }
    }
    e.sqrt()
}

/// Compares `u` (`m x k`), `s` (`k`), `vt` (`k x n`) with the matrix `a` (`m x n`).
pub fn errors(a: &[f32], m: usize, n: usize, u: &[f32], s: &[f32], vt: &[f32], lead: usize) -> Errors {
    let k = m.min(n);
    let (a, u, vt) = (widen(a), widen(u), widen(vt));
    let mut us = u.clone();
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] *= f64::from(s[j]);
        }
    }
    let rec = mul(&us, &vt, m, k, n);
    let recon = diff(&rec, &a) / fro(&a).max(f64::MIN_POSITIVE);
    let orth_u = orth_cols(&u, m, k, lead.min(k));
    let orth_v = orth_cols(&transpose(&vt, k, n), n, k, k);
    let want = host_singular_values(&a, m, n);
    let top = want[0].max(f64::MIN_POSITIVE);
    let mut values = 0.0f64;
    let mut rel_values = 0.0f64;
    for j in 0..k {
        let d = (f64::from(s[j]) - want[j]).abs();
        values = values.max(d / top);
        if want[j] > 0.05 * top {
            rel_values = rel_values.max(d / want[j]);
        }
    }
    let sorted = s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0);
    Errors { recon, orth_u, orth_v, values, rel_values, sorted }
}

/// Decomposes the `m x n` matrix (a batch `batch` of them in `data`) and checks every matrix.
pub fn check_batch(ctx: &Context, data: &[f32], m: usize, n: usize, batch: usize, label: &str) {
    check_lead(ctx, data, m, n, batch, usize::MAX, label);
}

/// [`check_batch`] where only the first `lead` columns of `u` have to be orthonormal (a big matrix
/// with many zero singular values leaves the others less orthogonal, see [`luma::Svd`]).
pub fn check_lead(ctx: &Context, data: &[f32], m: usize, n: usize, batch: usize, lead: usize, label: &str) {
    let k = m.min(n);
    let a = mat(ctx, data, n, m, batch);
    let svd = a.svd();
    assert_eq!(svd.u.shape().dims(), [k, m, batch, 1], "{label}: u shape");
    assert_eq!(svd.s.shape().dims(), [k, 1, batch, 1], "{label}: s shape");
    assert_eq!(svd.vt.shape().dims(), [n, k, batch, 1], "{label}: vt shape");
    let (u, s, vt) = (svd.u.to_vec().unwrap(), svd.s.to_vec().unwrap(), svd.vt.to_vec().unwrap());
    let single = a.singular_values().to_vec().unwrap();
    for b in 0..batch {
        let e = errors(&data[b * m * n..(b + 1) * m * n], m, n, &u[b * m * k..(b + 1) * m * k], &s[b * k..(b + 1) * k], &vt[b * k * n..(b + 1) * k * n], lead);
        let tol = 3e-6 * (m.max(n) as f64).sqrt().max(2.0);
        let orth = 1.5e-6 * (m.max(n) as f64).max(8.0);
        if b == 0 {
            eprintln!("{label} {m}x{n}: {e:?}");
        }
        assert!(e.recon < tol && e.orth_u < orth && e.orth_v < orth, "{label} ({m}x{n}) matrix {b}: {e:?}");
        assert!(e.values < tol && e.rel_values < tol * 4.0 && e.sorted, "{label} ({m}x{n}) matrix {b}: {e:?}");
        let sv = &single[b * k..(b + 1) * k];
        let top = f64::from(sv[0]).max(1e-30);
        for j in 0..k {
            assert!((f64::from(sv[j]) - f64::from(s[b * k + j])).abs() / top < tol, "{label} ({m}x{n}) matrix {b}: singular_values differ from svd at {j}: {} vs {}", sv[j], s[b * k + j]);
        }
    }
}

#[test]
fn svd_square_sizes() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        check_batch(&ctx, &random(n, n, 10 + n as u64), n, n, 1, "square");
    }
}

#[test]
fn svd_tall_and_wide() {
    let ctx = Context::new().unwrap();
    for &(m, n) in &[(2usize, 1usize), (1, 2), (5, 3), (3, 5), (8, 1), (1, 8), (40, 7), (7, 40), (33, 32), (32, 33), (64, 31), (31, 64), (100, 33), (33, 100), (300, 20), (20, 300), (130, 129), (129, 130)] {
        check_batch(&ctx, &random(m, n, 20 + (m * 3 + n) as u64), m, n, 1, "rect");
    }
}

#[test]
fn svd_batches() {
    let ctx = Context::new().unwrap();
    for &(m, n, batch) in &[(4usize, 4usize, 1usize), (4, 4, 3), (4, 4, 100), (4, 4, 5000), (8, 8, 700), (16, 16, 100), (32, 32, 40), (12, 5, 30), (5, 12, 30), (20, 20, 7), (33, 33, 3), (40, 17, 3), (70, 70, 2)] {
        let mut data = Vec::new();
        for b in 0..batch {
            data.extend(random(m, n, 5 + (m * 7 + n + b * 13) as u64));
        }
        check_batch(&ctx, &data, m, n, batch, "batch");
    }
}

#[test]
fn svd_rank_deficient() {
    let ctx = Context::new().unwrap();
    for &(m, n, r) in &[(4usize, 4usize, 2usize), (8, 8, 1), (16, 12, 5), (12, 16, 5), (31, 31, 10), (50, 50, 20), (100, 60, 30), (150, 150, 40)] {
        let spec: Vec<f64> = (0..r).map(|i| 1.0 + i as f64).collect();
        let a = with_spectrum(m, n, &spec, 7 + (m + n) as u64);
        check_lead(&ctx, &a, m, n, 1, if m.max(n) > 96 { r } else { usize::MAX }, &format!("rank {r}"));
        // The numerical rank is r.
        assert_eq!(mat(&ctx, &a, n, m, 1).rank().to_vec().unwrap(), vec![r as u32], "rank of {m}x{n} of rank {r}");
    }
}

#[test]
fn svd_exact_zero_values_complete_u() {
    let ctx = Context::new().unwrap();
    // A matrix with a zero column and a zero row, duplicated columns: exact zero singular values.
    for &n in &[3usize, 4, 8, 17, 32, 100, 130] {
        let mut a = random(n, n, 3);
        for i in 0..n {
            a[i * n + 1] = 0.0;
            a[i * n + 2.min(n - 1)] = a[i * n];
        }
        check_batch(&ctx, &a, n, n, 1, "zero column");
    }
    check_batch(&ctx, &[0.0; 16], 4, 4, 1, "zero 4x4");
    check_batch(&ctx, &[0.0], 1, 1, 1, "zero 1x1");
    check_batch(&ctx, &[0.0; 6 * 3], 6, 3, 1, "zero 6x3");
    // The same on the global-memory path: zero matrices (tall, wide, square), half of the columns zero.
    for &(m, n) in &[(100usize, 100usize), (120, 100), (100, 120)] {
        check_batch(&ctx, &vec![0.0; m * n], m, n, 1, "zero big");
    }
    for &(m, n) in &[(110usize, 110usize), (130, 105), (105, 130)] {
        let mut a = random(m, n, 9);
        for i in 0..m {
            for j in 0..n {
                if (m < n && i >= m / 2) || (m >= n && j >= n / 2) {
                    a[i * n + j] = 0.0;
                }
            }
        }
        check_batch(&ctx, &a, m, n, 1, "half zero big");
    }
}

#[test]
fn svd_ill_conditioned() {
    let ctx = Context::new().unwrap();
    for &n in &[4usize, 16, 32, 50, 100] {
        let spec: Vec<f64> = (0..n).map(|i| 10f64.powf(-6.0 * i as f64 / (n - 1).max(1) as f64)).collect();
        let a = with_spectrum(n, n, &spec, 11 + n as u64);
        let m = mat(&ctx, &a, n, n, 1);
        let svd = m.svd();
        let (u, s, vt) = (svd.u.to_vec().unwrap(), svd.s.to_vec().unwrap(), svd.vt.to_vec().unwrap());
        let e = errors(&a, n, n, &u, &s, &vt, usize::MAX);
        assert!(e.recon < 1e-5 && e.orth_u < 4e-7 * n as f64 + 1e-5 && e.orth_v < 4e-7 * n as f64 + 1e-5 && e.sorted, "n = {n}: {e:?}");
        // The small values come out to a few eps of the largest, absolutely (the host reference is of the rounded matrix).
        let want = host_singular_values(&widen(&a), n, n);
        for j in 0..n {
            let err = (f64::from(s[j]) - want[j]).abs() / want[0];
            assert!(err < 3e-7 * (n as f64).sqrt().max(2.0), "n = {n}: sigma {j} = {} want {} (error {err} of the largest)", s[j], want[j]);
        }
        let cond = m.cond().to_vec().unwrap()[0];
        assert!((f64::from(cond) / (want[0] / want[n - 1]) - 1.0).abs() < 2e-2, "n = {n}: cond {cond}");
    }
}

#[test]
fn svd_repeated_values_identity_and_trivial() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 5, 32, 40, 100] {
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        check_batch(&ctx, &id, n, n, 1, "identity");
        let s = mat(&ctx, &id, n, n, 1).singular_values().to_vec().unwrap();
        assert!(s.iter().all(|&x| (x - 1.0).abs() < 1e-6), "identity {n}: {s:?}");
        // Orthogonal times a scalar: all values equal.
        let q: Vec<f32> = orthogonal(n, 4).iter().map(|&x| (3.0 * x) as f32).collect();
        check_batch(&ctx, &q, n, n, 1, "3 Q");
        // A few repeated values.
        let spec: Vec<f64> = (0..n).map(|i| [4.0, 4.0, 2.0, 1.0][i % 4]).collect();
        check_batch(&ctx, &with_spectrum(n, n, &spec, 9), n, n, 1, "repeated");
    }
    let one = mat(&ctx, &[-3.5], 1, 1, 1).svd();
    assert_eq!(one.s.to_vec().unwrap(), vec![3.5]);
    let (u, vt) = (one.u.to_vec().unwrap()[0], one.vt.to_vec().unwrap()[0]);
    assert!((u * 3.5 * vt + 3.5).abs() < 1e-6);
}

#[test]
fn svd_scales() {
    let ctx = Context::new().unwrap();
    for &scale in &[1e-12f32, 1e-6, 1e6, 1e12, 1e18] {
        for &n in &[5usize, 40, 100] {
            let a: Vec<f32> = random(n, n, 6).iter().map(|&x| x * scale).collect();
            let s = mat(&ctx, &a, n, n, 1).singular_values().to_vec().unwrap();
            let unit = mat(&ctx, &random(n, n, 6), n, n, 1).singular_values().to_vec().unwrap();
            for j in 0..n {
                assert!((f64::from(s[j]) / f64::from(scale) - f64::from(unit[j])).abs() < 5e-5 * f64::from(unit[0]), "scale {scale}, n = {n}, value {j}: {} vs {}", s[j], unit[j] * scale);
            }
        }
    }
}

#[test]
fn svd_nan_and_inf_give_nan_and_terminate() {
    let ctx = Context::new().unwrap();
    for &(m, n) in &[(1usize, 1usize), (4, 4), (6, 3), (3, 6), (33, 33), (100, 100), (150, 90)] {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut a = random(m, n, 2);
            a[(m * n) / 2] = bad;
            let svd = mat(&ctx, &a, n, m, 1).svd();
            let s = svd.s.to_vec().unwrap();
            assert!(s.iter().any(|x| x.is_nan() || x.is_infinite()), "{m}x{n} with {bad}: {s:?}");
            let all = svd.u.to_vec().unwrap().iter().chain(svd.vt.to_vec().unwrap().iter()).any(|x| x.is_nan());
            assert!(all, "{m}x{n} with {bad}: no NaN in the vectors");
        }
    }
    // A bad matrix does not spoil the others of its batch.
    let mut data = Vec::new();
    for b in 0..5 {
        data.extend(random(6, 6, 30 + b));
    }
    data[2 * 36 + 7] = f32::NAN;
    let s = mat(&ctx, &data, 6, 6, 5).singular_values().to_vec().unwrap();
    for b in 0..5 {
        let bad = s[b * 6..(b + 1) * 6].iter().any(|x| x.is_nan());
        assert_eq!(bad, b == 2, "matrix {b}");
    }
}

fn pinv_checks(ctx: &Context, a: &[f32], m: usize, n: usize, batch: usize, label: &str) {
    let p = mat(ctx, a, n, m, batch).pinv();
    assert_eq!(p.shape().dims(), [m, n, batch, 1], "{label}: pinv shape");
    let p = p.to_vec().unwrap();
    for b in 0..batch {
        let am = widen(&a[b * m * n..(b + 1) * m * n]);
        let pm = widen(&p[b * m * n..(b + 1) * m * n]);
        let apa = mul(&mul(&am, &pm, m, n, m), &am, m, m, n);
        let pap = mul(&mul(&pm, &am, n, m, n), &pm, n, n, m);
        let e1 = diff(&apa, &am) / fro(&am);
        let e2 = diff(&pap, &pm) / fro(&pm);
        // The error grows with the condition number of the part of the matrix that is inverted.
        let sv = host_singular_values(&am, m, n);
        let cond = sv[0] / sv.iter().copied().filter(|&x| x > 1e-4 * sv[0]).fold(f64::INFINITY, f64::min);
        let tol = 3e-6 * cond.max(10.0);
        assert!(e1 < tol && e2 < tol, "{label} ({m}x{n}) matrix {b}: A P A: {e1}, P A P: {e2} (cond {cond})");
        // (A P) and (P A) are symmetric.
        let ap = mul(&am, &pm, m, n, m);
        let sym = diff(&ap, &transpose(&ap, m, m)) / fro(&ap);
        assert!(sym < tol, "{label} ({m}x{n}) matrix {b}: A P not symmetric: {sym}");
    }
}

#[test]
fn pinv_penrose_conditions() {
    let ctx = Context::new().unwrap();
    for &(m, n) in &[(1usize, 1usize), (3, 3), (6, 4), (4, 6), (20, 20), (50, 30), (30, 50), (100, 100), (150, 40)] {
        pinv_checks(&ctx, &random(m, n, 40 + (m + n) as u64), m, n, 1, "full rank");
    }
    // Rank deficient: the pseudo-inverse of a rank 3 matrix.
    for &(m, n) in &[(8usize, 8usize), (12, 7), (40, 40), (100, 90)] {
        pinv_checks(&ctx, &with_spectrum(m, n, &[3.0, 2.0, 1.0], 4), m, n, 1, "rank 3");
    }
    let mut batch = Vec::new();
    for b in 0..20 {
        batch.extend(random(6, 4, 77 + b));
    }
    pinv_checks(&ctx, &batch, 6, 4, 20, "batch");
    // Zero matrix: the zero pseudo-inverse, no NaN.
    let z = mat(&ctx, &[0.0; 12], 4, 3, 1).pinv().to_vec().unwrap();
    assert!(z.iter().all(|&x| x == 0.0), "{z:?}");
}

#[test]
fn pinv_solves_least_squares_and_tolerances() {
    let ctx = Context::new().unwrap();
    // pinv of a full column rank tall matrix gives the least squares solution.
    let (m, n) = (30usize, 5usize);
    let a = random(m, n, 3);
    let b = values(m, 4, -1.0, 1.0);
    let x = mat(&ctx, &a, n, m, 1).pinv().matmul_f32(&mat(&ctx, &b, 1, m, 1)).to_vec().unwrap();
    // Normal equations: Aᵀ (A x - b) = 0.
    let (a64, x64, b64) = (widen(&a), widen(&x), widen(&b));
    let r: Vec<f64> = mul(&a64, &x64, m, n, 1).iter().zip(&b64).map(|(p, q)| p - q).collect();
    let g = mul(&transpose(&a64, m, n), &r, n, m, 1);
    assert!(fro(&g) < 1e-4, "normal equations residual {}", fro(&g));
    // An explicit tolerance drops small singular values: diag(1, 1e-3) with tol 1e-2 has rank 1.
    let d = mat(&ctx, &[1.0, 0.0, 0.0, 1e-3], 2, 2, 1);
    assert_eq!(d.rank().to_vec().unwrap(), vec![2]);
    assert_eq!(d.rank_tol(1e-2).to_vec().unwrap(), vec![1]);
    let p = d.pinv_tol(1e-2).to_vec().unwrap();
    assert!((p[0] - 1.0).abs() < 1e-6 && p[3].abs() < 1e-9, "{p:?}");
    let p = d.pinv().to_vec().unwrap();
    assert!((p[3] - 1e3).abs() < 1e-2, "{p:?}");
}

#[test]
fn rank_of_batches_and_known_ranks() {
    let ctx = Context::new().unwrap();
    let mut data = Vec::new();
    let ranks = [1usize, 2, 3, 4, 4, 0];
    for (b, &r) in ranks.iter().enumerate() {
        let spec: Vec<f64> = (0..r).map(|i| 2.0 - 0.3 * i as f64).collect();
        data.extend(with_spectrum(4, 4, &spec, 50 + b as u64));
    }
    let got = mat(&ctx, &data, 4, 4, 6).rank();
    assert_eq!(got.shape().dims(), [1, 1, 6, 1]);
    assert_eq!(got.to_vec().unwrap(), vec![1, 2, 3, 4, 4, 0]);
    // A big matrix.
    let a = with_spectrum(120, 100, &(0..37).map(|i| 1.0 + i as f64).collect::<Vec<_>>(), 5);
    assert_eq!(mat(&ctx, &a, 100, 120, 1).rank().to_vec().unwrap(), vec![37]);
}

/// The rank of big matrices (decided by two QR factorisations when they can bound it, by the
/// singular values otherwise) is what counting the singular values above the cut gives.
#[test]
fn rank_of_big_matrices_matches_the_singular_values() {
    let ctx = Context::new().unwrap();
    let count = |s: &[f32], tol: f32| s.iter().filter(|&&x| x > tol * s[0]).count() as u32;
    let cases: Vec<(usize, usize, Vec<f64>, f32)> = vec![
        // Well separated from the cut: decided by the factorisations.
        (150, 150, (0..70).map(|i| 1.0 - 0.01 * i as f64).collect(), 1e-4),
        (150, 150, (0..150).map(|i| 1.0 - 0.006 * i as f64).collect(), 1e-4),
        (170, 100, (0..40).map(|i| 2.0 - 0.03 * i as f64).collect(), 1e-4),
        (100, 170, (0..40).map(|i| 2.0 - 0.03 * i as f64).collect(), 1e-4),
        // Values on both sides of the cut, a few times away from it.
        (130, 130, (0..60).map(|i| 1.0 - 0.01 * i as f64).chain((0..20).map(|i| 3e-3 * (1.0 - 0.01 * i as f64))).chain((0..20).map(|i| 3e-5 * (1.0 - 0.01 * i as f64))).collect(), 1e-4),
        // A decay that crosses the cut gradually.
        (120, 120, (0..120).map(|i| 0.9f64.powi(i)).collect(), 1e-4),
    ];
    for (i, (m, n, spec, tol)) in cases.iter().enumerate() {
        let a = with_spectrum(*m, *n, spec, 300 + i as u64);
        let arr = mat(&ctx, &a, *n, *m, 1);
        let want = count(&arr.singular_values().to_vec().unwrap(), *tol);
        assert_eq!(arr.rank_tol(*tol).to_vec().unwrap(), vec![want], "case {i}: {m}x{n}");
        let expected = spec.iter().filter(|&&x| x > f64::from(*tol) * spec[0]).count() as u32;
        assert_eq!(want, expected, "case {i}: the host count and the singular values differ");
    }
    // A random full rank matrix at the default tolerance, and one with a zero first column.
    let a = random(140, 140, 77);
    let arr = mat(&ctx, &a, 140, 140, 1);
    let tol = 140.0 * f32::EPSILON;
    assert_eq!(arr.rank().to_vec().unwrap(), vec![count(&arr.singular_values().to_vec().unwrap(), tol)]);
    let mut b = random(110, 110, 78);
    for r in 0..110 {
        b[r * 110] = 0.0;
    }
    assert_eq!(mat(&ctx, &b, 110, 110, 1).rank().to_vec().unwrap(), vec![109]);
    // A shift matrix (ones above the diagonal): the first QR factorisation sees a zero column.
    let mut shift = vec![0.0f32; 100 * 100];
    for r in 0..99 {
        shift[r * 100 + r + 1] = 1.0;
    }
    assert_eq!(mat(&ctx, &shift, 100, 100, 1).rank().to_vec().unwrap(), vec![99]);
    // Zero, and a batch of big matrices of different ranks.
    assert_eq!(mat(&ctx, &vec![0.0; 100 * 100], 100, 100, 1).rank().to_vec().unwrap(), vec![0]);
    let mut data = Vec::new();
    for (b, r) in [30usize, 100, 55].into_iter().enumerate() {
        let spec: Vec<f64> = (0..r).map(|i| 1.0 - 0.008 * i as f64).collect();
        data.extend(with_spectrum(110, 100, &spec, 400 + b as u64));
    }
    let got = mat(&ctx, &data, 100, 110, 3).rank();
    assert_eq!(got.shape().dims(), [1, 1, 3, 1]);
    assert_eq!(got.to_vec().unwrap(), vec![30, 100, 55]);
}

#[test]
fn svd_shape_errors_poison() {
    let ctx = Context::new().unwrap();
    let a = Array::<f32>::from_slice_on(&ctx, &[1.0; 16], [2, 2, 2, 2]).unwrap();
    assert!(a.svd().s.to_vec().is_err());
    assert!(a.svd().u.to_vec().is_err());
    assert!(a.singular_values().to_vec().is_err());
    assert!(a.pinv().to_vec().is_err());
    assert!(a.rank().to_vec().is_err());
}

#[test]
fn svd_results_are_lazy_values_that_compose() {
    let ctx = Context::new().unwrap();
    // U S Vt through matmul, the way a user reconstructs.
    let (m, n) = (24usize, 10usize);
    let a = random(m, n, 8);
    let am = mat(&ctx, &a, n, m, 1);
    let svd = am.svd();
    let us = (&svd.u * &svd.s.reshape([n, 1])).cache();
    let rec = us.matmul_f32(&svd.vt).to_vec().unwrap();
    let e = diff(&widen(&rec), &widen(&a)) / fro(&widen(&a));
    assert!(e < 1e-5, "{e}");
    // The input is not modified.
    assert_eq!(am.to_vec().unwrap(), a);
}
