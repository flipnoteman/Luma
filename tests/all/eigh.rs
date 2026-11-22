//! `eigh` and `eigvalsh` against an `f64` host Jacobi eigensolver: sizes around the workgroup-kernel
//! and global-memory thresholds, batches, positive / negative / indefinite / singular spectra,
//! repeated eigenvalues, only the lower triangle read, `NaN` input and shape errors. Checks:
//! ascending values within a few `eps ||A||` of the reference, the residual `||A V - V L||`,
//! orthonormality of `V`.

use crate::common::values;
use crate::dense_host::*;
use crate::svd::{mat, orthogonal};
use luma::{Array, Context};

const SIZES: [usize; 13] = [1, 2, 3, 4, 5, 7, 16, 31, 32, 33, 64, 100, 257];

/// Eigenvalues (ascending) of the symmetric row-major `n x n` `a` by cyclic Jacobi in `f64`.
fn host_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    for _ in 0..60 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-28 * fro(&a).powi(2) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * x - s * y;
                    a[k * n + q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[q * n + k] = s * x + c * y;
                }
            }
        }
    }
    let mut l: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    l.sort_by(f64::total_cmp);
    l
}

/// An exactly symmetric matrix from a random one (`(M + Mᵀ) / 2`).
fn symmetric(n: usize, seed: u64) -> Vec<f32> {
    let m = values(n * n, seed, -1.0, 1.0);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = 0.5 * (m[i * n + j] + m[j * n + i]);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

/// `Q diag(l) Qᵀ`, exactly symmetric after rounding to `f32`.
fn with_eigenvalues(l: &[f64], seed: u64) -> Vec<f32> {
    let n = l.len();
    let q = orthogonal(n, seed);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = (0..n).map(|p| q[i * n + p] * l[p] * q[j * n + p]).sum::<f64>() as f32;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

/// What an eigendecomposition of one matrix got wrong.
#[derive(Debug)]
struct Errors {
    /// `max |l - l_ref| / ||A||_F`.
    values: f64,
    /// `||A V - V diag(l)||_F / ||A||_F`.
    residual: f64,
    /// `||Vᵀ V - I||_F`.
    orth: f64,
    sorted: bool,
}

fn errors(a: &[f32], n: usize, l: &[f32], v: &[f32]) -> Errors {
    let (a64, v64) = (widen(a), widen(v));
    let norm = fro(&a64).max(f64::MIN_POSITIVE);
    let want = host_eigenvalues(&a64, n);
    let values = (0..n).map(|i| (f64::from(l[i]) - want[i]).abs()).fold(0.0, f64::max) / norm;
    let av = mul(&a64, &v64, n, n, n);
    let mut vl = v64.clone();
    for i in 0..n {
        for j in 0..n {
            vl[i * n + j] *= f64::from(l[j]);
        }
    }
    let residual = diff(&av, &vl) / norm;
    let vtv = mul(&transpose(&v64, n, n), &v64, n, n, n);
    let mut orth = 0.0;
    for i in 0..n {
        for j in 0..n {
            let d = vtv[i * n + j] - if i == j { 1.0 } else { 0.0 };
            orth += d * d;
        }
    }
    Errors { values, residual, orth: f64::sqrt(orth), sorted: l.windows(2).all(|w| w[0] <= w[1]) }
}

fn check(ctx: &Context, data: &[f32], n: usize, batch: usize, label: &str) {
    let a = mat(ctx, data, n, n, batch);
    let e = a.eigh();
    assert_eq!(e.values.shape().dims(), [n, 1, batch, 1], "{label}: values shape");
    assert_eq!(e.vectors.shape().dims(), [n, n, batch, 1], "{label}: vectors shape");
    let (l, v) = (e.values.to_vec().unwrap(), e.vectors.to_vec().unwrap());
    let only = a.eigvalsh().to_vec().unwrap();
    let tol = 4e-6 * (n as f64).sqrt().max(2.0);
    let orth = 4e-7 * (n as f64).max(20.0);
    for b in 0..batch {
        let m = &data[b * n * n..(b + 1) * n * n];
        let e = errors(m, n, &l[b * n..(b + 1) * n], &v[b * n * n..(b + 1) * n * n]);
        if b == 0 {
            eprintln!("{label} {n}: {e:?}");
        }
        assert!(e.values < tol && e.residual < tol && e.orth < orth && e.sorted, "{label} (n = {n}) matrix {b}: {e:?}");
        let scale = fro(&widen(m)).max(1e-30);
        for i in 0..n {
            assert!((f64::from(only[b * n + i]) - f64::from(l[b * n + i])).abs() / scale < tol, "{label} (n = {n}) matrix {b}: eigvalsh differs at {i}");
        }
    }
}

#[test]
fn eigh_every_size() {
    let ctx = Context::new().unwrap();
    for &n in &SIZES {
        check(&ctx, &symmetric(n, 10 + n as u64), n, 1, "random");
    }
}

#[test]
fn eigh_batches() {
    let ctx = Context::new().unwrap();
    for &(n, batch) in &[(1usize, 9usize), (2, 100), (4, 1), (4, 3), (4, 100), (4, 5000), (8, 700), (16, 100), (32, 40), (20, 7), (33, 3), (70, 2), (100, 3)] {
        let mut data = Vec::new();
        for b in 0..batch {
            data.extend(symmetric(n, 5 + (n * 7 + b * 13) as u64));
        }
        check(&ctx, &data, n, batch, "batch");
    }
}

#[test]
fn eigh_definite_indefinite_and_singular() {
    let ctx = Context::new().unwrap();
    for &n in &[2usize, 5, 16, 31, 40, 100] {
        let k = n as f64;
        // Positive definite, negative definite, indefinite, singular (zero eigenvalues), and a rank-1 negative one
        // (the shifted matrix is singular).
        let pos: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 / k).collect();
        let neg: Vec<f64> = pos.iter().map(|x| -x).collect();
        let ind: Vec<f64> = (0..n).map(|i| (i as f64 - k / 2.0) / 3.0).collect();
        let sing: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 2.0 + i as f64 }).collect();
        let mut rank1 = vec![0.0; n];
        rank1[0] = -5.0;
        for (name, l) in [("positive", &pos), ("negative", &neg), ("indefinite", &ind), ("singular", &sing), ("rank-1 negative", &rank1)] {
            check(&ctx, &with_eigenvalues(l, 3 + n as u64), n, 1, name);
        }
    }
}

#[test]
fn eigh_repeated_zero_identity_and_ill_conditioned() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 2, 4, 9, 32, 33, 100] {
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        check(&ctx, &id, n, 1, "identity");
        check(&ctx, &vec![0.0; n * n], n, 1, "zero");
        let rep: Vec<f64> = (0..n).map(|i| [2.0, 2.0, -1.0, 2.0, 0.5][i % 5]).collect();
        check(&ctx, &with_eigenvalues(&rep, 8), n, 1, "repeated");
        let wide: Vec<f64> = (0..n).map(|i| (if i % 2 == 0 { 1.0 } else { -1.0 }) * 10f64.powf(-6.0 * i as f64 / (n - 1).max(1) as f64)).collect();
        check(&ctx, &with_eigenvalues(&wide, 12), n, 1, "graded");
    }
    // Known eigenvalues of [[2, 1], [1, 2]] and a diagonal matrix in the wrong order.
    let e = mat(&ctx, &[2.0, 1.0, 1.0, 2.0], 2, 2, 1).eigh().values.to_vec().unwrap();
    assert!((e[0] - 1.0).abs() < 1e-6 && (e[1] - 3.0).abs() < 1e-6, "{e:?}");
    let e = mat(&ctx, &[5.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0], 3, 3, 1).eigh().values.to_vec().unwrap();
    assert!((e[0] + 2.0).abs() < 1e-6 && (e[1] - 1.0).abs() < 1e-6 && (e[2] - 5.0).abs() < 1e-6, "{e:?}");
}

#[test]
fn eigh_reads_only_the_lower_triangle() {
    let ctx = Context::new().unwrap();
    for &n in &[3usize, 16, 40, 100] {
        let a = symmetric(n, 21);
        let mut junk = a.clone();
        for i in 0..n {
            for j in i + 1..n {
                junk[i * n + j] = if (i + j) % 2 == 0 { f32::NAN } else { 1e6 };
            }
        }
        let good = mat(&ctx, &a, n, n, 1).eigh();
        let bad = mat(&ctx, &junk, n, n, 1).eigh();
        assert_eq!(good.values.to_vec().unwrap(), bad.values.to_vec().unwrap(), "n = {n}: values");
        let (gv, bv) = (good.vectors.to_vec().unwrap(), bad.vectors.to_vec().unwrap());
        assert_eq!(gv, bv, "n = {n}: vectors");
    }
}

#[test]
fn eigh_scales() {
    let ctx = Context::new().unwrap();
    for &scale in &[1e-12f32, 1e-6, 1e6, 1e12, 1e18] {
        for &n in &[5usize, 40, 100] {
            let a: Vec<f32> = symmetric(n, 6).iter().map(|&x| x * scale).collect();
            let l = mat(&ctx, &a, n, n, 1).eigvalsh().to_vec().unwrap();
            let unit = mat(&ctx, &symmetric(n, 6), n, n, 1).eigvalsh().to_vec().unwrap();
            let norm = fro(&widen(&symmetric(n, 6)));
            for j in 0..n {
                assert!((f64::from(l[j]) / f64::from(scale) - f64::from(unit[j])).abs() < 2e-6 * norm, "scale {scale}, n = {n}, value {j}: {} vs {}", l[j], unit[j] * scale);
            }
        }
    }
}

#[test]
fn eigh_nan_gives_nan_and_terminates() {
    let ctx = Context::new().unwrap();
    for &n in &[1usize, 4, 33, 100] {
        let mut a = symmetric(n, 2);
        a[(n * n) / 2 + n / 2 - (n / 2) % 2] = f32::NAN;
        // The NaN has to be in the lower triangle to be read.
        a[(n - 1) * n] = f32::NAN;
        let e = mat(&ctx, &a, n, n, 1).eigh();
        assert!(e.values.to_vec().unwrap().iter().any(|x| x.is_nan()), "n = {n}");
    }
}

#[test]
fn eigh_shape_errors_poison() {
    let ctx = Context::new().unwrap();
    let wide = Array::<f32>::from_slice_on(&ctx, &[1.0; 6], [3, 2]).unwrap();
    assert!(wide.eigh().values.to_vec().is_err());
    assert!(wide.eigh().vectors.to_vec().is_err());
    assert!(wide.eigvalsh().to_vec().is_err());
    let four = Array::<f32>::from_slice_on(&ctx, &[1.0; 16], [2, 2, 2, 2]).unwrap();
    assert!(four.eigh().values.to_vec().is_err());
}
