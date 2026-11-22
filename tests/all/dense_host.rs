//! Host `f64` reference routines shared by the dense linear algebra tests (`cholesky`, `lu`,
//! `qr`, `svd`, ...). Matrices are row-major `rows x cols` slices, as luma stores `[cols, rows]`.

#![allow(dead_code)]

use crate::common::values;

/// `C = A B` in `f64`, `A` `m x k`, `B` `k x n`.
pub fn mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let x = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += x * b[p * n + j];
            }
        }
    }
    c
}

pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

pub fn widen(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&x| f64::from(x)).collect()
}

/// Frobenius norm.
pub fn fro(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `||a - b||_F`.
pub fn diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// A random symmetric positive definite `n x n` matrix `M Mᵀ / n + shift I`, in `f32` precision
/// (the matrix the device sees is exactly the returned values).
pub fn spd(n: usize, seed: u64, shift: f32) -> Vec<f32> {
    let m = values(n * n, seed, -1.0, 1.0);
    let m = widen(&m);
    let mt = transpose(&m, n, n);
    let g = mul(&m, &mt, n, n, n);
    let mut a: Vec<f32> = g.iter().map(|&x| (x / n as f64) as f32).collect();
    for i in 0..n {
        a[i * n + i] += shift;
    }
    // Exactly symmetric.
    for i in 0..n {
        for j in 0..i {
            a[j * n + i] = a[i * n + j];
        }
    }
    a
}

/// The Kac-Murdock-Szego matrix `rho^|i-j|`: SPD with condition number about `((1 + rho) / (1 - rho))^2`.
pub fn kms(n: usize, rho: f64) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = rho.powi((i as i32 - j as i32).abs()) as f32;
        }
    }
    a
}

/// Cholesky in `f64` of the lower triangle; `None` if a pivot is not positive.
pub fn chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 0.0 {
            return None;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solves `A X = B` in `f64` by Gaussian elimination with partial pivoting (`B` is `n x k`).
pub fn solve(a: &[f64], b: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs())).unwrap();
        if p != c {
            for j in 0..n {
                a.swap(c * n + j, p * n + j);
            }
            for j in 0..k {
                x.swap(c * k + j, p * k + j);
            }
        }
        for i in c + 1..n {
            let f = a[i * n + c] / a[c * n + c];
            if f != 0.0 {
                for j in c..n {
                    a[i * n + j] -= f * a[c * n + j];
                }
                for j in 0..k {
                    x[i * k + j] -= f * x[c * k + j];
                }
            }
        }
    }
    for c in (0..n).rev() {
        for j in 0..k {
            let mut s = x[c * k + j];
            for p in c + 1..n {
                s -= a[c * n + p] * x[p * k + j];
            }
            x[c * k + j] = s / a[c * n + c];
        }
    }
    x
}

/// `||A X - B||_F / (||A||_F ||X||_F)`: the backward error of a solution.
pub fn residual(a: &[f64], x: &[f64], b: &[f64], n: usize, k: usize) -> f64 {
    let ax = mul(a, x, n, n, k);
    diff(&ax, b) / (fro(a) * fro(x)).max(f64::MIN_POSITIVE)
}
