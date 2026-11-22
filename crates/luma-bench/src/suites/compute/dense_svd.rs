//! Dense linear algebra, continued: the SVD and the symmetric eigenproblem (one-sided Jacobi), one
//! big matrix and batches of small ones. Every case is verified by backward error against `f64`
//! host matrix-vector products with a random vector (`O(n^2)`) before it is timed.

use super::{host, pick, upload};
use luma::eval_all;
use crate::harness::*;

fn widen(a: &[f32]) -> Vec<f64> {
    a.iter().map(|&x| f64::from(x)).collect()
}

/// `M z` for the `rows x cols` row-major matrix `m`.
fn matvec(m: &[f64], rows: usize, cols: usize, z: &[f64]) -> Vec<f64> {
    (0..rows).map(|i| (0..cols).map(|j| m[i * cols + j] * z[j]).sum()).collect()
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300)
}

/// A random symmetric `n x n` matrix (both triangles filled).
fn symmetric(n: usize, seed: u64) -> Vec<f32> {
    let r = values(n * n, seed, -1.0, 1.0);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            a[i * n + j] = r[i * n + j];
            a[j * n + i] = r[i * n + j];
        }
    }
    a
}

/// Errors of one `n x n` SVD from the vector `z`: `||A z - U S Vt z|| / ||A z||` and the loss of
/// orthogonality of `U` and `V` on a vector.
fn svd_error(a: &[f32], u: &[f32], s: &[f32], vt: &[f32], n: usize) -> (f64, f64, f64) {
    let z: Vec<f64> = widen(&values(n, 5, -1.0, 1.0));
    let (a, u, vt) = (widen(a), widen(u), widen(vt));
    let sz: Vec<f64> = matvec(&vt, n, n, &z).iter().zip(s).map(|(x, &s)| x * f64::from(s)).collect();
    let back = rel_diff(&matvec(&u, n, n, &sz), &matvec(&a, n, n, &z));
    // Uᵀ U z = z and V Vᵀ z = z.
    let ut: Vec<f64> = (0..n * n).map(|ij| u[(ij % n) * n + ij / n]).collect();
    let orth_u = rel_diff(&matvec(&ut, n, n, &matvec(&u, n, n, &z)), &z);
    let v: Vec<f64> = (0..n * n).map(|ij| vt[(ij % n) * n + ij / n]).collect();
    let orth_v = rel_diff(&matvec(&v, n, n, &matvec(&vt, n, n, &z)), &z);
    (back, orth_u, orth_v)
}

/// The worst `||A v_k - l_k v_k|| / ||A||_F` over a few eigenpairs, and `||Vᵀ V z - z||`.
fn eigh_error(a: &[f32], l: &[f32], v: &[f32], n: usize) -> (f64, f64) {
    let (a64, v64) = (widen(a), widen(v));
    let norm: f64 = a64.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut worst = 0.0f64;
    for k in [0, n / 2, n - 1] {
        let col: Vec<f64> = (0..n).map(|i| v64[i * n + k]).collect();
        let av = matvec(&a64, n, n, &col);
        let r: f64 = av.iter().zip(&col).map(|(x, y)| (x - f64::from(l[k]) * y).powi(2)).sum::<f64>().sqrt();
        worst = worst.max(r / norm);
    }
    let z: Vec<f64> = widen(&values(n, 5, -1.0, 1.0));
    let vt: Vec<f64> = (0..n * n).map(|ij| v64[(ij % n) * n + ij / n]).collect();
    (worst, rel_diff(&matvec(&vt, n, n, &matvec(&v64, n, n, &z)), &z))
}

pub fn run(h: &mut Harness) {
    h.guarded("dense_svd", |h| {
        let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[32, 64], &[32, 64, 128]) } else { pick(h, &[64, 256], &[64, 256, 1024]) };
        for &n in sizes {
            let a = values(n * n, 120 + n as u64, -1.0, 1.0);
            let al = upload(h, &a, [n, n]);
            let svd = al.svd();
            let (u, s, vt) = (host(&svd.u), host(&svd.s), host(&svd.vt));
            let (back, ou, ov) = svd_error(&a, &u, &s, &vt, n);
            assert!(back < 2e-5 && ou < 1e-4 && ov < 1e-4, "svd {n}: backward error {back}, orthogonality {ou} / {ov}");
            // Nominal flops of a full SVD, 22 n^3, so that rates compare with other libraries'.
            let flops = 22.0 * (n as f64).powi(3);
            h.case("dense/svd", &format!("{n}x{n}"), Work::Flops(flops), &mut || {
                let r = al.svd();
                eval_all(&[&r.u, &r.s, &r.vt]).unwrap()
            });
            let sv = host(&al.singular_values());
            let top = f64::from(s[0]);
            assert!(sv.iter().zip(&s).all(|(x, y)| (f64::from(x - y)).abs() < 1e-5 * top), "singular_values {n} differ from svd");
            h.case("dense/svd_values", &format!("{n}x{n}"), Work::Flops(flops / 2.0), &mut || al.singular_values().eval().unwrap());

            let sym = symmetric(n, 130 + n as u64);
            let sl = upload(h, &sym, [n, n]);
            let e = sl.eigh();
            let (l, v) = (host(&e.values), host(&e.vectors));
            let (res, orth) = eigh_error(&sym, &l, &v, n);
            assert!(res < 2e-5 && orth < 1e-4 && l.windows(2).all(|w| w[0] <= w[1]), "eigh {n}: residual {res}, orthogonality {orth}");
            h.case("dense/eigh", &format!("{n}x{n}"), Work::Flops(flops), &mut || {
                let e = sl.eigh();
                eval_all(&[&e.values, &e.vectors]).unwrap()
            });
        }
        let batches: &[(usize, usize)] = pick(h, &[(8, 1024), (32, 256)], &[(8, 4096), (16, 4096), (32, 4096)]);
        for &(n, batch) in batches {
            let mut data = Vec::with_capacity(batch * n * n);
            let mut sym = Vec::with_capacity(batch * n * n);
            for b in 0..batch {
                data.extend(values(n * n, 7 + b as u64, -1.0, 1.0));
                sym.extend(symmetric(n, 9 + b as u64));
            }
            let al = upload(h, &data, [n, n, batch]);
            let svd = al.svd();
            let (u, s, vt) = (host(&svd.u), host(&svd.s), host(&svd.vt));
            for b in [0, batch / 2, batch - 1] {
                let r = b * n * n..(b + 1) * n * n;
                let (back, ou, ov) = svd_error(&data[r.clone()], &u[r.clone()], &s[b * n..(b + 1) * n], &vt[r], n);
                assert!(back < 2e-5 && ou < 1e-4 && ov < 1e-4, "batched svd {n}x{batch}: matrix {b}: {back}, {ou}, {ov}");
            }
            let flops = 22.0 * (n as f64).powi(3) * batch as f64;
            h.case("dense/svd_batched", &format!("{batch} x {n}x{n}"), Work::Flops(flops), &mut || {
                let r = al.svd();
                eval_all(&[&r.u, &r.s, &r.vt]).unwrap()
            });
            h.case("dense/svd_values_batched", &format!("{batch} x {n}x{n}"), Work::Flops(flops / 2.0), &mut || al.singular_values().eval().unwrap());
            let sl = upload(h, &sym, [n, n, batch]);
            let e = sl.eigh();
            let (l, v) = (host(&e.values), host(&e.vectors));
            for b in [0, batch / 2, batch - 1] {
                let r = b * n * n..(b + 1) * n * n;
                let (res, orth) = eigh_error(&sym[r.clone()], &l[b * n..(b + 1) * n], &v[r], n);
                assert!(res < 2e-5 && orth < 1e-4, "batched eigh {n}x{batch}: matrix {b}: {res}, {orth}");
            }
            h.case("dense/eigh_batched", &format!("{batch} x {n}x{n}"), Work::Flops(flops), &mut || {
                let e = sl.eigh();
                eval_all(&[&e.values, &e.vectors]).unwrap()
            });
        }
    });
}
