//! Dense linear algebra: Cholesky, triangular and SPD solves (single big matrices and batches of
//! small ones), verified by backward error against an `f64` host matrix-vector product with a
//! random vector (`O(n^2)`, so that n = 4096 verifies in milliseconds).

use super::{host, pick, upload};
use crate::harness::*;

/// A symmetric, strictly diagonally dominant (so positive definite) `n x n` matrix.
fn dd_spd(n: usize, seed: u64) -> Vec<f32> {
    let r = values(n * n, seed, -0.5, 0.5);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            a[i * n + j] = r[i * n + j];
            a[j * n + i] = r[i * n + j];
        }
        a[i * n + i] = n as f32 * 0.5 + 1.0 + r[i * n + i].abs();
    }
    a
}

/// `M z` for the `n x n` row-major matrix `get(i, j)`.
fn matvec(get: &dyn Fn(usize, usize) -> f64, n: usize, z: &[f64]) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| get(i, j) * z[j]).sum()).collect()
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    d / b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300)
}

/// `|| L Lᵀ z - A z || / || A z ||` for row-major `n x n` matrices.
fn chol_error(l: &[f32], a: &[f32], n: usize) -> f64 {
    let z: Vec<f64> = values(n, 5, -1.0, 1.0).iter().map(|&x| f64::from(x)).collect();
    let tri = |i: usize, j: usize| if j > i { 0.0 } else { f64::from(l[i * n + j]) };
    let lt_z = matvec(&|i, j| tri(j, i), n, &z);
    rel_diff(&matvec(&tri, n, &lt_z), &matvec(&|i, j| f64::from(a[i * n + j]), n, &z))
}

pub fn run(h: &mut Harness) {
    h.guarded("dense", |h| {
        let sizes: &[usize] = if h.ctx.is_cpu() { pick(h, &[128, 256], &[256, 512, 1024]) } else { pick(h, &[256, 1024], &[256, 1024, 4096]) };
        for &n in sizes {
            let a = dd_spd(n, 90 + n as u64);
            let al = upload(h, &a, [n, n]);
            let l = host(&al.cholesky());
            let err = chol_error(&l, &a, n);
            assert!(err < 2e-4, "cholesky {n}: backward error {err}");
            let flops = (n as f64).powi(3) / 3.0;
            h.case("dense/cholesky", &format!("{n}x{n}"), Work::Flops(flops), &mut || al.cholesky().eval().unwrap());

            let k = 16.min(n);
            let b = values(n * k, 91, -1.0, 1.0);
            let bl = upload(h, &b, [k, n]);
            let x = host(&al.solve_spd(&bl));
            let (xv, bv): (Vec<f64>, Vec<f64>) = (x.iter().map(|&v| f64::from(v)).collect(), b.iter().map(|&v| f64::from(v)).collect());
            let ax: Vec<f64> = (0..n * k).map(|ij| (0..n).map(|p| f64::from(a[(ij / k) * n + p]) * xv[p * k + ij % k]).sum()).collect();
            let r = rel_diff(&ax, &bv);
            assert!(r < 1e-4, "solve_spd {n}: residual {r}");
            h.case("dense/solve_spd", &format!("{n}x{n}, {k} rhs"), Work::Flops(flops + 2.0 * (n * n * k) as f64), &mut || al.solve_spd(&bl).eval().unwrap());
        }
        let batches: &[(usize, usize)] = pick(h, &[(8, 1024), (32, 256)], &[(8, 16384), (16, 4096), (32, 1024), (64, 256)]);
        for &(n, batch) in batches {
            let mut data = Vec::with_capacity(batch * n * n);
            for b in 0..batch {
                data.extend(dd_spd(n, 5 + b as u64));
            }
            let al = upload(h, &data, [n, n, batch]);
            let l = host(&al.cholesky());
            for b in [0, batch / 2, batch - 1] {
                let err = chol_error(&l[b * n * n..(b + 1) * n * n], &data[b * n * n..(b + 1) * n * n], n);
                assert!(err < 2e-4, "batched cholesky {n}x{batch}: matrix {b}: {err}");
            }
            let flops = (n as f64).powi(3) / 3.0 * batch as f64;
            h.case("dense/cholesky_batched", &format!("{batch} x {n}x{n}"), Work::Flops(flops), &mut || al.cholesky().eval().unwrap());
        }
    });
}
